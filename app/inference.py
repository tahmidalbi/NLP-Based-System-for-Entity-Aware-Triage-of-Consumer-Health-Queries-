"""
Core inference pipelines for BanglaCare (guide section 17.1).

Two architectures, one interface. Both expose `.predict(text)` returning the
same dict shape, so the UI never branches on which model is loaded:

  BiLSTMPipeline       frozen FastText (600D) + BiLSTM  ~2.1M trainable params
  TransformerPipeline  fine-tuned BanglaBERT           ~111M params

Both load the LIVE FastText models / the live tokenizer rather than the
precomputed training cache - a demo must handle words it has never seen
before, which is what FastText subwords and WordPiece are for.

Inference sequence (guide 17.1, steps 16-24) is identical for both:
  normalize -> tokenize -> encode -> shared encoder -> NER CRF decode
  -> entity-aware severity head -> temperature-scaled softmax -> confidence.

Kept free of any Gradio import so it can be driven from a CLI or a notebook.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from labels import NER_ID2LABEL, SEVERITY_ID2LABEL  # noqa: E402
from ner_eval import extract_spans  # noqa: E402
from severity_eval import softmax  # noqa: E402
from text_utils import normalize_text  # noqa: E402

MAX_LEN = 512  # same cap as scripts/dataset.py and dataset_transformer.py

DISCLAIMER = (
    "This system prioritizes messages; it does not diagnose or prescribe. "
    "For any genuine emergency, seek immediate medical care."
)

DEFAULTS = {
    "bilstm": {
        "checkpoint": ROOT / "checkpoints" / "best_joint.pt",
        "calibration": ROOT / "checkpoints" / "calibration.json",
        "label": "BiLSTM + FastText",
    },
    "transformer": {
        "checkpoint": ROOT / "checkpoints" / "best_joint_transformer.pt",
        "calibration": ROOT / "checkpoints" / "calibration_transformer.json",
        "label": "BanglaBERT",
    },
}


def _empty_result(error, arch=None):
    return {
        "arch": arch, "tokens": [], "entities": [], "severity": None,
        "severity_probs": None, "confidence": None, "needs_review": False,
        "disclaimer": DISCLAIMER, "error": error,
    }


class BasePipeline:
    """Shared plumbing: calibration, tokenization, span extraction, packaging.

    Subclasses implement _build() and _forward(); everything else - including
    the guarantee that both architectures return byte-identical result shapes -
    lives here so the two paths cannot drift apart.
    """

    arch = None

    def __init__(self, checkpoint_path=None, calibration_path=None, device=None, **kwargs):
        cfg = DEFAULTS[self.arch]
        self.checkpoint_path = Path(checkpoint_path or cfg["checkpoint"])
        self.calibration_path = Path(calibration_path or cfg["calibration"])
        self.label = cfg["label"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for path, what in [(self.checkpoint_path, "checkpoint"),
                           (self.calibration_path, "calibration")]:
            if not path.exists():
                raise SystemExit(f"Missing {self.arch} {what} at {path}")

        print(f"[BanglaCare:{self.arch}] device: {self.device}")
        self._build(**kwargs)

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())

        calibration = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        self.temperature = calibration["temperature"]
        self.review_threshold = calibration["review_threshold"]
        print(
            f"[BanglaCare:{self.arch}] ready - {self.n_params:,} params, "
            f"T={self.temperature:.4f}, review<{self.review_threshold:.4f}"
        )

    # ------------------------------------------------------------ subclass API
    def _build(self, **kwargs):
        raise NotImplementedError

    def _forward(self, tokens):
        """tokens -> (ner_label_ids, severity_logits_1d, surviving_tokens)."""
        raise NotImplementedError

    # ------------------------------------------------------------ shared
    @torch.no_grad()
    def predict(self, text):
        text = (text or "").strip()
        if not text:
            return _empty_result("Please enter a health query.", self.arch)

        # Same conservative normalization the training data went through
        # (Phase 3), so the demo sees the distribution the model was fit on.
        tokens = normalize_text(text).split()[:MAX_LEN]
        if not tokens:
            return _empty_result("No usable text after cleaning - try rephrasing.", self.arch)

        ner_label_ids, severity_logits, tokens = self._forward(tokens)

        labels = [NER_ID2LABEL[i] for i in ner_label_ids]
        entities = [
            {"type": etype, "text": " ".join(tokens[start:end]), "start": start, "end": end}
            for etype, start, end in sorted(extract_spans(labels), key=lambda s: s[1])
        ]

        probs = softmax(severity_logits.reshape(1, -1), self.temperature)[0]
        pred_id = int(probs.argmax())
        confidence = float(probs[pred_id])

        return {
            "arch": self.arch,
            "arch_label": self.label,
            "n_params": self.n_params,
            "tokens": tokens,
            "entities": entities,
            "severity": SEVERITY_ID2LABEL[pred_id],
            "severity_probs": {SEVERITY_ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
            "confidence": confidence,
            "needs_review": confidence < self.review_threshold,
            "disclaimer": DISCLAIMER,
            "error": None,
        }


class BiLSTMPipeline(BasePipeline):
    """Frozen FastText 600D -> projection -> 2-layer BiLSTM -> CRF + severity."""

    arch = "bilstm"

    def _build(self, general_model_path=None, medical_model_path=None, **_):
        from featurizer import FastTextFeaturizer
        from model import BanglaCareModel

        general = Path(general_model_path or ROOT / "embeddings" / "cc.bn.300.bin")
        medical = Path(medical_model_path or ROOT / "embeddings" / "medical_fasttext.bin")
        for path, what in [(general, "general FastText"), (medical, "medical FastText")]:
            if not path.exists():
                raise SystemExit(f"Missing {what} model at {path}")

        print(f"[BanglaCare:bilstm] loading FastText models (~8GB, takes a minute)")
        self.featurizer = FastTextFeaturizer(
            general_model_path=general, medical_model_path=medical
        )
        self.model = BanglaCareModel().to(self.device)

    def _forward(self, tokens):
        vectors = self.featurizer.batch_vectors(tokens)          # (T, 600), OOV-safe
        features = torch.from_numpy(vectors).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(tokens)])
        mask = torch.ones(1, len(tokens), dtype=torch.bool, device=self.device)

        emissions, _ = self.model.ner_forward(features, lengths, mask)
        decoded = self.model.ner_head.decode(emissions, mask)[0]

        logits, _ = self.model.severity_forward(features, lengths, mask)
        return decoded, logits.float().cpu().numpy()[0], tokens


class TransformerPipeline(BasePipeline):
    """Fine-tuned BanglaBERT. Heads are identical to the BiLSTM variant; only
    the encoder differs (see scripts/model_transformer.py)."""

    arch = "transformer"

    def _build(self, model_name=None, no_normalize=False, **_):
        from dataset_transformer import (
            DEFAULT_MODEL,
            WordEncoder,
            load_tokenizer,
            make_normalizer,
        )
        from model_transformer import BanglaCareTransformer

        model_name = model_name or DEFAULT_MODEL
        print(f"[BanglaCare:transformer] loading tokenizer + config: {model_name}")
        self.encoder = WordEncoder(
            load_tokenizer(model_name), make_normalizer(not no_normalize)
        )
        # pretrained=False: the fine-tuned checkpoint supplies every weight, so
        # there is no reason to pull the pretrained ones off the hub first.
        self.model = BanglaCareTransformer(model_name, pretrained=False).to(self.device)

    def _forward(self, tokens):
        item = self.encoder.encode(tokens)
        n_words = item["n_words"]

        input_ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        first_pos = torch.tensor([item["first_pos"]], dtype=torch.long, device=self.device)
        word_mask = torch.ones(1, n_words, dtype=torch.bool, device=self.device)

        emissions, _ = self.model.ner_forward(input_ids, attention_mask, first_pos)
        decoded = self.model.ner_head.decode(emissions, word_mask)[0]

        logits, _ = self.model.severity_forward(
            input_ids, attention_mask, first_pos, word_mask
        )
        # item["tokens"] is the surviving prefix if the query was truncated.
        return decoded, logits.float().cpu().numpy()[0], item["tokens"]


PIPELINES = {"bilstm": BiLSTMPipeline, "transformer": TransformerPipeline}


class PipelineRegistry:
    """Lazily builds and caches pipelines.

    Loading is deferred because the two architectures are wildly different
    weights: the BiLSTM path needs ~8GB of FastText binaries in RAM, the
    transformer ~450MB. Building both at startup would make the demo slow to
    open and could exhaust memory on a laptop, so each is built only when a
    query actually asks for it.
    """

    def __init__(self, device=None, **build_kwargs):
        self.device = device
        self.build_kwargs = build_kwargs
        self._cache = {}

    def available(self):
        """Which architectures have all their files present on disk."""
        out = []
        for arch, cfg in DEFAULTS.items():
            if cfg["checkpoint"].exists() and cfg["calibration"].exists():
                out.append(arch)
        return out

    def is_loaded(self, arch):
        return arch in self._cache

    def get(self, arch):
        if arch not in self._cache:
            self._cache[arch] = PIPELINES[arch](device=self.device, **self.build_kwargs)
        return self._cache[arch]

    def predict(self, arch, text):
        return self.get(arch).predict(text)


# Backwards compatibility: the original single-architecture entry point.
BanglaCarePipeline = BiLSTMPipeline
