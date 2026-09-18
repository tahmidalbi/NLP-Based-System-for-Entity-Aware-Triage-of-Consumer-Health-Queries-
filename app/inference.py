"""
Core inference pipeline for BanglaCare (guide section 17.1).

Loads the frozen Phase 10 checkpoint, the two LIVE FastText models (not the
precomputed training cache - a demo must handle words it has never seen
before, which is exactly what FastText's subword n-grams are for), and the
Phase 11 calibration, then turns one raw health query into entities +
severity + calibrated confidence.

Kept separate from app.py (the Gradio UI) so this can be tested or reused -
from a CLI, a notebook, or a different UI - without a Gradio dependency.

Inference sequence (guide 17.1, steps 16-24):
  normalize -> whitespace-tokenize -> 300D+300D FastText lookup -> shared
  encoder -> NER CRF decode -> entity-aware severity head -> temperature-
  scaled softmax -> calibrated confidence.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from featurizer import FastTextFeaturizer  # noqa: E402
from labels import NER_ID2LABEL, SEVERITY_ID2LABEL  # noqa: E402
from model import BanglaCareModel  # noqa: E402
from ner_eval import extract_spans  # noqa: E402
from severity_eval import softmax  # noqa: E402
from text_utils import normalize_text  # noqa: E402

MAX_LEN = 512  # same cap as scripts/dataset.py (guide 5.5)

DISCLAIMER = (
    "This system prioritizes messages; it does not diagnose or prescribe. "
    "For any genuine emergency, seek immediate medical care."
)


def _empty_result(error):
    return {
        "tokens": [], "entities": [], "severity": None, "severity_probs": None,
        "confidence": None, "needs_review": False, "disclaimer": DISCLAIMER,
        "error": error,
    }


class BanglaCarePipeline:
    """Loads everything once at construction; call .predict(text) repeatedly.

    Deliberately does not import scripts/train_utils.py: that module pulls in
    dataset.py/pandas for the training data loaders, which the demo does not
    need. Device selection is inlined instead to keep this path light.
    """

    def __init__(self, checkpoint_path, calibration_path, general_model_path,
                 medical_model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[BanglaCare] device: {self.device}")
        print(f"[BanglaCare] loading general FastText: {general_model_path}")
        print(f"[BanglaCare] loading medical FastText: {medical_model_path}")
        print("[BanglaCare] (both models are large - this can take a minute)")
        self.featurizer = FastTextFeaturizer(
            general_model_path=general_model_path,
            medical_model_path=medical_model_path,
        )

        print(f"[BanglaCare] loading model checkpoint: {checkpoint_path}")
        self.model = BanglaCareModel().to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        self.temperature = calibration["temperature"]
        self.review_threshold = calibration["review_threshold"]
        print(
            f"[BanglaCare] ready - temperature={self.temperature:.4f} "
            f"review_threshold={self.review_threshold:.4f}"
        )

    @torch.no_grad()
    def predict(self, text):
        """One free-form health query -> dict with entities, severity, confidence.

        Never raises on ordinary bad input (empty string, whitespace-only,
        unsupported script) - those come back as a result dict with an
        "error" message instead, so the UI layer never has to wrap this in
        its own try/except for the common cases.
        """
        text = (text or "").strip()
        if not text:
            return _empty_result("Please enter a health query.")

        normalized = normalize_text(text)
        tokens = normalized.split()[:MAX_LEN]
        if not tokens:
            return _empty_result("No usable text after cleaning - try rephrasing.")

        vectors = self.featurizer.batch_vectors(tokens)  # (T, 600); OOV-safe via subwords
        features = torch.from_numpy(vectors).unsqueeze(0).to(self.device)  # (1, T, 600)
        lengths = torch.tensor([len(tokens)])
        mask = torch.ones(1, len(tokens), dtype=torch.bool, device=self.device)

        # NER branch (guide 17.1 steps 20-21: BiLSTM -> CRF decode -> spans)
        emissions, _ = self.model.ner_forward(features, lengths, mask)
        decoded_ids = self.model.ner_head.decode(emissions, mask)[0]  # one example
        ner_labels = [NER_ID2LABEL[i] for i in decoded_ids]
        spans = extract_spans(ner_labels)
        entities = [
            {"type": etype, "text": " ".join(tokens[start:end]), "start": start, "end": end}
            for etype, start, end in sorted(spans, key=lambda s: s[1])
        ]

        # Severity branch (guide 17.1 steps 22-24: entity-aware attention +
        # max-pool -> logits -> temperature-scaled softmax -> confidence)
        severity_logits, _ = self.model.severity_forward(features, lengths, mask)
        probs = softmax(severity_logits.cpu().numpy(), self.temperature)[0]
        pred_id = int(probs.argmax())
        confidence = float(probs[pred_id])

        return {
            "tokens": tokens,
            "entities": entities,
            "severity": SEVERITY_ID2LABEL[pred_id],
            "severity_probs": {SEVERITY_ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
            "confidence": confidence,
            "needs_review": confidence < self.review_threshold,
            "disclaimer": DISCLAIMER,
            "error": None,
        }
