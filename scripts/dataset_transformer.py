"""
Transformer-variant data pipeline (TRANSFORMER_PLAN.md section 7).

Same cleaned Phase 3 tokens as the BiLSTM pipeline (dataset.py), but encoded
into subwords. Labels stay at WORD level: each word's state is taken from its
FIRST subword, gathered into a compact (B, W, H) tensor. This keeps the NER
decode, the CRF and the severity attention at the same granularity as the
BiLSTM run, and avoids the pytorch-crf requirement that mask[:, 0] be True
(which would break if [CLS] sat at position 0 and were masked out).

Every batch carries TWO padding lengths - subword (S) and word (W) - kept
separate on purpose.

Batch keys
----------
  tokens          list[list[str]]  original (un-normalized) surviving words
  input_ids       (B, S) long
  attention_mask  (B, S) long
  first_pos       (B, W) long      subword index of each word's first subword; padded with 0
  word_mask       (B, W) bool      True at real words
  lengths         (B,)   long      real word counts (after any truncation)
  ner_tags        (B, W) long      NER batches only
  severity_ids    (B,)   long      severity batches only
"""

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from labels import NER_LABEL2ID, SEVERITY_LABEL2ID

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "csebuetnlp/banglabert"
MAX_LEN = 512  # subword cap (BERT-family position limit, incl. [CLS]/[SEP])


def load_tokenizer(model_name=DEFAULT_MODEL):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.is_fast:
        raise SystemExit(
            f"{model_name} loaded a slow tokenizer; word_ids() (label alignment) "
            "needs a fast one. Install `tokenizers` and a recent `transformers`."
        )
    return tokenizer


def make_normalizer(enabled=True):
    """Per-TOKEN csebuetnlp normalizer (plan section 6).

    Applied to each word, never to the joined sentence, so the token count -
    and therefore the IOB label alignment - cannot change. Falls back to the
    original token if normalization empties it.
    """
    if not enabled:
        return lambda token: token
    try:
        from normalizer import normalize
    except ImportError:
        raise SystemExit(
            "csebuetnlp normalizer not installed:\n"
            "  pip install git+https://github.com/csebuetnlp/normalizer\n"
            "(or pass --no-normalize for a non-csebuetnlp encoder)"
        )

    def _normalize(token):
        return (normalize(token) or "").strip() or token

    return _normalize


class WordEncoder:
    """tokens (+ optional IOB labels) -> subword ids + word-level alignment."""

    def __init__(self, tokenizer, normalize_fn=None, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.normalize_fn = normalize_fn or (lambda t: t)
        self.max_len = max_len
        self._has_subwords = {}

    def _prepare(self, tokens):
        """Normalize per token; swap in [UNK] for any word that would yield zero
        subwords, because a word with no subword vanishes from word_ids() and
        would silently shift every label after it."""
        out = []
        for token in tokens:
            word = self.normalize_fn(token)
            ok = self._has_subwords.get(word)
            if ok is None:
                ok = len(self.tokenizer(word, add_special_tokens=False)["input_ids"]) > 0
                self._has_subwords[word] = ok
            out.append(word if ok else self.tokenizer.unk_token)
        assert len(out) == len(tokens)
        return out

    def encode(self, tokens, labels=None):
        assert len(tokens) > 0, "empty example"
        words = self._prepare(tokens)
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
        )

        first_pos, seen = [], set()
        for pos, wid in enumerate(enc.word_ids()):
            if wid is None or wid in seen:  # special token, or a continuation subword
                continue
            seen.add(wid)
            first_pos.append(pos)

        n_words = len(first_pos)
        assert n_words <= len(tokens)  # equal unless truncated
        assert seen == set(range(n_words)), "words must survive as a prefix, in order"

        item = {
            "input_ids": enc["input_ids"],
            "first_pos": first_pos,
            "n_words": n_words,
            "tokens": list(tokens[:n_words]),
            "truncated": n_words < len(tokens),
        }
        if labels is not None:
            assert len(labels) == len(tokens)
            item["ner_label_ids"] = [NER_LABEL2ID[l] for l in labels[:n_words]]
            assert len(item["ner_label_ids"]) == len(first_pos)
        return item


class TransformerNERDataset(Dataset):
    """HealthNER split, encoded once up front."""

    def __init__(self, split, encoder, limit=None):
        path = ROOT / "data" / "processed" / "healthner" / f"{split}.json"
        with open(path, encoding="utf-8") as f:
            examples = json.load(f)
        if limit is not None:
            examples = examples[:limit]
        self.items = [encoder.encode(ex["tokens"], ex["labels"]) for ex in examples]
        self.n_truncated = sum(it["truncated"] for it in self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class TransformerSeverityDataset(Dataset):
    """Severity split. Only Text and Categories are read - Action Needed is
    never loaded (guide 3.3 leakage rule), same as dataset.SeverityDataset."""

    def __init__(self, split, encoder, limit=None):
        path = ROOT / "data" / "processed" / f"severity_{split}.csv"
        df = pd.read_csv(path, usecols=["Text", "Categories"])
        if limit is not None:
            df = df.iloc[:limit]
        texts = df["Text"].astype(str).tolist()
        severity_ids = [SEVERITY_LABEL2ID[c] for c in df["Categories"].tolist()]
        self.items = []
        for text, sid in zip(texts, severity_ids):
            item = encoder.encode(text.split())
            item["severity_id"] = sid
            self.items.append(item)
        self.n_truncated = sum(it["truncated"] for it in self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class TransformerCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        bsz = len(batch)
        max_sub = max(len(it["input_ids"]) for it in batch)  # subword padding
        max_word = max(it["n_words"] for it in batch)         # word padding

        input_ids = torch.full((bsz, max_sub), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(bsz, max_sub, dtype=torch.long)
        first_pos = torch.zeros(bsz, max_word, dtype=torch.long)  # 0, never -1: gather needs valid indices
        word_mask = torch.zeros(bsz, max_word, dtype=torch.bool)

        has_ner = "ner_label_ids" in batch[0]
        has_sev = "severity_id" in batch[0]
        ner_tags = torch.zeros(bsz, max_word, dtype=torch.long) if has_ner else None
        severity_ids = torch.zeros(bsz, dtype=torch.long) if has_sev else None

        for i, it in enumerate(batch):
            s, w = len(it["input_ids"]), it["n_words"]
            input_ids[i, :s] = torch.tensor(it["input_ids"], dtype=torch.long)
            attention_mask[i, :s] = 1
            first_pos[i, :w] = torch.tensor(it["first_pos"], dtype=torch.long)
            word_mask[i, :w] = True
            if has_ner:
                ner_tags[i, :w] = torch.tensor(it["ner_label_ids"], dtype=torch.long)
            if has_sev:
                severity_ids[i] = it["severity_id"]

        out = {
            "tokens": [it["tokens"] for it in batch],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "first_pos": first_pos,
            "word_mask": word_mask,
            "lengths": torch.tensor([it["n_words"] for it in batch], dtype=torch.long),
        }
        if has_ner:
            out["ner_tags"] = ner_tags
        if has_sev:
            out["severity_ids"] = severity_ids
        return out


def _loader(dataset, encoder, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=TransformerCollator(encoder.tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def ner_loader(encoder, split, batch_size, shuffle=None, limit=None):
    """HealthNER split -> DataLoader. Shuffles the train split by default."""
    if shuffle is None:
        shuffle = split == "train"
    ds = TransformerNERDataset(split, encoder, limit)
    if ds.n_truncated:
        print(f"  [{split}] NER: {ds.n_truncated}/{len(ds)} examples truncated at "
              f"{encoder.max_len} subwords")
    return _loader(ds, encoder, batch_size, shuffle)


def severity_loader(encoder, split, batch_size, shuffle=None, limit=None):
    if shuffle is None:
        shuffle = split == "train"
    ds = TransformerSeverityDataset(split, encoder, limit)
    if ds.n_truncated:
        print(f"  [{split}] severity: {ds.n_truncated}/{len(ds)} examples truncated")
    return _loader(ds, encoder, batch_size, shuffle)
