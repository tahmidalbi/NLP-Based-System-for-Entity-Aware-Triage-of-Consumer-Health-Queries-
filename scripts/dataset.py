"""
Phase 7 - "a data loader/input pipeline that yields token IDs/strings, 600D
FastText features, masks, NER tags when available, and severity labels when
available."

Two PyTorch Datasets (HealthNER, Severity) over the Phase 3 cleaned split
files, plus a shared Collator that does dynamic per-batch padding (guide
5.5) and looks up cached 600D FastText features (guide 9) at batch time.
"""

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from labels import NER_LABEL2ID, SEVERITY_LABEL2ID

ROOT = Path(__file__).resolve().parents[1]
MAX_LEN = 512  # guide 5.5 practical cap; real data never exceeds this (see Phase 3 report)


class HealthNERDataset(Dataset):
    """Token-aligned HealthNER examples (tokens list + per-token IOB label ids)."""

    def __init__(self, split):
        path = ROOT / "data" / "processed" / "healthner" / f"{split}.json"
        with open(path, encoding="utf-8") as f:
            self.examples = json.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex["tokens"][:MAX_LEN]
        label_ids = [NER_LABEL2ID[l] for l in ex["labels"][:MAX_LEN]]
        return {"tokens": tokens, "ner_label_ids": label_ids}


class SeverityDataset(Dataset):
    """Whitespace-tokenized severity queries. Action Needed is never loaded
    here - only Text (input) and Categories (target), per the guide 3.3
    leakage rule."""

    def __init__(self, split):
        path = ROOT / "data" / "processed" / f"severity_{split}.csv"
        df = pd.read_csv(path, usecols=["Text", "Categories"])
        self.texts = df["Text"].astype(str).tolist()
        self.severity_ids = [SEVERITY_LABEL2ID[c] for c in df["Categories"].tolist()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].split()[:MAX_LEN]
        return {"tokens": tokens, "severity_id": self.severity_ids[idx]}


class Collator:
    """Dynamic-padding collate function backed by a shared FastTextFeaturizer
    (so token vectors are cached and never recomputed across batches, guide 9.2)."""

    def __init__(self, featurizer):
        self.featurizer = featurizer

    def __call__(self, batch):
        lengths = [len(item["tokens"]) for item in batch]
        max_len = max(lengths)
        batch_size = len(batch)
        feat_dim = self.featurizer.dim

        features = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        token_strings = []

        has_ner = "ner_label_ids" in batch[0]
        has_severity = "severity_id" in batch[0]

        ner_tags = torch.zeros(batch_size, max_len, dtype=torch.long) if has_ner else None
        severity_ids = torch.zeros(batch_size, dtype=torch.long) if has_severity else None

        for i, item in enumerate(batch):
            toks = item["tokens"]
            n = len(toks)
            vecs = self.featurizer.batch_vectors(toks)  # (n, 600)
            features[i, :n] = torch.from_numpy(vecs)
            mask[i, :n] = True
            token_strings.append(toks)

            if has_ner:
                tags = item["ner_label_ids"]
                ner_tags[i, :n] = torch.tensor(tags, dtype=torch.long)
            if has_severity:
                severity_ids[i] = item["severity_id"]

        out = {
            "tokens": token_strings,          # list[list[str]] - the token strings themselves
            "features": features,             # (B, T, 600)
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "mask": mask,                     # (B, T) bool
        }
        if has_ner:
            out["ner_tags"] = ner_tags        # (B, T)
        if has_severity:
            out["severity_ids"] = severity_ids  # (B,)
        return out
