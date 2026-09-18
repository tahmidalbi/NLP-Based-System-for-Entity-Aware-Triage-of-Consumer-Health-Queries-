"""
Precompute the 600D FastText feature cache for every token in the task data.

Guide 9.2 says to cache token vectors so the same token is never recomputed.
This script takes that one step further and does the caching *offline*, once:

    x_i = [ GeneralFastText(w_i) ; MedicalFastText(w_i) ]   (300D + 300D)

Both embeddings are frozen (guide 8.2), and the token vocabulary of the task
data is finite and fully known after Phase 3 - so every vector the training
loop will ever need can be materialised ahead of time. Training then loads a
~120MB .npz instead of 5.6GB + 2.6GB of FastText models.

This is what makes Kaggle GPU training practical: a GPU notebook has roughly
13GB of RAM, which is uncomfortable for two FastText models plus PyTorch. Run
this once on a CPU notebook (~30GB RAM), save the .npz as a Kaggle Dataset,
and every GPU notebook afterwards is memory-light.

Peak RAM here stays around 5.6GB because the two models are loaded and freed
one at a time, never together (the same trick phase7_verify.py uses).

IMPORTANT: the vocabulary is collected from data/processed/ - the Phase 3
output that dataset.py actually reads - not from data/splits/. Collator
raises KeyError on any token missing from the cache, so the two must be built
from identical files.

Usage
-----
  python scripts/build_feature_cache.py
  python scripts/build_feature_cache.py \
      --general /kaggle/input/banglacare-artifacts/cc.bn.300.bin \
      --medical /kaggle/input/banglacare-artifacts/medical_fasttext.bin \
      --out /kaggle/working/feature_cache.npz

Output
------
  feature_cache.npz                    (tokens + vectors)
  results/feature_cache_report.md      (vocabulary/coverage statistics)
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EMBEDDINGS = ROOT / "embeddings"
RESULTS = ROOT / "results"

HEALTHNER_SPLITS = ["train", "valid", "test"]
SEVERITY_SPLITS = ["train", "val", "test"]


def collect_vocabulary():
    """Every token the model can ever be asked for, across all six split files.

    Validation and test tokens are included deliberately. This is a frozen
    embedding lookup, not a fitted statistic - reading a held-out token's
    FastText vector leaks nothing, and Phase 12 needs those vectors to run the
    test sets at all. The decontamination that matters happened in Phase 4,
    where held-out *texts* were kept out of the corpus the medical embedding
    was trained on.
    """
    vocab = set()
    per_source = {}

    for split in HEALTHNER_SPLITS:
        path = PROCESSED / "healthner" / f"{split}.json"
        with open(path, encoding="utf-8") as f:
            examples = json.load(f)
        before = len(vocab)
        for ex in examples:
            vocab.update(ex["tokens"])
        per_source[f"healthner_{split}"] = {
            "rows": len(examples),
            "new_tokens": len(vocab) - before,
        }

    for split in SEVERITY_SPLITS:
        path = PROCESSED / f"severity_{split}.csv"
        df = pd.read_csv(path, usecols=["Text"])
        before = len(vocab)
        for text in df["Text"].astype(str):
            vocab.update(text.split())
        per_source[f"severity_{split}"] = {
            "rows": len(df),
            "new_tokens": len(vocab) - before,
        }

    return sorted(vocab), per_source


def build_vectors(tokens, general_path, medical_path):
    """(V, 600) float32 matrix, loading and freeing one model at a time."""
    import fasttext

    print(f"Loading general FastText: {general_path}")
    t0 = time.time()
    general = fasttext.load_model(str(general_path))
    general_dim = general.get_dimension()
    general_vocab = set(general.get_words())
    general_block = np.stack([general.get_word_vector(t) for t in tokens]).astype(np.float32)
    general_oov = sum(1 for t in tokens if t not in general_vocab)
    del general, general_vocab
    gc.collect()
    print(f"  done in {time.time() - t0:.1f}s, dim={general_dim}, freed")

    print(f"Loading medical FastText: {medical_path}")
    t0 = time.time()
    medical = fasttext.load_model(str(medical_path))
    medical_dim = medical.get_dimension()
    medical_vocab = set(medical.get_words())
    medical_block = np.stack([medical.get_word_vector(t) for t in tokens]).astype(np.float32)
    medical_oov = sum(1 for t in tokens if t not in medical_vocab)
    del medical, medical_vocab
    gc.collect()
    print(f"  done in {time.time() - t0:.1f}s, dim={medical_dim}, freed")

    vectors = np.concatenate([general_block, medical_block], axis=1)
    del general_block, medical_block
    gc.collect()

    stats = {
        "general_dim": general_dim,
        "medical_dim": medical_dim,
        "general_oov": general_oov,
        "medical_oov": medical_oov,
    }
    return vectors, stats


def save_cache(path, tokens, vectors):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, tokens=np.array(tokens, dtype=object), vectors=vectors)


def load_feature_cache(path):
    """dict[token] -> 600D float32 view, ready for FastTextFeaturizer.

    The values are rows of the loaded matrix rather than copies, so the dict
    costs the matrix plus key overhead, not twice the matrix.
    """
    data = np.load(path, allow_pickle=True)
    tokens = data["tokens"]
    vectors = data["vectors"]
    return {str(tok): vectors[i] for i, tok in enumerate(tokens)}


def write_report(tokens, vectors, per_source, stats, out_path, elapsed):
    n_tokens, dim = vectors.shape
    size_mb = vectors.nbytes / 1e6

    lines = [
        "# BanglaCare - Feature cache report",
        "",
        f"- Unique tokens cached: **{n_tokens}**",
        f"- Vector dimension: {dim} "
        f"(general {stats['general_dim']} + medical {stats['medical_dim']})",
        f"- Matrix size in memory: {size_mb:.1f} MB (float32)",
        f"- Written to: `{out_path}`",
        f"- Build time: {elapsed:.1f}s",
        "",
        "## Vocabulary sources",
        "",
        "Tokens are collected from `data/processed/` (Phase 3 output) - the same",
        "files `dataset.py` reads - so the cache is complete by construction and",
        "`Collator` can never hit a missing token.",
        "",
        "| Source | Rows | New tokens contributed |",
        "|---|---|---|",
    ]
    for source, info in per_source.items():
        lines.append(f"| {source} | {info['rows']} | {info['new_tokens']} |")

    lines += [
        "",
        "## Subword coverage",
        "",
        "Tokens absent from each model's learned word list. These are *not*",
        "failures: FastText reconstructs them from character n-grams (guide 9),",
        "which is exactly why the design tolerates Banglish spellings and unseen",
        "medicine names without collapsing to a single UNK vector.",
        "",
        f"- Not in general FastText word list: {stats['general_oov']} "
        f"({100 * stats['general_oov'] / n_tokens:.2f}%)",
        f"- Not in medical FastText word list: {stats['medical_oov']} "
        f"({100 * stats['medical_oov'] / n_tokens:.2f}%)",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--general", type=Path, default=EMBEDDINGS / "cc.bn.300.bin")
    parser.add_argument("--medical", type=Path, default=EMBEDDINGS / "medical_fasttext.bin")
    parser.add_argument("--out", type=Path, default=ROOT / "feature_cache.npz")
    args = parser.parse_args()

    for path, name in [(args.general, "general"), (args.medical, "medical")]:
        if not path.exists():
            raise SystemExit(
                f"Missing {name} FastText model at {path}\n"
                "See SETUP.md section 4 (download cc.bn.300.bin) or run "
                "scripts/phase5_medical_fasttext.py (builds medical_fasttext.bin)."
            )
    if not (PROCESSED / "healthner" / "train.json").exists():
        raise SystemExit(
            f"Missing {PROCESSED} - run scripts/phase3_preprocess.py first."
        )

    t0 = time.time()

    print("Collecting vocabulary from data/processed/ ...")
    tokens, per_source = collect_vocabulary()
    print(f"  {len(tokens)} unique tokens")

    vectors, stats = build_vectors(tokens, args.general, args.medical)

    print(f"Saving cache to {args.out} ...")
    save_cache(args.out, tokens, vectors)
    elapsed = time.time() - t0

    RESULTS.mkdir(parents=True, exist_ok=True)
    report = write_report(tokens, vectors, per_source, stats, args.out, elapsed)
    report_path = RESULTS / "feature_cache_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")

    print(
        f"\nDone in {elapsed:.1f}s - {len(tokens)} tokens x {vectors.shape[1]}D "
        f"({vectors.nbytes / 1e6:.1f} MB)"
    )
    print("\nUse it with:")
    print("    from build_feature_cache import load_feature_cache")
    print("    from featurizer import FastTextFeaturizer")
    print(f"    featurizer = FastTextFeaturizer(precomputed_cache=load_feature_cache('{args.out.name}'))")


if __name__ == "__main__":
    main()
