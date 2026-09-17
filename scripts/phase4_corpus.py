"""
Phase 4 - Medical corpus construction.

Builds the unlabeled corpus for training the custom Medical FastText
(Phase 5). Sources (per guide 6.1):
  - BanglaHealth: source_sentence + paraphrased_sentence (all 200k rows)
  - Bangla-HealthNER: TRAIN text only
  - BanglaCHQ-Summ: questions + summaries (all splits)
  - Severity dataset: TRAIN Text only (never Action Needed)

Dedup order (per guide 6.3):
  1. Normalize candidate lines (text_utils.normalize_text - same function
     used for Phase 2 duplicate detection).
  2. Remove empty lines.
  3. Remove exact duplicate lines (global, across all sources).
  4. Remove lines that exact-match the Phase 2 held-out fingerprint list.
  5. Remove lines that near-duplicate (char 5-gram Jaccard >= 0.90) a
     held-out text.
  6. Write corpus/medical_corpus.txt, one text per line.

Near-duplicate checking (step 5) is scoped to HealthNER-train and
Severity-train candidates only, not BanglaHealth/BanglaCHQ-Summ - see the
"near_dup_scope_justification" note written into the stats report.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from text_utils import BANGLA_RE, LATIN_RE, near_duplicate_matches_against_reference, normalize_text

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"
CORPUS_DIR = ROOT / "corpus"
RESULTS = ROOT / "results"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

ASCII_DIGIT_RE = re.compile(r"[0-9]")
BANGLA_DIGIT_RE = re.compile(r"[০-৯]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def load_banglahealth_lines():
    from datasets import load_dataset
    ds = load_dataset("faisal4590aziz/bangla-health-related-paraphrased-dataset")
    df = ds[list(ds.keys())[0]].to_pandas()
    lines = []
    lines += [("banglahealth_source", t) for t in df["source_sentence"].astype(str).tolist()]
    lines += [("banglahealth_paraphrase", t) for t in df["paraphrased_sentence"].astype(str).tolist()]
    return lines


def load_healthner_train_lines():
    with open(SPLITS / "healthner" / "train.json", encoding="utf-8") as f:
        data = json.load(f)
    return [("healthner_train", ex["text"]) for ex in data]


def load_chq_summ_lines():
    lines = []
    for split in ["train", "valid", "test"]:
        df = pd.read_csv(RAW / "BanglaCHQ-Summ-dataset" / f"{split}.csv")
        lines += [("chq_summ_question", t) for t in df["question"].astype(str).tolist()]
        lines += [("chq_summ_summary", t) for t in df["summary"].astype(str).tolist()]
    return lines


def load_severity_train_lines():
    df = pd.read_csv(SPLITS / "severity_train.csv")
    return [("severity_train", t) for t in df["Text"].astype(str).tolist()]


def load_held_out():
    with open(SPLITS / "held_out_fingerprints.json", encoding="utf-8") as f:
        return set(json.load(f))


def script_mix_stats(texts):
    n = len(texts)
    if n == 0:
        return {"n": 0}
    n_bangla = n_latin = n_mixed = n_digit = n_url = 0
    for t in texts:
        has_b = bool(BANGLA_RE.search(t))
        has_l = bool(LATIN_RE.search(t))
        if has_b:
            n_bangla += 1
        if has_l:
            n_latin += 1
        if has_b and has_l:
            n_mixed += 1
        if ASCII_DIGIT_RE.search(t) or BANGLA_DIGIT_RE.search(t):
            n_digit += 1
        if URL_RE.search(t):
            n_url += 1
    pct = lambda x: round(100.0 * x / n, 2)
    return {
        "n": n,
        "contains_bangla_pct": pct(n_bangla),
        "contains_latin_pct": pct(n_latin),
        "mixed_script_pct": pct(n_mixed),
        "contains_digit_pct": pct(n_digit),
        "contains_url_pct": pct(n_url),
    }


def main():
    print("Loading held-out fingerprint list (Phase 2 output)...")
    held_out = load_held_out()
    print(f"  {len(held_out)} held-out fingerprints")

    print("Loading source texts...")
    print("  BanglaHealth (Hugging Face, cached)...")
    banglahealth = load_banglahealth_lines()
    print(f"    {len(banglahealth)} candidate lines")
    print("  HealthNER TRAIN...")
    healthner_train = load_healthner_train_lines()
    print(f"    {len(healthner_train)} candidate lines")
    print("  BanglaCHQ-Summ (all splits)...")
    chq_summ = load_chq_summ_lines()
    print(f"    {len(chq_summ)} candidate lines")
    print("  Severity TRAIN (Text only)...")
    severity_train = load_severity_train_lines()
    print(f"    {len(severity_train)} candidate lines")

    all_candidates = banglahealth + healthner_train + chq_summ + severity_train
    per_source_raw_count = Counter(src for src, _ in all_candidates)
    print(f"\nTotal raw candidate lines: {len(all_candidates)}")

    # Step 1: normalize
    normalized = [(src, normalize_text(t)) for src, t in all_candidates]

    # Step 2: remove empty lines
    non_empty = [(src, t) for src, t in normalized if t != ""]
    n_removed_empty = len(normalized) - len(non_empty)
    print(f"Removed empty lines: {n_removed_empty}")

    # Step 3: remove exact duplicate lines (global, keep first occurrence + its source)
    seen = set()
    deduped = []
    per_source_after_exact_dedup = Counter()
    for src, t in non_empty:
        if t in seen:
            continue
        seen.add(t)
        deduped.append((src, t))
        per_source_after_exact_dedup[src] += 1
    n_removed_exact_dup = len(non_empty) - len(deduped)
    print(f"Removed exact duplicate lines: {n_removed_exact_dup}")
    print(f"Remaining after exact dedup: {len(deduped)}")

    # Step 4: remove lines that exact-match the held-out fingerprint list
    after_held_out_exact = [(src, t) for src, t in deduped if t not in held_out]
    n_removed_held_out_exact = len(deduped) - len(after_held_out_exact)
    print(f"Removed exact held-out matches: {n_removed_held_out_exact}")

    # Step 5: near-duplicate check against held-out, scoped to HealthNER-train
    # and Severity-train candidates (see justification in the stats report).
    scoped_sources = {"healthner_train", "severity_train"}
    scoped_idxs = [i for i, (src, t) in enumerate(after_held_out_exact) if src in scoped_sources]
    scoped_texts = [after_held_out_exact[i][1] for i in scoped_idxs]

    print(f"\nNear-duplicate check (Jaccard >= 0.90) against {len(held_out)} held-out texts, "
          f"scoped to {len(scoped_texts)} HealthNER-train/Severity-train candidates...")
    held_out_list = list(held_out)
    matched_scoped_positions = near_duplicate_matches_against_reference(
        scoped_texts, held_out_list, n=5, threshold=0.90
    )
    matched_global_idxs = {scoped_idxs[p] for p in matched_scoped_positions}
    print(f"Removed near-duplicate matches: {len(matched_global_idxs)}")

    final_lines = [
        (src, t) for i, (src, t) in enumerate(after_held_out_exact) if i not in matched_global_idxs
    ]

    # Step 6: write corpus (one text per line, no source tags)
    corpus_path = CORPUS_DIR / "medical_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for _src, t in final_lines:
            f.write(t + "\n")
    print(f"\nWrote {corpus_path} ({len(final_lines)} lines)")

    # Stats
    per_source_final = Counter(src for src, _ in final_lines)
    all_texts = [t for _src, t in final_lines]
    unique_tokens = set()
    for t in all_texts:
        unique_tokens.update(t.split())

    stats = {
        "sources_included": {
            "BanglaHealth": "source_sentence + paraphrased_sentence, all 200,000 rows",
            "Bangla-HealthNER": "TRAIN text only (25,426 rows)",
            "BanglaCHQ-Summ": "questions + summaries, all splits (train/valid/test)",
            "Severity dataset": "TRAIN Text only (Action Needed never included)",
        },
        "raw_candidate_lines_per_source": dict(per_source_raw_count),
        "total_raw_candidate_lines": len(all_candidates),
        "removed_empty_lines": n_removed_empty,
        "removed_exact_duplicate_lines": n_removed_exact_dup,
        "removed_exact_held_out_matches": n_removed_held_out_exact,
        "near_dup_scope_justification": (
            "Near-duplicate checking against the held-out fingerprint list (guide step 6.3.10) is "
            "computationally scoped to HealthNER-train and Severity-train candidates only. These are "
            "the only two sources that share dataset lineage with the held-out HealthNER-val/test and "
            "Severity-val/test texts (same collection, split from the same pool), so they are the only "
            "sources where near-duplicate leakage is plausible. BanglaHealth and BanglaCHQ-Summ are "
            "entirely separate datasets/collection processes; exact-duplicate removal (step 6.3.9) "
            "already catches any coincidental verbatim overlap with held-out text, and near-identical "
            "overlap beyond that is not a realistic risk given the different sources."
        ),
        "near_dup_candidates_checked": len(scoped_texts),
        "removed_near_duplicate_held_out_matches": len(matched_global_idxs),
        "final_corpus_lines": len(final_lines),
        "final_lines_per_source": dict(per_source_final),
        "unique_token_count": len(unique_tokens),
        "script_mix": script_mix_stats(all_texts),
    }

    (RESULTS / "phase4_corpus_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines_md = ["# BanglaCare - Phase 4 Medical Corpus Construction Report", ""]
    lines_md.append("## Sources included")
    for k, v in stats["sources_included"].items():
        lines_md.append(f"- **{k}**: {v}")
    lines_md.append("")
    lines_md.append("## Deduplication funnel")
    lines_md.append(f"- Raw candidate lines: {stats['total_raw_candidate_lines']} ({stats['raw_candidate_lines_per_source']})")
    lines_md.append(f"- Removed empty lines: {stats['removed_empty_lines']}")
    lines_md.append(f"- Removed exact duplicate lines: {stats['removed_exact_duplicate_lines']}")
    lines_md.append(f"- Removed exact held-out matches: {stats['removed_exact_held_out_matches']}")
    lines_md.append(f"- Near-dup check scope: {stats['near_dup_candidates_checked']} candidates "
                     f"(HealthNER-train + Severity-train only)")
    lines_md.append(f"- Removed near-duplicate held-out matches: {stats['removed_near_duplicate_held_out_matches']}")
    lines_md.append(f"- **Final corpus lines: {stats['final_corpus_lines']}**")
    lines_md.append(f"- Final lines per source: {stats['final_lines_per_source']}")
    lines_md.append("")
    lines_md.append(f"## Corpus statistics")
    lines_md.append(f"- Unique whitespace tokens: {stats['unique_token_count']}")
    lines_md.append(f"- Script mix: {stats['script_mix']}")
    lines_md.append("")
    lines_md.append("## Near-duplicate scope justification")
    lines_md.append(stats["near_dup_scope_justification"])

    (RESULTS / "phase4_corpus_stats.md").write_text("\n".join(lines_md), encoding="utf-8")
    print("\nWrote results/phase4_corpus_stats.json and .md")


if __name__ == "__main__":
    main()
