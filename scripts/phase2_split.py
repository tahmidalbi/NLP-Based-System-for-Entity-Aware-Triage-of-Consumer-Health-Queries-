"""
Phase 2 - Splitting and leakage prevention.

1. HealthNER: freeze the official train/valid/test files as-is (no reshuffling),
   copied into data/splits/healthner/ so the "frozen splits" are an explicit
   artifact separate from data/raw/.
2. Severity: duplicate-aware stratified 80/10/10 split.
     - Group exact duplicates (Text normalized: NFC + whitespace + lowercase Latin).
     - Group near-duplicates (char 5-gram Jaccard >= 0.90).
     - Assign whole groups to one split (never split a duplicate/near-dup group).
     - Allocate groups to train/val/test greedily per dominant class label,
       preserving the four-class distribution as closely as possible.
3. Held-out fingerprint list: HealthNER(val+test) + Severity(val+test) text,
   normalized. Saved for Phase 4 to exclude from the medical corpus.

Raw files under data/raw/ are only read, never modified.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from text_utils import UnionFind, find_near_duplicate_pairs, normalize_text

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"
SEED = 42

SPLITS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. HealthNER: freeze official splits (copy, do not reshuffle)
# ---------------------------------------------------------------------------
def freeze_healthner_splits():
    out_dir = SPLITS / "healthner"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ["train", "valid", "test"]:
        with open(RAW / "Bangla-HealthNER" / f"{split}.json", encoding="utf-8") as f:
            data = json.load(f)
        with open(out_dir / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        counts[split] = len(data)
    return counts


# ---------------------------------------------------------------------------
# 2. Severity: duplicate-aware stratified split
# ---------------------------------------------------------------------------
def build_severity_groups(df):
    """Union-find groups over exact + near-duplicate normalized Text."""
    texts_norm = df["Text"].astype(str).apply(normalize_text).tolist()
    n = len(texts_norm)
    uf = UnionFind(n)

    # exact duplicates
    first_seen = {}
    for i, t in enumerate(texts_norm):
        if t in first_seen:
            uf.union(first_seen[t], i)
        else:
            first_seen[t] = i

    # near duplicates (bounded-cost bucketed search)
    near_pairs = find_near_duplicate_pairs(texts_norm, n=5, threshold=0.90)
    for i, j, _sim in near_pairs:
        uf.union(i, j)

    groups = uf.groups()
    return groups, near_pairs


def allocate_groups_to_splits(df, groups, train_frac=0.8, val_frac=0.1):
    """Greedy group-level stratified allocation, preserving class proportions
    as closely as possible while never splitting a duplicate/near-dup group.
    """
    rng = random.Random(SEED)

    group_info = []
    for g in groups:
        labels = df.iloc[g]["Categories"].tolist()
        dominant = Counter(labels).most_common(1)[0][0]
        mixed = len(set(labels)) > 1
        group_info.append({
            "indices": g,
            "size": len(g),
            "dominant_label": dominant,
            "mixed_labels": mixed,
            "label_set": sorted(set(labels)),
        })

    by_class = defaultdict(list)
    for gi in group_info:
        by_class[gi["dominant_label"]].append(gi)

    split_assignment = {}  # row index -> split name
    per_class_targets = {}
    per_class_actual = {}

    for label, glist in by_class.items():
        rng.shuffle(glist)
        n_rows = sum(g["size"] for g in glist)
        targets = {
            "train": round(n_rows * train_frac),
            "val": round(n_rows * val_frac),
        }
        targets["test"] = n_rows - targets["train"] - targets["val"]
        per_class_targets[label] = dict(targets)

        actual = {"train": 0, "val": 0, "test": 0}
        for gi in glist:
            # assign to whichever split is furthest below its target (by ratio)
            def deficit(split_name):
                t = targets[split_name] or 1
                return (t - actual[split_name]) / t

            best_split = max(["train", "val", "test"], key=deficit)
            for idx in gi["indices"]:
                split_assignment[idx] = best_split
            actual[best_split] += gi["size"]
        per_class_actual[label] = actual

    return split_assignment, per_class_targets, per_class_actual, group_info


def split_severity():
    xlsx_path = RAW / "Bangla-Healthcare-Severity-Dataset" / "Bangla-Healthcare_Dataset.xlsx"
    df = pd.read_excel(xlsx_path)

    groups, near_pairs = build_severity_groups(df)
    exact_dup_groups = [g for g in groups if len(g) > 1]

    split_assignment, per_class_targets, per_class_actual, group_info = allocate_groups_to_splits(df, groups)

    df = df.copy()
    df["__split"] = [split_assignment[i] for i in range(len(df))]

    out = {}
    for split in ["train", "val", "test"]:
        sub = df[df["__split"] == split].drop(columns=["__split"])
        sub.to_csv(SPLITS / f"severity_{split}.csv", index=False, encoding="utf-8")
        out[split] = sub

    mixed_label_groups = [g for g in group_info if g["mixed_labels"]]

    stats = {
        "total_rows": len(df),
        "total_groups": len(groups),
        "exact_or_near_duplicate_groups": len(exact_dup_groups),
        "near_duplicate_pairs_found": len(near_pairs),
        "mixed_label_duplicate_groups": len(mixed_label_groups),
        "mixed_label_group_examples": [
            {"rows": g["indices"], "labels": g["label_set"]} for g in mixed_label_groups[:10]
        ],
        "split_sizes": {k: len(v) for k, v in out.items()},
        "split_class_pct": {
            split: {
                k: round(100 * v / len(sub), 2)
                for k, v in sub["Categories"].value_counts().to_dict().items()
            }
            for split, sub in out.items()
        },
        "per_class_row_targets": per_class_targets,
        "per_class_row_actual": per_class_actual,
    }
    return out, stats


# ---------------------------------------------------------------------------
# 3. Held-out fingerprint list
# ---------------------------------------------------------------------------
def build_held_out(healthner_counts, severity_splits):
    fingerprints = set()

    for split in ["valid", "test"]:
        with open(SPLITS / "healthner" / f"{split}.json", encoding="utf-8") as f:
            data = json.load(f)
        for ex in data:
            fingerprints.add(normalize_text(ex["text"]))

    for split in ["val", "test"]:
        for t in severity_splits[split]["Text"].astype(str).tolist():
            fingerprints.add(normalize_text(t))

    fp_list = sorted(fingerprints)
    with open(SPLITS / "held_out_fingerprints.json", "w", encoding="utf-8") as f:
        json.dump(fp_list, f, ensure_ascii=False, indent=None)

    return len(fp_list)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(healthner_counts, severity_stats, held_out_count):
    report = {
        "healthner_frozen_split_sizes": healthner_counts,
        "severity_split": severity_stats,
        "held_out_fingerprint_count": held_out_count,
    }
    (RESULTS / "phase2_split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = ["# BanglaCare - Phase 2 Split Report", ""]
    lines.append("## HealthNER (frozen official splits)")
    for k, v in healthner_counts.items():
        lines.append(f"- {k}: {v} rows")
    lines.append("")

    lines.append("## Severity (duplicate-aware stratified 80/10/10)")
    s = severity_stats
    lines.append(f"- Total rows: {s['total_rows']}")
    lines.append(f"- Total duplicate-aware groups: {s['total_groups']}")
    lines.append(f"- Exact/near-duplicate groups (size > 1): {s['exact_or_near_duplicate_groups']}")
    lines.append(f"- Near-duplicate pairs found (Jaccard >= 0.90, not already exact): {s['near_duplicate_pairs_found']}")
    lines.append(f"- Duplicate groups with inconsistent Categories labels: {s['mixed_label_duplicate_groups']}")
    if s["mixed_label_group_examples"]:
        lines.append("  Examples (row indices -> label set):")
        for ex in s["mixed_label_group_examples"]:
            lines.append(f"    {ex['rows']} -> {ex['labels']}")
    lines.append("")
    lines.append(f"- Split sizes: {s['split_sizes']}")
    lines.append(f"- Split class %: {s['split_class_pct']}")
    lines.append("")

    lines.append("## Held-out fingerprint list")
    lines.append(f"- {held_out_count} unique normalized texts (HealthNER val+test U Severity val+test)")
    lines.append("- Saved to data/splits/held_out_fingerprints.json")
    lines.append("- These texts must never enter the Phase 4 medical corpus.")

    (RESULTS / "phase2_split_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Freezing HealthNER official splits...")
    healthner_counts = freeze_healthner_splits()
    print(healthner_counts)

    print("Building duplicate-aware severity split...")
    severity_splits, severity_stats = split_severity()
    print("Split sizes:", severity_stats["split_sizes"])
    print("Split class %:", severity_stats["split_class_pct"])
    print("Mixed-label duplicate groups:", severity_stats["mixed_label_duplicate_groups"])

    print("Building held-out fingerprint list...")
    held_out_count = build_held_out(healthner_counts, severity_splits)
    print("Held-out fingerprints:", held_out_count)

    write_report(healthner_counts, severity_stats, held_out_count)
    print("\nWrote results/phase2_split_report.json and .md")


if __name__ == "__main__":
    main()
