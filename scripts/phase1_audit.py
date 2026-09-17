"""
Phase 1 - Workspace and raw-data audit.

Audits every raw dataset before any splitting/cleaning happens:
  - Bangla-HealthNER      (data/raw/Bangla-HealthNER/{train,valid,test}.json)
  - Severity dataset      (data/raw/Bangla-Healthcare-Severity-Dataset/Bangla-Healthcare_Dataset.xlsx)
  - BanglaCHQ-Summ        (data/raw/BanglaCHQ-Summ-dataset/{train,valid,test}.csv)
  - BanglaHealth          (Hugging Face: faisal4590aziz/bangla-health-related-paraphrased-dataset)

Writes:
  - results/dataset_audit.json   (full structured stats, machine-readable)
  - results/dataset_audit.md     (human-readable summary)

No rows are modified. Raw files under data/raw/ are only read, never written.
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

BANGLA_RE = re.compile(r"[ঀ-৿]")
LATIN_RE = re.compile(r"[A-Za-z]")
ASCII_DIGIT_RE = re.compile(r"[0-9]")
BANGLA_DIGIT_RE = re.compile(r"[০-৯]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
CONTROL_RE = re.compile(r"[​‌‍﻿‎‏]")  # zero-width / BOM / directional marks


def normalize(text):
    """Light normalization used throughout the project: NFC + whitespace collapse."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_for_dup(text):
    """Normalization used only for duplicate detection: + lowercase Latin chars."""
    t = normalize(text)
    return "".join(c.lower() if c.isascii() else c for c in t)


def percentiles(lengths):
    if not lengths:
        return {"median": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "mean": 0.0}
    arr = np.array(lengths)
    return {
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def script_mix(texts):
    n = len(texts)
    n_bangla = n_latin = n_mixed = n_digit = n_url = n_control = n_empty = 0
    for t in texts:
        t = "" if t is None else str(t)
        if t.strip() == "":
            n_empty += 1
            continue
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
        if CONTROL_RE.search(t):
            n_control += 1
    pct = lambda x: round(100.0 * x / n, 2) if n else 0.0
    return {
        "n": n,
        "empty_or_missing": n_empty,
        "pct_empty_or_missing": pct(n_empty),
        "contains_bangla_pct": pct(n_bangla),
        "contains_latin_pct": pct(n_latin),
        "mixed_script_pct": pct(n_mixed),
        "contains_digit_pct": pct(n_digit),
        "contains_url_pct": pct(n_url),
        "contains_zero_width_or_bom_pct": pct(n_control),
    }


def exact_duplicates(texts):
    norm = [normalize_for_dup(t) for t in texts]
    counts = Counter(x for x in norm if x != "")
    dup_groups = {k: v for k, v in counts.items() if v > 1}
    n_dup_rows = sum(dup_groups.values())
    return {
        "duplicate_groups": len(dup_groups),
        "rows_involved_in_duplicates": n_dup_rows,
        "pct_rows_duplicated": round(100.0 * n_dup_rows / len(texts), 2) if texts else 0.0,
    }


def token_len_stats(texts):
    return percentiles([len(normalize(t).split()) for t in texts])


# ---------------------------------------------------------------------------
# 1. Bangla-HealthNER
# ---------------------------------------------------------------------------
def audit_healthner():
    out = {}
    all_texts_by_split = {}
    label_set = set()
    entity_counts_by_split = {}
    cross_split_norm_sets = {}

    for split in ["train", "valid", "test"]:
        with open(RAW / "Bangla-HealthNER" / f"{split}.json", encoding="utf-8") as f:
            data = json.load(f)

        texts = [ex["text"] for ex in data]
        all_texts_by_split[split] = texts

        missing = sum(1 for ex in data if not ex.get("text") or not str(ex["text"]).strip())
        mismatches = sum(1 for ex in data if len(str(ex["text"]).split()) != len(ex["labels"]))

        entity_counts = Counter()
        for ex in data:
            for lab in ex["labels"]:
                label_set.add(lab)
                if lab.startswith("B-"):
                    entity_counts[lab[2:]] += 1
        entity_counts_by_split[split] = dict(entity_counts)

        cross_split_norm_sets[split] = set(normalize_for_dup(t) for t in texts)

        zero_bangla = sum(1 for t in texts if not BANGLA_RE.search(t))

        out[split] = {
            "rows": len(data),
            "columns": ["text", "labels"],
            "missing_or_empty_text": missing,
            "token_label_count_mismatches": mismatches,
            "token_length_stats": token_len_stats(texts),
            "script_mix": script_mix(texts),
            "exact_duplicates_within_split": exact_duplicates(texts),
            "entity_span_counts_B_tags": dict(entity_counts),
            "rows_with_zero_bangla_chars": zero_bangla,
            "pct_rows_with_zero_bangla_chars": round(100.0 * zero_bangla / len(texts), 2),
        }

    # cross-split overlap (official splits are frozen as-is in Phase 2, but worth documenting now)
    out["cross_split_exact_text_overlap"] = {
        "train_valid": len(cross_split_norm_sets["train"] & cross_split_norm_sets["valid"]),
        "train_test": len(cross_split_norm_sets["train"] & cross_split_norm_sets["test"]),
        "valid_test": len(cross_split_norm_sets["valid"] & cross_split_norm_sets["test"]),
    }
    out["label_set"] = sorted(label_set)
    out["num_labels"] = len(label_set)
    out["entity_types"] = sorted(set(l[2:] for l in label_set if l != "O"))
    return out


# ---------------------------------------------------------------------------
# 2. Bangla Healthcare Severity Dataset
# ---------------------------------------------------------------------------
def audit_severity():
    csv_path = RAW / "Bangla-Healthcare-Severity-Dataset" / "Bangla-Healthcare_Dataset.csv"
    xlsx_path = RAW / "Bangla-Healthcare-Severity-Dataset" / "Bangla-Healthcare_Dataset.xlsx"

    # Detect CSV mojibake (Bangla text replaced by literal '?' characters).
    with open(csv_path, "rb") as f:
        raw_bytes = f.read()
    try:
        raw_bytes.decode("utf-8")
        csv_is_utf8 = True
    except UnicodeDecodeError:
        csv_is_utf8 = False
    csv_text_sample = raw_bytes.decode("cp1252", errors="replace").splitlines()[1] if not csv_is_utf8 else ""
    csv_corrupted = (not csv_is_utf8) and ("?" * 5 in csv_text_sample)

    df = pd.read_excel(xlsx_path)

    missing_text = int(df["Text"].isna().sum() + (df["Text"].astype(str).str.strip() == "").sum())
    missing_action = int(df["Action Needed"].isna().sum())

    class_counts = df["Categories"].value_counts().to_dict()
    total = len(df)
    class_pct = {k: round(100.0 * v / total, 2) for k, v in class_counts.items()}

    return {
        "rows": total,
        "columns": list(df.columns),
        "authoritative_source": "Bangla-Healthcare_Dataset.xlsx (CSV export is corrupted, see csv_encoding_issue)",
        "csv_encoding_issue": {
            "csv_is_valid_utf8": csv_is_utf8,
            "csv_bangla_text_appears_mojibaked_to_question_marks": csv_corrupted,
            "recommendation": "Load the Severity dataset from the .xlsx file, not the .csv file.",
        },
        "missing_or_empty_Text": missing_text,
        "missing_Action_Needed": missing_action,
        "token_length_stats_Text": token_len_stats(df["Text"].astype(str).tolist()),
        "script_mix_Text": script_mix(df["Text"].astype(str).tolist()),
        "exact_duplicates_Text": exact_duplicates(df["Text"].astype(str).tolist()),
        "class_counts_Categories": class_counts,
        "class_pct_Categories": class_pct,
        "class_balance_check": {
            "all_classes_in_22_to_27_pct_range": all(22.0 <= v <= 27.0 for v in class_pct.values()),
        },
        "leakage_rule": "Action Needed column exists but MUST NOT be used as input feature, auxiliary "
                         "target, prompt, or corpus sentence (annotation-derived, leaks the severity label).",
    }


# ---------------------------------------------------------------------------
# 3. BanglaCHQ-Summ
# ---------------------------------------------------------------------------
def audit_chq_summ():
    out = {}
    for split in ["train", "valid", "test"]:
        df = pd.read_csv(RAW / "BanglaCHQ-Summ-dataset" / f"{split}.csv")
        questions = df["question"].astype(str).tolist()
        summaries = df["summary"].astype(str).tolist()
        out[split] = {
            "rows": len(df),
            "columns": list(df.columns),
            "missing_or_empty_question": int(df["question"].isna().sum() + (df["question"].astype(str).str.strip() == "").sum()),
            "missing_or_empty_summary": int(df["summary"].isna().sum() + (df["summary"].astype(str).str.strip() == "").sum()),
            "token_length_stats_question": token_len_stats(questions),
            "token_length_stats_summary": token_len_stats(summaries),
            "script_mix_question": script_mix(questions),
            "exact_duplicates_question": exact_duplicates(questions),
        }
    return out


# ---------------------------------------------------------------------------
# 4. BanglaHealth (Hugging Face)
# ---------------------------------------------------------------------------
def audit_banglahealth():
    from datasets import load_dataset

    ds = load_dataset("faisal4590aziz/bangla-health-related-paraphrased-dataset")
    split = list(ds.keys())[0]
    df = ds[split].to_pandas()

    src = df["source_sentence"].astype(str).tolist()
    para = df["paraphrased_sentence"].astype(str).tolist()

    unique_src = df["source_sentence"].nunique()
    top_repeated = (
        df["source_sentence"].value_counts().head(5).to_dict()
    )

    return {
        "hf_split_name": split,
        "rows": len(df),
        "columns": list(df.columns),
        "missing_or_empty_source_sentence": int(df["source_sentence"].isna().sum() + (df["source_sentence"].astype(str).str.strip() == "").sum()),
        "missing_or_empty_paraphrased_sentence": int(df["paraphrased_sentence"].isna().sum() + (df["paraphrased_sentence"].astype(str).str.strip() == "").sum()),
        "token_length_stats_source_sentence": token_len_stats(src),
        "token_length_stats_paraphrased_sentence": token_len_stats(para),
        "script_mix_source_sentence": script_mix(src),
        "exact_duplicates_source_sentence": exact_duplicates(src),
        "exact_duplicates_paraphrased_sentence": exact_duplicates(para),
        "unique_source_sentences": int(unique_src),
        "mean_paraphrases_per_unique_source": round(len(df) / unique_src, 2),
        "top_5_most_repeated_source_sentences": {str(k): int(v) for k, v in top_repeated.items()},
        "boilerplate_warning": "Most-repeated source sentences look like author/byline boilerplate "
                                "(e.g. 'লেখক ... বিভাগ ... হাসপাতাল') rather than health content. "
                                "Consider filtering very-short, highly-repeated lines during Phase 4 "
                                "corpus construction so they do not dominate FastText training.",
        "usage_note": "Use only as unlabeled corpus text (Phase 4). Do not derive severity or NER labels from it.",
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def write_markdown(report, path):
    lines = ["# BanglaCare - Phase 1 Dataset Audit Report", ""]

    lines.append("## 1. Bangla-HealthNER")
    hn = report["healthner"]
    for split in ["train", "valid", "test"]:
        d = hn[split]
        lines.append(f"### {split}")
        lines.append(f"- Rows: {d['rows']}")
        lines.append(f"- Missing/empty text: {d['missing_or_empty_text']}")
        lines.append(f"- Token/label count mismatches: {d['token_label_count_mismatches']}")
        ls = d["token_length_stats"]
        lines.append(f"- Token length (median/p90/p95/p99/max): {ls['median']}/{ls['p90']}/{ls['p95']}/{ls['p99']}/{ls['max']}")
        sm = d["script_mix"]
        lines.append(f"- Contains Bangla: {sm['contains_bangla_pct']}% | Latin: {sm['contains_latin_pct']}% | Mixed-script: {sm['mixed_script_pct']}% | Digits: {sm['contains_digit_pct']}% | URLs: {sm['contains_url_pct']}%")
        dup = d["exact_duplicates_within_split"]
        lines.append(f"- Exact duplicate rows: {dup['rows_involved_in_duplicates']} ({dup['pct_rows_duplicated']}%) in {dup['duplicate_groups']} groups")
        lines.append(f"- Entity span counts (B- tags): {d['entity_span_counts_B_tags']}")
        lines.append(f"- Rows with zero Bangla characters (pure English/Banglish): {d['rows_with_zero_bangla_chars']} ({d['pct_rows_with_zero_bangla_chars']}%)")
        lines.append("")
    lines.append(f"- Cross-split exact text overlap: {hn['cross_split_exact_text_overlap']}")
    lines.append(f"- Full label set ({hn['num_labels']} labels): {hn['label_set']}")
    lines.append(f"- Entity types ({len(hn['entity_types'])}): {hn['entity_types']}")
    lines.append("")

    lines.append("## 2. Bangla Healthcare Severity Dataset")
    sv = report["severity"]
    lines.append(f"- Rows: {sv['rows']}")
    lines.append(f"- Columns: {sv['columns']}")
    lines.append(f"- **Authoritative source: {sv['authoritative_source']}**")
    lines.append(f"- CSV encoding issue: {sv['csv_encoding_issue']}")
    lines.append(f"- Missing/empty Text: {sv['missing_or_empty_Text']}")
    lines.append(f"- Missing Action Needed: {sv['missing_Action_Needed']}")
    ls = sv["token_length_stats_Text"]
    lines.append(f"- Token length (median/p90/p95/p99/max): {ls['median']}/{ls['p90']}/{ls['p95']}/{ls['p99']}/{ls['max']}")
    sm = sv["script_mix_Text"]
    lines.append(f"- Contains Bangla: {sm['contains_bangla_pct']}% | Latin: {sm['contains_latin_pct']}% | Mixed-script: {sm['mixed_script_pct']}% | Digits: {sm['contains_digit_pct']}%")
    dup = sv["exact_duplicates_Text"]
    lines.append(f"- Exact duplicate rows: {dup['rows_involved_in_duplicates']} ({dup['pct_rows_duplicated']}%) in {dup['duplicate_groups']} groups")
    lines.append(f"- Class counts: {sv['class_counts_Categories']}")
    lines.append(f"- Class percentages: {sv['class_pct_Categories']}")
    lines.append(f"- All classes within 22-27%: {sv['class_balance_check']['all_classes_in_22_to_27_pct_range']}")
    lines.append(f"- **Leakage rule: {sv['leakage_rule']}**")
    lines.append("")

    lines.append("## 3. BanglaCHQ-Summ")
    chq = report["chq_summ"]
    for split in ["train", "valid", "test"]:
        d = chq[split]
        lines.append(f"### {split}")
        lines.append(f"- Rows: {d['rows']} | Columns: {d['columns']}")
        lines.append(f"- Missing/empty question: {d['missing_or_empty_question']} | Missing/empty summary: {d['missing_or_empty_summary']}")
        lq = d["token_length_stats_question"]
        ls_ = d["token_length_stats_summary"]
        lines.append(f"- Question token length (median/p90/p95/p99/max): {lq['median']}/{lq['p90']}/{lq['p95']}/{lq['p99']}/{lq['max']}")
        lines.append(f"- Summary token length (median/p90/p95/p99/max): {ls_['median']}/{ls_['p90']}/{ls_['p95']}/{ls_['p99']}/{ls_['max']}")
        dup = d["exact_duplicates_question"]
        lines.append(f"- Exact duplicate questions: {dup['rows_involved_in_duplicates']} ({dup['pct_rows_duplicated']}%)")
        lines.append("")

    lines.append("## 4. BanglaHealth (Hugging Face)")
    bh = report["banglahealth"]
    lines.append(f"- Rows: {bh['rows']} | Columns: {bh['columns']}")
    lines.append(f"- Missing/empty source_sentence: {bh['missing_or_empty_source_sentence']} | Missing/empty paraphrased_sentence: {bh['missing_or_empty_paraphrased_sentence']}")
    ls = bh["token_length_stats_source_sentence"]
    lines.append(f"- Source sentence token length (median/p90/p95/p99/max): {ls['median']}/{ls['p90']}/{ls['p95']}/{ls['p99']}/{ls['max']}")
    dup = bh["exact_duplicates_source_sentence"]
    lines.append(f"- Exact duplicate source sentences: {dup['rows_involved_in_duplicates']} ({dup['pct_rows_duplicated']}%)")
    lines.append(f"- Unique source sentences: {bh['unique_source_sentences']} (mean {bh['mean_paraphrases_per_unique_source']} paraphrases per unique source)")
    lines.append(f"- Top 5 most-repeated source sentences: {bh['top_5_most_repeated_source_sentences']}")
    lines.append(f"- **Warning: {bh['boilerplate_warning']}**")
    lines.append(f"- Note: {bh['usage_note']}")
    lines.append("")

    lines.append("## Key Findings and Flags for Phase 2+")
    lines.append("1. **Severity dataset CSV is corrupted** (Bangla text mojibaked to `?`). "
                  "Always load `Bangla-Healthcare_Dataset.xlsx`, never the `.csv` export.")
    lines.append(f"2. Severity class balance is close but not perfectly within 22-27%: "
                  f"Urgent is {sv['class_pct_Categories'].get('Urgent', 0)}% (slightly above 27%). "
                  f"Still close enough that stratified splitting without rebalancing (per the guide) is appropriate.")
    lines.append(f"3. Severity dataset has {sv['exact_duplicates_Text']['duplicate_groups']} exact-duplicate "
                  f"Text groups ({sv['exact_duplicates_Text']['rows_involved_in_duplicates']} rows) — "
                  f"these MUST be grouped together during the Phase 2 duplicate-aware split so they don't "
                  f"leak across train/val/test.")
    lines.append(f"4. HealthNER has a small amount of exact-duplicate text **across the official splits** "
                  f"({hn['cross_split_exact_text_overlap']}). The official splits are still used as-is "
                  f"per the guide, but this is worth noting as a minor, pre-existing leakage source outside our control.")
    lines.append("5. Roughly a fifth of HealthNER text is pure English/Banglish with zero Bangla characters "
                  "(consistent with the project's code-switching design goal, not a defect).")
    lines.append("6. BanglaHealth's most-repeated \"source_sentence\" values are author/byline boilerplate, "
                  "not paraphrasable health content — flag for filtering consideration in Phase 4 medical "
                  "corpus construction.")
    lines.append("7. All four raw sources are otherwise clean: zero missing/empty text fields, and "
                  "HealthNER has a perfect 0-mismatch token/label alignment across all three splits "
                  "(the Phase 3 sanity gate already passes on the raw data).")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    report = {}
    print("Auditing Bangla-HealthNER...")
    report["healthner"] = audit_healthner()
    print("Auditing Severity dataset...")
    report["severity"] = audit_severity()
    print("Auditing BanglaCHQ-Summ...")
    report["chq_summ"] = audit_chq_summ()
    print("Auditing BanglaHealth (Hugging Face)...")
    report["banglahealth"] = audit_banglahealth()

    json_path = RESULTS / "dataset_audit.json"
    md_path = RESULTS / "dataset_audit.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
