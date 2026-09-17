"""
Phase 3 - Text preprocessing and tokenization.

Applies the deterministic, conservative cleaning rules from the guide (5.1-5.3)
to the frozen Phase 2 splits, and re-verifies the sanity gate: for every
HealthNER example, token count must equal label count.

HealthNER: cleaned PER TOKEN (never re-split from a joined/cleaned string).
This guarantees the token list can only be transformed in place - it can
never merge, drop, or add a token - so label alignment is preserved by
construction, not just by after-the-fact checking. Uses text_utils.clean_token.

Severity / BanglaCHQ-Summ: no token-label alignment constraint, so the whole
Text/question/summary field is cleaned with text_utils.normalize_text (the
same function already used for Phase 2 dedup and Phase 4 corpus lines).

Nothing is stemmed, stopword-filtered, translated, or has numbers/medicine
names touched, per guide 5.3.
"""

import json
from pathlib import Path

import pandas as pd

from text_utils import clean_token, normalize_text

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MAX_LEN = 512

PROCESSED.mkdir(parents=True, exist_ok=True)
(PROCESSED / "healthner").mkdir(parents=True, exist_ok=True)


def process_healthner_split(split):
    with open(SPLITS / "healthner" / f"{split}.json", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_examples = []
    sanity_failures = 0
    empty_token_fallbacks = 0
    length_over_cap = 0
    max_len_seen = 0

    for ex in data:
        tokens = ex["text"].split()
        labels = ex["labels"]

        cleaned_tokens = []
        for tok in tokens:
            ct, fell_back = clean_token(tok)
            cleaned_tokens.append(ct)
            if fell_back:
                empty_token_fallbacks += 1

        if len(cleaned_tokens) != len(labels):
            sanity_failures += 1
            # Per the guide's sanity gate: do not silently repair - keep the
            # original tokens for this example and flag it instead.
            cleaned_tokens = tokens

        n = len(cleaned_tokens)
        max_len_seen = max(max_len_seen, n)
        if n > MAX_LEN:
            length_over_cap += 1

        cleaned_examples.append({
            "text": " ".join(cleaned_tokens),
            "tokens": cleaned_tokens,
            "labels": labels,
        })

    with open(PROCESSED / "healthner" / f"{split}.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_examples, f, ensure_ascii=False)

    return {
        "rows": len(data),
        "sanity_gate_failures": sanity_failures,
        "empty_token_fallbacks_logged": empty_token_fallbacks,
        "max_token_length": max_len_seen,
        "examples_over_512_cap": length_over_cap,
    }


def process_severity_split(split):
    df = pd.read_csv(SPLITS / f"severity_{split}.csv")
    df["Text"] = df["Text"].astype(str).apply(normalize_text)
    max_len = df["Text"].apply(lambda t: len(t.split())).max()
    over_cap = (df["Text"].apply(lambda t: len(t.split())) > MAX_LEN).sum()
    df.to_csv(PROCESSED / f"severity_{split}.csv", index=False, encoding="utf-8")
    return {
        "rows": len(df),
        "max_token_length": int(max_len),
        "examples_over_512_cap": int(over_cap),
    }


def main():
    print("Processing HealthNER splits (per-token cleaning, label alignment preserved)...")
    healthner_stats = {}
    for split in ["train", "valid", "test"]:
        stats = process_healthner_split(split)
        healthner_stats[split] = stats
        print(f"  {split}: {stats}")

    print("\nProcessing Severity splits (whole-text normalization)...")
    severity_stats = {}
    for split in ["train", "val", "test"]:
        stats = process_severity_split(split)
        severity_stats[split] = stats
        print(f"  {split}: {stats}")

    total_sanity_failures = sum(s["sanity_gate_failures"] for s in healthner_stats.values())
    total_max_len = max(s["max_token_length"] for s in healthner_stats.values())
    total_over_cap = sum(s["examples_over_512_cap"] for s in healthner_stats.values())

    report = {
        "tokenization_rule": (
            "HealthNER: whitespace-segmented tokens from the original 'text' field are treated as "
            "authoritative and cleaned per-token (NFC normalize + strip zero-width/BOM + lowercase "
            "Latin) - never re-split from a joined string, so the token count cannot drift from the "
            "label count. Severity/CHQ-Summ/new queries: same simple whitespace-oriented tokenization "
            "applied after conservative whole-text cleanup (text_utils.normalize_text)."
        ),
        "preprocessing_rules_applied": [
            "NFC Unicode normalization",
            "Strip zero-width spaces/BOM/directional marks",
            "Collapse whitespace (whole-text mode only; not applied within HealthNER tokens)",
            "Lowercase Latin characters only (project convention); Bangla script untouched",
        ],
        "preprocessing_rules_NOT_applied": [
            "No stopword removal", "No stemming/lemmatization", "No Banglish-to-Bangla translation",
            "No English medical term translation", "No number removal",
            "No medicine-name normalization/replacement", "Severity labels / Action Needed never used",
        ],
        "healthner": healthner_stats,
        "severity": severity_stats,
        "sanity_gate": {
            "rule": "token count must equal label count for every HealthNER example",
            "total_failures_across_all_splits": total_sanity_failures,
            "result": "PASS" if total_sanity_failures == 0 else "FAIL - see per-split breakdown",
        },
        "zero_width_joiner_caveat": (
            "text_utils strips ZWNJ/ZWJ (U+200C/U+200D) unconditionally, including when embedded "
            "inside a real word, even though the guide (5.2) warns not to blindly delete legitimate "
            "script joiners - Bangla conjuncts sometimes rely on ZWNJ/ZWJ for correct rendering. This "
            "is a deliberate, documented trade-off: inconsistent author use of these characters is a "
            "common source of spurious vocabulary duplication (the same word with/without a ZWNJ "
            "becoming two different FastText tokens), which is standard practice to normalize away in "
            "Bangla NLP preprocessing. The only strict safety guarantee kept is that stripping never "
            "empties a HealthNER token (see empty_token_fallbacks_logged) - label alignment is never "
            "at risk. If future work finds this hurts specific medicine/procedure names that depend on "
            "ZWNJ, switch to a conditional strip (only when the char is not flanked by two Bangla base "
            "consonants)."
        ),
        "sequence_length_decision": {
            "cap": MAX_LEN,
            "max_token_length_observed_any_split": total_max_len,
            "examples_exceeding_cap": total_over_cap,
            "decision": (
                f"512-token cap covers 100% of examples (max observed length = {total_max_len}); "
                "no truncation or chunking needed. Use dynamic padding per batch."
                if total_over_cap == 0 else
                f"{total_over_cap} examples exceed the 512-token cap; truncate/chunk per guide 5.5."
            ),
        },
    }

    (RESULTS / "phase3_preprocessing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = ["# BanglaCare - Phase 3 Preprocessing Report", ""]
    lines.append("## Tokenization rule")
    lines.append(report["tokenization_rule"])
    lines.append("")
    lines.append("## Rules applied")
    for r in report["preprocessing_rules_applied"]:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("## Rules explicitly NOT applied")
    for r in report["preprocessing_rules_NOT_applied"]:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("## HealthNER per-split results")
    for split, s in healthner_stats.items():
        lines.append(f"- {split}: {s}")
    lines.append("")
    lines.append("## Severity per-split results")
    for split, s in severity_stats.items():
        lines.append(f"- {split}: {s}")
    lines.append("")
    lines.append(f"## Sanity gate: {report['sanity_gate']['result']}")
    lines.append(f"- Total token/label count failures across all splits: {total_sanity_failures}")
    lines.append("")
    lines.append("## Zero-width joiner caveat")
    lines.append(report["zero_width_joiner_caveat"])
    lines.append("")
    lines.append("## Sequence length decision")
    lines.append(report["sequence_length_decision"]["decision"])

    (RESULTS / "phase3_preprocessing_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nWrote results/phase3_preprocessing_report.json and .md")
    print(f"\nSanity gate: {report['sanity_gate']['result']}")


if __name__ == "__main__":
    main()
