"""
Step 0 of TRANSFORMER_PLAN.md - subword length audit.

The BiLSTM ran on words; the transformer runs on subwords, and Bangla
morphology expands badly. Phase 3 found zero examples over 512 WORDS, but at
subword level some may exceed the 512 cap. This script measures it, and
prints the truncation decision the plan asks you to write down.

  python scripts/audit_subword_lengths.py
  python scripts/audit_subword_lengths.py --model xlm-roberta-base --no-normalize

Output: results/subword_length_audit.json / .md
"""

import argparse
import json

import numpy as np

from dataset_transformer import DEFAULT_MODEL, ROOT, WordEncoder, load_tokenizer, make_normalizer

RESULTS = ROOT / "results"


def audit_split(encoder, tokens_lists, max_len):
    """Subword length (incl. [CLS]/[SEP]) of every example, un-truncated."""
    tok = encoder.tokenizer
    lens = []
    for tokens in tokens_lists:
        words = encoder._prepare(tokens)
        ids = tok(words, is_split_into_words=True, truncation=False)["input_ids"]
        lens.append(len(ids))
    lens = np.array(lens)
    over = int((lens > max_len).sum())
    return {
        "n": int(len(lens)),
        "median": float(np.median(lens)),
        "p95": float(np.percentile(lens, 95)),
        "p99": float(np.percentile(lens, 99)),
        "max": int(lens.max()),
        "over_limit": over,
        "over_limit_pct": over / len(lens),
    }


def main():
    parser = argparse.ArgumentParser(description="Subword length audit")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    encoder = WordEncoder(tokenizer, make_normalizer(not args.no_normalize), args.max_len)

    stats = {}
    for split in ["train", "valid", "test"]:
        path = ROOT / "data" / "processed" / "healthner" / f"{split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        stats[f"ner_{split}"] = audit_split(encoder, [ex["tokens"] for ex in data], args.max_len)

    import pandas as pd
    for split in ["train", "val", "test"]:
        df = pd.read_csv(ROOT / "data" / "processed" / f"severity_{split}.csv", usecols=["Text"])
        stats[f"severity_{split}"] = audit_split(
            encoder, [str(t).split() for t in df["Text"]], args.max_len
        )

    print(f"model: {args.model} | limit: {args.max_len} subwords\n")
    for name, s in stats.items():
        print(f"{name:<16} n={s['n']:>6} median={s['median']:.0f} p95={s['p95']:.0f} "
              f"p99={s['p99']:.0f} max={s['max']} "
              f"over_{args.max_len}={s['over_limit']} ({s['over_limit_pct']:.2%})")

    worst = max(s["over_limit_pct"] for k, s in stats.items() if k.startswith("ner_"))
    if worst < 0.01:
        decision = ("Plain truncation is fine (<1% of NER examples exceed the cap). "
                    "Log the truncated counts (the loaders print them).")
    else:
        decision = ("MORE than 1% of NER examples exceed the cap. Plain truncation "
                    "silently deletes gold entities from long queries - a sliding "
                    "window with overlap is needed before trusting recall.")
    print(f"\nDECISION: {decision}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "subword_length_audit.json").write_text(
        json.dumps({"model": args.model, "max_len": args.max_len, "stats": stats,
                    "decision": decision}, indent=2), encoding="utf-8")
    lines = [f"# Subword length audit ({args.model}, cap {args.max_len})", "",
             "| Split | n | median | p95 | p99 | max | over cap |", "|---|---|---|---|---|---|---|"]
    for name, s in stats.items():
        lines.append(f"| {name} | {s['n']} | {s['median']:.0f} | {s['p95']:.0f} | "
                     f"{s['p99']:.0f} | {s['max']} | {s['over_limit']} ({s['over_limit_pct']:.2%}) |")
    lines += ["", f"**Decision:** {decision}"]
    (RESULTS / "subword_length_audit.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
