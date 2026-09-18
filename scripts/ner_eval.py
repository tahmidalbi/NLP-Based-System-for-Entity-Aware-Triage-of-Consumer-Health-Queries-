"""
Entity-level NER evaluation (guide 11.4 and 15.2).

Phase 8 selects checkpoints on validation entity-level F1, Phase 10 uses the
same number as its "NER must not collapse" guardrail, and Phase 12 reports it
as the primary NER metric. All three go through this module, so span
extraction is defined exactly once.

Metrics produced (guide 15.2):
  - entity-level precision / recall / F1   (micro over all types; primary)
  - per-entity F1 for all seven types      (HealthNER is imbalanced - the
                                            guide requires per-type scores)
  - macro F1 over the types present in gold
  - token-level micro F1 over non-O positions (optional comparison metric)

IOB decoding convention
-----------------------
A span is (entity_type, start, end) with `end` exclusive. Two conventions
exist for a dangling `I-X` (an I- tag at position 0, after an `O`, or after a
different entity type):

  strict=False (default)  `I-X` opens a new span. This is what conlleval and
                          seqeval's default mode do, so the numbers stay
                          comparable with prior HealthNER work.
  strict=True             `I-X` is discarded as an invalid transition.

The default matters in practice: the CRF is free to emit any label sequence,
and an untrained model emits dangling I- tags constantly (see
results/phase7_verification.txt). Whichever convention is used must be stated
in the report, because the two produce different numbers.

A malformed or unknown label is treated as `O`.
"""

from collections import defaultdict

from labels import ENTITY_TYPES, NER_ID2LABEL


def parse_label(label):
    """'B-Health Condition' -> ('B', 'Health Condition'). 'O'/junk -> (None, None).

    Splits on the FIRST hyphen only, because entity types themselves contain
    spaces and must survive intact ('Health Condition', 'Medical Procedure').
    """
    if not label or label == "O":
        return None, None
    prefix, sep, etype = label.partition("-")
    if not sep or prefix not in ("B", "I") or not etype:
        return None, None
    return prefix, etype


def extract_spans(labels, strict=False):
    """IOB label sequence -> set of (entity_type, start, end_exclusive) spans."""
    spans = set()
    cur_type = None
    cur_start = None

    for i, label in enumerate(labels):
        prefix, etype = parse_label(label)

        if prefix == "B":
            # Always closes the previous span, even when the type is identical -
            # two adjacent B-X tags are two separate entities, not one.
            if cur_type is not None:
                spans.add((cur_type, cur_start, i))
            cur_type, cur_start = etype, i

        elif prefix == "I":
            if cur_type == etype:
                continue  # extend the open span
            if cur_type is not None:
                spans.add((cur_type, cur_start, i))
            cur_type, cur_start = (None, None) if strict else (etype, i)

        else:  # O, or a malformed label
            if cur_type is not None:
                spans.add((cur_type, cur_start, i))
            cur_type, cur_start = None, None

    if cur_type is not None:  # span running to the end of the sequence
        spans.add((cur_type, cur_start, len(labels)))
    return spans


def ids_to_labels(label_ids):
    """[0, 1, 8] -> ['O', 'B-Symptom', 'I-Symptom'] via labels.NER_ID2LABEL."""
    return [NER_ID2LABEL[int(i)] for i in label_ids]


def _prf(tp, fp, fn):
    """Precision/recall/F1 with the empty cases defined as 0.0, never a crash."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


class NEREvaluator:
    """Streaming accumulator - add batches during a validation pass, then compute().

    Stores only counters, so a full HealthNER split costs a few hundred bytes.
    """

    def __init__(self, strict=False):
        self.strict = strict
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)
        self.token_tp = 0
        self.token_fp = 0
        self.token_fn = 0
        self.n_sequences = 0
        self.n_tokens = 0

    def add(self, gold_labels, pred_labels):
        """One sequence, as two equal-length lists of label strings."""
        if len(gold_labels) != len(pred_labels):
            raise ValueError(
                f"gold/pred length mismatch: {len(gold_labels)} vs {len(pred_labels)}"
            )

        gold = extract_spans(gold_labels, self.strict)
        pred = extract_spans(pred_labels, self.strict)

        for etype, _, _ in pred & gold:
            self.tp[etype] += 1
        for etype, _, _ in pred - gold:
            self.fp[etype] += 1
        for etype, _, _ in gold - pred:
            self.fn[etype] += 1

        for g, p in zip(gold_labels, pred_labels):
            if g != "O" and p == g:
                self.token_tp += 1
            if p != "O" and p != g:
                self.token_fp += 1
            if g != "O" and p != g:
                self.token_fn += 1

        self.n_sequences += 1
        self.n_tokens += len(gold_labels)

    def add_ids(self, gold_ids, pred_ids):
        """Same as add(), for label-id sequences."""
        self.add(ids_to_labels(gold_ids), ids_to_labels(pred_ids))

    def add_batch(self, pred_ids_list, gold_tags, lengths):
        """Consume one collated batch straight from the training loop.

        pred_ids_list: what CRF.decode() returns - list of per-example id lists,
                       already trimmed to each example's real length.
        gold_tags:     padded (B, T) tensor/array of gold label ids.
        lengths:       (B,) real token counts.

        Works with torch tensors or plain lists via .tolist() duck-typing, so
        this module never imports torch.
        """
        gold_rows = gold_tags.tolist() if hasattr(gold_tags, "tolist") else list(gold_tags)
        length_list = lengths.tolist() if hasattr(lengths, "tolist") else list(lengths)

        for pred_ids, gold_row, n in zip(pred_ids_list, gold_rows, length_list):
            n = int(n)
            if len(pred_ids) != n:
                raise ValueError(
                    f"decoded length {len(pred_ids)} != real length {n}; pass the "
                    "CRF decode output and the same mask/lengths used in the forward pass"
                )
            self.add_ids(gold_row[:n], pred_ids)

    def compute(self):
        tp = sum(self.tp.values())
        fp = sum(self.fp.values())
        fn = sum(self.fn.values())
        precision, recall, f1 = _prf(tp, fp, fn)

        per_type = {}
        for etype in ENTITY_TYPES:
            t, f_p, f_n = self.tp[etype], self.fp[etype], self.fn[etype]
            p, r, f = _prf(t, f_p, f_n)
            per_type[etype] = {
                "precision": p,
                "recall": r,
                "f1": f,
                "support": t + f_n,  # gold spans of this type
                "predicted": t + f_p,
                "tp": t,
                "fp": f_p,
                "fn": f_n,
            }

        # Macro over types that actually occur in gold - a type with zero
        # support would otherwise drag the mean to 0 and hide real movement.
        supported = [m["f1"] for m in per_type.values() if m["support"] > 0]
        macro_f1 = sum(supported) / len(supported) if supported else 0.0

        tok_p, tok_r, tok_f1 = _prf(self.token_tp, self.token_fp, self.token_fn)

        return {
            "strict": self.strict,
            "n_sequences": self.n_sequences,
            "n_tokens": self.n_tokens,
            "entity_precision": precision,
            "entity_recall": recall,
            "entity_f1": f1,          # primary metric (guide 11.4)
            "micro_f1": f1,           # same quantity, named as in guide 15.2
            "macro_f1": macro_f1,
            "gold_spans": tp + fn,
            "predicted_spans": tp + fp,
            "per_type": per_type,
            "token_precision": tok_p,
            "token_recall": tok_r,
            "token_f1": tok_f1,
        }


def evaluate_ner(gold_sequences, pred_sequences, strict=False):
    """One-shot evaluation over two lists of label-string sequences."""
    ev = NEREvaluator(strict=strict)
    for gold, pred in zip(gold_sequences, pred_sequences):
        ev.add(gold, pred)
    return ev.compute()


def format_report(metrics, title="NER evaluation"):
    """Markdown report for results/ - same house style as the other phases."""
    decoding = "strict" if metrics["strict"] else "conlleval-style (dangling I- opens a span)"
    lines = [
        f"# {title}",
        "",
        f"- Sequences: {metrics['n_sequences']} ({metrics['n_tokens']} tokens)",
        f"- Gold spans: {metrics['gold_spans']} | predicted spans: {metrics['predicted_spans']}",
        f"- IOB decoding: {decoding}",
        "",
        "## Entity-level (primary)",
        "",
        f"- **Entity F1 (micro): {metrics['entity_f1']:.4f}**",
        f"- Precision: {metrics['entity_precision']:.4f}",
        f"- Recall: {metrics['entity_recall']:.4f}",
        f"- Macro F1 over types present in gold: {metrics['macro_f1']:.4f}",
        "",
        "## Per entity type",
        "",
        "| Entity type | Precision | Recall | F1 | Gold | Pred |",
        "|---|---|---|---|---|---|",
    ]
    for etype in ENTITY_TYPES:
        m = metrics["per_type"][etype]
        lines.append(
            f"| {etype} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} "
            f"| {m['support']} | {m['predicted']} |"
        )
    lines += [
        "",
        "## Token level (non-O positions, comparison metric only)",
        "",
        f"- Precision: {metrics['token_precision']:.4f}",
        f"- Recall: {metrics['token_recall']:.4f}",
        f"- F1: {metrics['token_f1']:.4f}",
    ]
    return "\n".join(lines)
