"""
Tests for scripts/ner_eval.py.

Run with:  python scripts/test_ner_eval.py

No pytest dependency and no GPU - plain asserts and a counter, so this can be
run on any machine (including inside a Kaggle notebook) before trusting a
single NER number.

Why this file exists: if extract_spans() is subtly wrong, every NER figure in
the final report is wrong and nothing crashes to tell you. The adjacent
same-type case (B-X B-X) in particular is silently merged into one span by a
naive implementation, which inflates precision.
"""

import sys
import traceback

from labels import NER_LABEL2ID
from ner_eval import (
    NEREvaluator,
    evaluate_ner,
    extract_spans,
    ids_to_labels,
    parse_label,
)

_passed = 0
_failed = 0


def check(name, fn):
    global _passed, _failed
    try:
        fn()
    except Exception:
        _failed += 1
        print(f"  FAIL  {name}")
        print("        " + traceback.format_exc().strip().replace("\n", "\n        "))
    else:
        _passed += 1
        print(f"  ok    {name}")


# ---------------------------------------------------------------- parse_label


def test_parse_plain():
    assert parse_label("B-Symptom") == ("B", "Symptom")
    assert parse_label("I-Medicine") == ("I", "Medicine")


def test_parse_type_containing_a_space():
    # The real reason parse_label splits on the FIRST hyphen only.
    assert parse_label("B-Health Condition") == ("B", "Health Condition")
    assert parse_label("I-Medical Procedure") == ("I", "Medical Procedure")


def test_parse_non_entity():
    assert parse_label("O") == (None, None)
    assert parse_label("") == (None, None)
    assert parse_label("garbage") == (None, None)
    assert parse_label("X-Symptom") == (None, None)  # bad prefix
    assert parse_label("B-") == (None, None)         # empty type


# -------------------------------------------------------------- extract_spans


def test_empty_sequence():
    assert extract_spans([]) == set()


def test_all_O():
    assert extract_spans(["O", "O", "O"]) == set()


def test_span_at_position_zero():
    labels = ["B-Symptom", "I-Symptom", "O"]
    assert extract_spans(labels) == {("Symptom", 0, 2)}


def test_span_running_to_end_of_sequence():
    # No trailing O to close the span - the final flush must catch it.
    labels = ["O", "B-Age", "I-Age"]
    assert extract_spans(labels) == {("Age", 1, 3)}


def test_single_token_span():
    labels = ["O", "B-Medicine", "O"]
    assert extract_spans(labels) == {("Medicine", 1, 2)}


def test_adjacent_same_type_spans():
    # THE classic bug: B-X immediately followed by B-X is TWO entities.
    # A naive implementation that only closes on O or type-change merges them.
    labels = ["B-Symptom", "B-Symptom"]
    assert extract_spans(labels) == {("Symptom", 0, 1), ("Symptom", 1, 2)}


def test_adjacent_same_type_multitoken():
    labels = ["B-Symptom", "I-Symptom", "B-Symptom", "I-Symptom"]
    assert extract_spans(labels) == {("Symptom", 0, 2), ("Symptom", 2, 4)}


def test_adjacent_different_type_spans():
    labels = ["B-Symptom", "B-Medicine"]
    assert extract_spans(labels) == {("Symptom", 0, 1), ("Medicine", 1, 2)}


def test_dangling_I_at_position_zero_lenient():
    labels = ["I-Symptom", "I-Symptom", "O"]
    assert extract_spans(labels, strict=False) == {("Symptom", 0, 2)}


def test_dangling_I_at_position_zero_strict():
    labels = ["I-Symptom", "I-Symptom", "O"]
    assert extract_spans(labels, strict=True) == set()


def test_dangling_I_after_O_lenient():
    labels = ["O", "I-Medicine", "O"]
    assert extract_spans(labels, strict=False) == {("Medicine", 1, 2)}


def test_dangling_I_after_O_strict():
    labels = ["O", "I-Medicine", "O"]
    assert extract_spans(labels, strict=True) == set()


def test_I_after_different_type_lenient():
    # The open Symptom span must still be closed at index 1, and a new
    # Medicine span opened there.
    labels = ["B-Symptom", "I-Medicine"]
    assert extract_spans(labels, strict=False) == {("Symptom", 0, 1), ("Medicine", 1, 2)}


def test_I_after_different_type_strict():
    labels = ["B-Symptom", "I-Medicine"]
    assert extract_spans(labels, strict=True) == {("Symptom", 0, 1)}


def test_strict_does_not_drop_the_span_before_a_dangling_I():
    # Regression guard: discarding the invalid I- must not also discard the
    # perfectly valid span that preceded it.
    labels = ["B-Age", "I-Age", "I-Dosage", "O", "B-Symptom"]
    assert extract_spans(labels, strict=True) == {("Age", 0, 2), ("Symptom", 4, 5)}


def test_type_with_space_end_to_end():
    labels = ["B-Health Condition", "I-Health Condition", "O", "B-Medical Procedure"]
    assert extract_spans(labels) == {
        ("Health Condition", 0, 2),
        ("Medical Procedure", 3, 4),
    }


def test_malformed_label_closes_span_like_O():
    labels = ["B-Symptom", "I-Symptom", "junk", "B-Symptom"]
    assert extract_spans(labels) == {("Symptom", 0, 2), ("Symptom", 3, 4)}


def test_all_seven_types_parse():
    from labels import ENTITY_TYPES

    labels = []
    for etype in ENTITY_TYPES:
        labels += [f"B-{etype}", f"I-{etype}", "O"]
    spans = extract_spans(labels)
    assert len(spans) == len(ENTITY_TYPES) == 7
    assert {s[0] for s in spans} == set(ENTITY_TYPES)


# ------------------------------------------------------------- ids_to_labels


def test_ids_to_labels_roundtrip():
    labels = ["O", "B-Symptom", "I-Symptom", "B-Medical Procedure"]
    ids = [NER_LABEL2ID[l] for l in labels]
    assert ids_to_labels(ids) == labels


def test_ids_to_labels_accepts_numpy_style_ints():
    # CRF.decode() returns plain ints, but .tolist() on a tensor can hand back
    # numpy scalars in some paths - int() coercion must absorb that.
    class FakeInt:
        def __init__(self, v):
            self.v = v

        def __int__(self):
            return self.v

    assert ids_to_labels([FakeInt(0), FakeInt(1)]) == ["O", "B-Symptom"]


# ------------------------------------------------------------------- metrics


def test_perfect_prediction():
    gold = [["B-Symptom", "I-Symptom", "O", "B-Medicine"]]
    m = evaluate_ner(gold, gold)
    assert m["entity_precision"] == 1.0
    assert m["entity_recall"] == 1.0
    assert m["entity_f1"] == 1.0
    assert m["gold_spans"] == 2 and m["predicted_spans"] == 2


def test_completely_wrong_prediction():
    gold = [["B-Symptom", "I-Symptom", "O"]]
    pred = [["O", "O", "B-Medicine"]]
    m = evaluate_ner(gold, pred)
    assert m["entity_f1"] == 0.0
    assert m["per_type"]["Symptom"]["fn"] == 1
    assert m["per_type"]["Medicine"]["fp"] == 1


def test_boundary_error_counts_as_both_fp_and_fn():
    # Gold is a 2-token span, prediction is a 1-token span at the same start.
    # Exact-match span evaluation must give zero credit, not partial credit.
    gold = [["B-Symptom", "I-Symptom"]]
    pred = [["B-Symptom", "O"]]
    m = evaluate_ner(gold, pred)
    assert m["entity_f1"] == 0.0
    assert m["per_type"]["Symptom"]["tp"] == 0
    assert m["per_type"]["Symptom"]["fp"] == 1
    assert m["per_type"]["Symptom"]["fn"] == 1


def test_right_boundary_wrong_type():
    gold = [["B-Symptom", "I-Symptom"]]
    pred = [["B-Medicine", "I-Medicine"]]
    m = evaluate_ner(gold, pred)
    assert m["entity_f1"] == 0.0
    assert m["per_type"]["Symptom"]["fn"] == 1
    assert m["per_type"]["Medicine"]["fp"] == 1


def test_no_predictions_at_all_does_not_crash():
    gold = [["B-Symptom", "I-Symptom"]]
    pred = [["O", "O"]]
    m = evaluate_ner(gold, pred)
    assert m["entity_precision"] == 0.0   # 0/0 defined as 0.0
    assert m["entity_recall"] == 0.0
    assert m["entity_f1"] == 0.0


def test_no_gold_and_no_pred_does_not_crash():
    m = evaluate_ner([["O", "O"]], [["O", "O"]])
    assert m["entity_f1"] == 0.0
    assert m["macro_f1"] == 0.0


def test_partial_credit_across_two_sequences():
    gold = [
        ["B-Symptom", "I-Symptom", "O"],   # 1 gold span
        ["B-Medicine", "O", "B-Age"],      # 2 gold spans
    ]
    pred = [
        ["B-Symptom", "I-Symptom", "O"],   # correct
        ["B-Medicine", "O", "O"],          # Age missed
    ]
    m = evaluate_ner(gold, pred)
    assert m["gold_spans"] == 3
    assert m["predicted_spans"] == 2
    assert m["entity_precision"] == 1.0            # 2/2
    assert abs(m["entity_recall"] - 2 / 3) < 1e-12  # 2/3
    assert m["per_type"]["Age"]["f1"] == 0.0
    assert m["n_sequences"] == 2


def test_macro_ignores_types_absent_from_gold():
    # Only Symptom occurs in gold; macro must be that type's F1 (1.0), not
    # 1/7 of it.
    gold = [["B-Symptom"]]
    pred = [["B-Symptom"]]
    m = evaluate_ner(gold, pred)
    assert m["macro_f1"] == 1.0


def test_per_type_support_counts():
    gold = [["B-Symptom", "B-Symptom", "B-Age"]]
    pred = [["O", "O", "O"]]
    m = evaluate_ner(gold, pred)
    assert m["per_type"]["Symptom"]["support"] == 2
    assert m["per_type"]["Age"]["support"] == 1
    assert m["per_type"]["Medicine"]["support"] == 0


def test_token_level_metric():
    gold = [["B-Symptom", "I-Symptom", "O"]]
    pred = [["B-Symptom", "O", "O"]]
    m = evaluate_ner(gold, pred)
    # non-O gold positions: 2, of which 1 matches exactly
    assert abs(m["token_precision"] - 1.0) < 1e-12   # 1 correct / 1 predicted
    assert abs(m["token_recall"] - 0.5) < 1e-12      # 1 correct / 2 gold
    assert abs(m["token_f1"] - 2 / 3) < 1e-12


def test_length_mismatch_raises():
    ev = NEREvaluator()
    try:
        ev.add(["O", "O"], ["O"])
    except ValueError:
        return
    raise AssertionError("expected ValueError on gold/pred length mismatch")


def test_strict_flag_reaches_the_metrics():
    gold = [["I-Symptom", "I-Symptom"]]
    pred = [["I-Symptom", "I-Symptom"]]
    assert evaluate_ner(gold, pred, strict=False)["entity_f1"] == 1.0
    # Under strict decoding neither side has any span at all.
    strict = evaluate_ner(gold, pred, strict=True)
    assert strict["gold_spans"] == 0 and strict["predicted_spans"] == 0


# ----------------------------------------------------------------- add_batch


def test_add_batch_trims_padding():
    # Batch of 2, padded to T=4. Gold row 2 has 2 real tokens; the padded
    # positions carry id 0 ('O') and must not be scored.
    gold_tags = [
        [NER_LABEL2ID["B-Symptom"], NER_LABEL2ID["I-Symptom"], 0, 0],
        [NER_LABEL2ID["B-Age"], 0, 0, 0],
    ]
    lengths = [2, 1]
    pred_ids_list = [
        [NER_LABEL2ID["B-Symptom"], NER_LABEL2ID["I-Symptom"]],
        [NER_LABEL2ID["B-Age"]],
    ]
    ev = NEREvaluator()
    ev.add_batch(pred_ids_list, gold_tags, lengths)
    m = ev.compute()
    assert m["entity_f1"] == 1.0
    assert m["n_tokens"] == 3          # 2 + 1, not 8
    assert m["gold_spans"] == 2


def test_add_batch_rejects_untrimmed_predictions():
    # Passing padded predictions instead of CRF.decode() output is a real and
    # easy mistake; it must fail loudly rather than score padding.
    gold_tags = [[NER_LABEL2ID["B-Symptom"], 0, 0]]
    lengths = [1]
    pred_padded = [[NER_LABEL2ID["B-Symptom"], 0, 0]]
    ev = NEREvaluator()
    try:
        ev.add_batch(pred_padded, gold_tags, lengths)
    except ValueError:
        return
    raise AssertionError("expected ValueError on untrimmed prediction lengths")


def test_add_batch_accepts_tolist_objects():
    class FakeTensor:
        def __init__(self, data):
            self.data = data

        def tolist(self):
            return self.data

    gold_tags = FakeTensor([[NER_LABEL2ID["B-Symptom"], 0]])
    lengths = FakeTensor([1])
    ev = NEREvaluator()
    ev.add_batch([[NER_LABEL2ID["B-Symptom"]]], gold_tags, lengths)
    assert ev.compute()["entity_f1"] == 1.0


def main():
    print("ner_eval tests")
    print("=" * 60)
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        check(name, fn)

    print("=" * 60)
    print(f"{_passed} passed, {_failed} failed, {len(tests)} total")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
