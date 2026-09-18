"""
Severity classification metrics (guide 15.1).

Mirrors ner_eval.py for the other head: Phase 9 selects on validation Macro-F1,
Phase 10 uses Macro-F1 as its primary selection metric, Phase 11 refits a
temperature on the stored logits, and Phase 12 reports the full table.

Metrics produced (guide 15.1):
  - Macro-F1          primary; treats all four classes equally
  - Accuracy          easy overall summary
  - Weighted-F1       frequency-weighted summary
  - per-class P/R/F1  shows which classes are difficult
  - Emergency Recall  safety-relevant high-priority miss rate
  - confusion matrix  systematic class confusions
  - ECE               whether the confidence is calibrated

Raw logits are retained (the validation split is only ~526 rows), because
Phase 11 needs them to fit the temperature and Phase 12 needs them for the
reliability diagram.

Implemented with numpy only - no sklearn - to match the rest of the repo and
avoid another pinned dependency in the Kaggle environment.
"""

import numpy as np

from labels import SEVERITY_LABELS


def softmax(logits, temperature=1.0):
    """Row-wise softmax with optional temperature (guide 14.2)."""
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def confusion_matrix(y_true, y_pred, n_classes=len(SEVERITY_LABELS)):
    """cm[i, j] = count of gold class i predicted as class j."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def expected_calibration_error(probs, labels, n_bins=15):
    """ECE: mean |confidence - accuracy| over equal-width confidence bins.

    Also returns the per-bin data the reliability diagram needs (guide 15.1).
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    n = len(labels)

    for lo, hi in zip(edges[:-1], edges[1:]):
        # Upper edge inclusive on the last bin so confidence == 1.0 is counted.
        in_bin = (confidences > lo) & (confidences <= hi) if hi < 1.0 else (
            (confidences > lo) & (confidences <= 1.0)
        )
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0,
                         "confidence": 0.0, "accuracy": 0.0})
            continue
        bin_conf = float(confidences[in_bin].mean())
        bin_acc = float(correct[in_bin].mean())
        ece += (count / n) * abs(bin_conf - bin_acc)
        bins.append({"lo": float(lo), "hi": float(hi), "count": count,
                     "confidence": bin_conf, "accuracy": bin_acc})

    return float(ece), bins


class SeverityEvaluator:
    """Accumulates logits + gold labels across a validation/test pass."""

    def __init__(self):
        self._logits = []
        self._labels = []

    def add_batch(self, logits, labels):
        """logits: (B, 4) tensor/array. labels: (B,) tensor/array of class ids."""
        self._logits.append(np.asarray(
            logits.detach().cpu().numpy() if hasattr(logits, "detach") else logits,
            dtype=np.float32,
        ))
        self._labels.append(np.asarray(
            labels.detach().cpu().numpy() if hasattr(labels, "detach") else labels,
            dtype=np.int64,
        ))

    @property
    def logits(self):
        return np.concatenate(self._logits, axis=0) if self._logits else np.zeros((0, 4))

    @property
    def labels(self):
        return np.concatenate(self._labels, axis=0) if self._labels else np.zeros((0,), int)

    def compute(self, temperature=1.0, n_bins=15):
        logits = self.logits
        labels = self.labels
        if len(labels) == 0:
            raise ValueError("no batches added")

        probs = softmax(logits, temperature)
        preds = probs.argmax(axis=1)
        cm = confusion_matrix(labels, preds)

        accuracy = float((preds == labels).mean())

        per_class = {}
        f1s, supports = [], []
        for i, name in enumerate(SEVERITY_LABELS):
            tp = int(cm[i, i])
            fn = int(cm[i, :].sum() - tp)
            fp = int(cm[:, i].sum() - tp)
            p, r, f = _prf(tp, fp, fn)
            support = tp + fn
            per_class[name] = {
                "precision": p, "recall": r, "f1": f,
                "support": support, "tp": tp, "fp": fp, "fn": fn,
            }
            f1s.append(f)
            supports.append(support)

        macro_f1 = float(np.mean(f1s))
        total = sum(supports)
        weighted_f1 = float(
            sum(f * s for f, s in zip(f1s, supports)) / total
        ) if total else 0.0

        ece, bins = expected_calibration_error(probs, labels, n_bins)

        return {
            "n": int(len(labels)),
            "temperature": float(temperature),
            "accuracy": accuracy,
            "macro_f1": macro_f1,          # primary metric (guide 15.1)
            "weighted_f1": weighted_f1,
            # Emergency is class 0 and is the safety-critical one: a missed
            # emergency is the costliest error this system can make.
            "emergency_recall": per_class["Emergency"]["recall"],
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "ece": ece,
            "reliability_bins": bins,
            "mean_confidence": float(probs.max(axis=1).mean()),
        }


def format_report(metrics, title="Severity evaluation"):
    """Markdown report in the same house style as the other phases."""
    lines = [
        f"# {title}",
        "",
        f"- Examples: {metrics['n']}",
        f"- Temperature: {metrics['temperature']:.4f}"
        + ("  (uncalibrated)" if metrics["temperature"] == 1.0 else "  (calibrated)"),
        "",
        "## Headline",
        "",
        f"- **Macro-F1: {metrics['macro_f1']:.4f}**",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Weighted-F1: {metrics['weighted_f1']:.4f}",
        f"- **Emergency Recall: {metrics['emergency_recall']:.4f}**",
        f"- ECE: {metrics['ece']:.4f} (mean confidence {metrics['mean_confidence']:.4f})",
        "",
        "## Per class",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for name in SEVERITY_LABELS:
        m = metrics["per_class"][name]
        lines.append(
            f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} "
            f"| {m['f1']:.4f} | {m['support']} |"
        )

    lines += ["", "## Confusion matrix", "", "Rows = gold, columns = predicted.", ""]
    header = "| gold \\ pred | " + " | ".join(SEVERITY_LABELS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(SEVERITY_LABELS) + 1))
    for i, name in enumerate(SEVERITY_LABELS):
        row = metrics["confusion_matrix"][i]
        lines.append(f"| **{name}** | " + " | ".join(str(v) for v in row) + " |")

    return "\n".join(lines)
