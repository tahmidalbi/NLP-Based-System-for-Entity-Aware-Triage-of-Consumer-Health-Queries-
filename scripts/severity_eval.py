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


def _nll(logits, labels, temperature):
    """Mean negative log-likelihood of the gold class under softmax(logits/T)."""
    probs = softmax(logits, temperature)
    picked = probs[np.arange(len(labels)), labels]
    picked = np.clip(picked, 1e-12, 1.0)  # guard log(0) from a saturated softmax
    return float(-np.log(picked).mean())


def fit_temperature(logits, labels, t_min=0.05, t_max=5.0, coarse_steps=200, refine_rounds=6):
    """Scalar temperature scaling (guide 14.2): the T minimizing validation NLL.

    Pure numpy grid search rather than a gradient-based fit (e.g. LBFGS on
    log T): NLL(T) for a fixed, tiny 4-class logit set is a cheap 1D function
    to evaluate a few hundred times, and a search needs no autograd, no
    convexity assumption, and cannot diverge. Coarse log-spaced pass to find
    the neighbourhood, then a few rounds of golden-section-style bisection to
    refine it.

    Temperature scaling does not change which class wins the softmax (guide
    14.2), so this must be fit AFTER checkpoint selection, only to calibrate
    the confidence number - never used to pick the checkpoint itself.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    grid = np.geomspace(t_min, t_max, coarse_steps)
    losses = [_nll(logits, labels, t) for t in grid]
    best_idx = int(np.argmin(losses))
    best_t = float(grid[best_idx])

    lo = grid[max(best_idx - 1, 0)]
    hi = grid[min(best_idx + 1, len(grid) - 1)]
    for _ in range(refine_rounds):
        candidates = np.linspace(lo, hi, 9)
        cand_losses = [_nll(logits, labels, t) for t in candidates]
        i = int(np.argmin(cand_losses))
        best_t = float(candidates[i])
        lo = candidates[max(i - 1, 0)]
        hi = candidates[min(i + 1, len(candidates) - 1)]

    return best_t, _nll(logits, labels, best_t)


def choose_review_threshold(confidences, is_error):
    """Data-driven low-confidence threshold (guide 14.3): never hard-code 0.5/0.7.

    Frames "should this prediction be flagged for human review" as a binary
    detector built on confidence alone, with ground truth = "the model's
    prediction was wrong". Sweeps every confidence value seen in validation as
    a candidate threshold (flag if confidence < threshold) and keeps the one
    maximising Youden's J = TPR - FPR, i.e. the threshold that best separates
    the model's errors from its correct predictions using confidence alone.

    Returns (threshold, stats) where stats reports what that threshold would
    actually do on the validation set - so the choice can be sanity-checked
    (see phase11's report) instead of trusted blindly.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    is_error = np.asarray(is_error, dtype=bool)
    n_errors = int(is_error.sum())
    n_correct = int((~is_error).sum())

    if n_errors == 0 or n_correct == 0:
        # Degenerate validation set (all correct or all wrong): no confidence
        # threshold can separate the two classes, so fall back to flagging
        # nothing rather than fabricating a boundary from no signal.
        return 0.0, {
            "n_errors": n_errors, "n_correct": n_correct,
            "youden_j": 0.0, "note": "degenerate validation set; threshold disabled",
        }

    candidates = np.unique(confidences)
    best_j, best_t = -1.0, float(candidates.min())

    for t in candidates:
        flagged = confidences < t
        tpr = float((flagged & is_error).sum()) / n_errors     # errors caught
        fpr = float((flagged & ~is_error).sum()) / n_correct   # correct ones needlessly flagged
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, float(t)

    flagged = confidences < best_t
    stats = {
        "n_errors": n_errors,
        "n_correct": n_correct,
        "youden_j": best_j,
        "errors_caught_recall": float((flagged & is_error).sum()) / n_errors,
        "correct_needlessly_flagged_rate": float((flagged & ~is_error).sum()) / n_correct,
        "pct_of_validation_flagged": float(flagged.mean()),
    }
    return best_t, stats


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
