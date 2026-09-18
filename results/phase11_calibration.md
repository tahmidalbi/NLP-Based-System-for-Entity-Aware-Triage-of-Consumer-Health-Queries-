# BanglaCare - Phase 11 calibration

- Checkpoint: `/kaggle/working/bc/checkpoints/best_joint.pt`
- Fitted on: severity validation only (526 examples) - test sets untouched (guide 14.1)

## Temperature scaling

- **Temperature: 1.6358**
- Validation NLL: 0.2786 -> 0.2426
- ECE: 0.0491 -> 0.0149
- Fit converged inside the search range.

## Low-confidence review threshold

Chosen by maximising Youden's J (guide 14.3: never hard-code 0.50/0.70) - the threshold that best separates the model's own errors from its correct predictions using calibrated confidence alone.

- **Flag for human review when confidence < 0.9196**
- Catches 91.1% of validation errors
- Needlessly flags 26.6% of correct predictions
- 32.1% of validation would be flagged

## Reliability diagram: figures/reliability_plot.png