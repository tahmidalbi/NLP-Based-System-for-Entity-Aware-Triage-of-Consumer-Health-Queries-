# BanglaCare - Phase 11 calibration

- Checkpoint: `/kaggle/working/checkpoints/best_joint_transformer.pt`
- Fitted on: severity validation only (526 examples) - test sets untouched (guide 14.1)

## Temperature scaling

- **Temperature: 1.2142**
- Validation NLL: 0.1984 -> 0.1939
- ECE: 0.0195 -> 0.0119
- Fit converged inside the search range.

## Low-confidence review threshold

Chosen by maximising Youden's J (guide 14.3: never hard-code 0.50/0.70) - the threshold that best separates the model's own errors from its correct predictions using calibrated confidence alone.

- **Flag for human review when confidence < 0.9392**
- Catches 100.0% of validation errors
- Needlessly flags 25.4% of correct predictions
- 30.6% of validation would be flagged

## Reliability diagram: figures/transformer_reliability_plot.png