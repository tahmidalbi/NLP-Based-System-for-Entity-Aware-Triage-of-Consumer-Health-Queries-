"""
Phase 7 + Section 10 verification.

Builds real batches from the Phase 3 processed data through the actual
Dataset/Collator/Model classes and checks:
  1. Shapes at every stage match the guide's fixed architecture exactly.
  2. The CRF NER loss is finite and Viterbi-decoding produces valid label ids.
  3. The severity head produces (B, 4) logits.
  4. The "protect the NER head" gradient rule (guide 10.4) actually holds:
     a severity-only backward pass must leave the NER emission Linear layer's
     gradients at None/zero, while still updating the shared encoder and the
     severity head; an NER-only backward pass must leave the severity head's
     gradients at None.

Local RAM note: this machine has ~7.9GB total RAM, and the two real FastText
models together need >8GB to hold simultaneously. Rather than skip real data,
this script extracts genuine 300D vectors for exactly the tokens appearing in
a small real verification batch by loading each FastText model ONE AT A TIME
(freeing each before loading the next), then feeds those real vectors through
the full Dataset -> Collator -> Model pipeline. On Kaggle (the guide's target
platform, with far more RAM) FastTextFeaturizer's production mode loads both
models together, as intended.
"""

import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
from labels import ENTITY_TYPES, NER_LABELS, SEVERITY_LABELS  # noqa: E402

N_NER_EXAMPLES = 6
N_SEVERITY_EXAMPLES = 6


def collect_verification_examples():
    with open(ROOT.parent / "data" / "processed" / "healthner" / "train.json", encoding="utf-8") as f:
        ner_examples = json.load(f)[:N_NER_EXAMPLES]

    import pandas as pd
    df = pd.read_csv(ROOT.parent / "data" / "processed" / "severity_train.csv", usecols=["Text", "Categories"])
    severity_examples = df.head(N_SEVERITY_EXAMPLES).to_dict("records")

    return ner_examples, severity_examples


def extract_real_vectors(ner_examples, severity_examples):
    """Load each FastText model one at a time, extract 600D vectors for
    exactly the tokens needed, and return a plain dict {token: np.ndarray(600)}."""
    import numpy as np

    all_tokens = set()
    for ex in ner_examples:
        all_tokens.update(ex["tokens"])
    for row in severity_examples:
        all_tokens.update(str(row["Text"]).split())
    all_tokens = sorted(all_tokens)
    print(f"Verification vocabulary: {len(all_tokens)} unique tokens")

    import fasttext

    print("Loading general FastText (cc.bn.300.bin)...")
    general = fasttext.load_model(str(ROOT.parent / "embeddings" / "cc.bn.300.bin"))
    general_vecs = {t: general.get_word_vector(t) for t in all_tokens}
    del general
    gc.collect()
    print("  done, freed.")

    print("Loading medical FastText (medical_fasttext.bin)...")
    medical = fasttext.load_model(str(ROOT.parent / "embeddings" / "medical_fasttext.bin"))
    medical_vecs = {t: medical.get_word_vector(t) for t in all_tokens}
    del medical
    gc.collect()
    print("  done, freed.")

    cache = {t: np.concatenate([general_vecs[t], medical_vecs[t]]).astype(np.float32) for t in all_tokens}
    return cache


def main():
    ner_examples, severity_examples = collect_verification_examples()
    cache = extract_real_vectors(ner_examples, severity_examples)

    import torch

    from featurizer import FastTextFeaturizer
    from dataset import Collator
    from model import BanglaCareModel

    featurizer = FastTextFeaturizer(precomputed_cache=cache)
    print(f"\nFeaturizer dim: {featurizer.dim} (expect 600)")
    collator = Collator(featurizer)

    report_lines = ["BanglaCare - Phase 7 + Section 10 verification", "=" * 60, ""]

    def log(line=""):
        print(line)
        report_lines.append(line)

    log(f"Featurizer dim: {featurizer.dim} (expect 600)")

    # --- Build a real NER batch straight from processed HealthNER data ---
    ner_batch_items = [
        {"tokens": ex["tokens"], "ner_label_ids": [NER_LABELS.index(l) for l in ex["labels"]]}
        for ex in ner_examples
    ]
    ner_batch = collator(ner_batch_items)

    log("\n## NER batch (from data/processed/healthner/train.json)")
    log(f"  batch size: {len(ner_batch['tokens'])}")
    log(f"  features shape: {tuple(ner_batch['features'].shape)}  (expect (B, T, 600))")
    log(f"  mask shape: {tuple(ner_batch['mask'].shape)}")
    log(f"  ner_tags shape: {tuple(ner_batch['ner_tags'].shape)}")
    log(f"  lengths: {ner_batch['lengths'].tolist()}")
    log(f"  sample tokens[0][:6]: {ner_batch['tokens'][0][:6]}")

    # --- Build a real severity batch straight from processed severity data ---
    severity_batch_items = [
        {"tokens": str(row["Text"]).split(), "severity_id": SEVERITY_LABELS.index(row["Categories"])}
        for row in severity_examples
    ]
    severity_batch = collator(severity_batch_items)

    log("\n## Severity batch (from data/processed/severity_train.csv)")
    log(f"  batch size: {len(severity_batch['tokens'])}")
    log(f"  features shape: {tuple(severity_batch['features'].shape)}  (expect (B, T, 600))")
    log(f"  mask shape: {tuple(severity_batch['mask'].shape)}")
    log(f"  severity_ids: {severity_batch['severity_ids'].tolist()} -> "
        f"{[SEVERITY_LABELS[i] for i in severity_batch['severity_ids'].tolist()]}")

    # --- Build the model ---
    torch.manual_seed(42)
    model = BanglaCareModel()
    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.encoder.parameters())
    n_ner = sum(p.numel() for p in model.ner_head.parameters())
    n_sev = sum(p.numel() for p in model.severity_head.parameters())
    log("\n## Model parameter counts")
    log(f"  total: {n_params:,}")
    log(f"  encoder (projection+BiLSTM): {n_encoder:,}")
    log(f"  ner_head (emission+CRF): {n_ner:,}")
    log(f"  severity_head: {n_sev:,}")

    # --- Forward pass: NER batch ---
    log("\n## NER forward pass")
    emissions, h_ner = model.ner_forward(ner_batch["features"], ner_batch["lengths"], ner_batch["mask"])
    log(f"  encoder output h shape: {tuple(h_ner.shape)}  (expect (B, T, 384))")
    log(f"  NER emissions shape: {tuple(emissions.shape)}  (expect (B, T, 15))")

    ner_loss = model.ner_head.loss(emissions, ner_batch["ner_tags"], ner_batch["mask"])
    log(f"  CRF NLL loss: {ner_loss.item():.4f} (finite: {torch.isfinite(ner_loss).item()})")

    decoded = model.ner_head.decode(emissions, ner_batch["mask"])
    valid_ids = all(0 <= lid < len(NER_LABELS) for seq in decoded for lid in seq)
    lengths_match = all(len(seq) == n for seq, n in zip(decoded, ner_batch["lengths"].tolist()))
    log(f"  CRF decode: {len(decoded)} sequences, all label ids valid: {valid_ids}, "
        f"decoded lengths match input lengths: {lengths_match}")
    log(f"  decoded[0] -> labels: {[NER_LABELS[i] for i in decoded[0]]}")

    # --- Forward pass: severity batch ---
    log("\n## Severity forward pass")
    severity_logits, h_sev = model.severity_forward(
        severity_batch["features"], severity_batch["lengths"], severity_batch["mask"]
    )
    log(f"  encoder output h shape: {tuple(h_sev.shape)}  (expect (B, T, 384))")
    log(f"  severity logits shape: {tuple(severity_logits.shape)}  (expect (B, 4))")

    sev_loss_fn = torch.nn.CrossEntropyLoss()
    severity_loss = sev_loss_fn(severity_logits, severity_batch["severity_ids"])
    log(f"  cross-entropy loss: {severity_loss.item():.4f} (finite: {torch.isfinite(severity_loss).item()})")

    # --- Gradient protection rule check (guide 10.4) ---
    log("\n## Gradient protection rule check (guide 10.4: 'Protect the NER head')")

    model.zero_grad()
    severity_logits, _ = model.severity_forward(
        severity_batch["features"], severity_batch["lengths"], severity_batch["mask"]
    )
    loss = sev_loss_fn(severity_logits, severity_batch["severity_ids"])
    loss.backward()

    ner_emission_grad = model.ner_head.emission.weight.grad
    ner_emission_untouched = ner_emission_grad is None or torch.all(ner_emission_grad == 0)
    encoder_grad = model.encoder.projection[0].weight.grad
    encoder_updated = encoder_grad is not None and torch.any(encoder_grad != 0)
    sev_head_grad = model.severity_head.mlp[0].weight.grad
    sev_head_updated = sev_head_grad is not None and torch.any(sev_head_grad != 0)

    log(f"  after severity-only backward:")
    log(f"    NER emission layer grad is None/zero (should be True): {ner_emission_untouched}")
    log(f"    shared encoder projection grad is nonzero (should be True): {encoder_updated}")
    log(f"    severity head grad is nonzero (should be True): {sev_head_updated}")

    model.zero_grad()
    emissions2, _ = model.ner_forward(ner_batch["features"], ner_batch["lengths"], ner_batch["mask"])
    ner_loss2 = model.ner_head.loss(emissions2, ner_batch["ner_tags"], ner_batch["mask"])
    ner_loss2.backward()

    sev_head_grad2 = model.severity_head.mlp[0].weight.grad
    sev_head_untouched = sev_head_grad2 is None or torch.all(sev_head_grad2 == 0)
    ner_emission_grad2 = model.ner_head.emission.weight.grad
    ner_emission_updated2 = ner_emission_grad2 is not None and torch.any(ner_emission_grad2 != 0)

    log(f"  after NER-only backward:")
    log(f"    severity head grad is None/zero (should be True, unused in this forward): {sev_head_untouched}")
    log(f"    NER emission layer grad is nonzero (should be True): {ner_emission_updated2}")

    all_checks = [
        ner_emission_untouched, encoder_updated, sev_head_updated,
        sev_head_untouched, ner_emission_updated2,
        valid_ids, lengths_match,
        torch.isfinite(ner_loss).item(), torch.isfinite(severity_loss).item(),
        tuple(emissions.shape[-1:]) == (15,), tuple(severity_logits.shape[-1:]) == (4,),
    ]
    log(f"\n## ALL CHECKS PASSED: {all(all_checks)}")

    RESULTS = ROOT.parent / "results"
    LOGS = ROOT.parent / "logs"
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    (RESULTS / "phase7_verification.txt").write_text("\n".join(report_lines), encoding="utf-8")

    arch_summary = {
        "entity_types": ENTITY_TYPES,
        "ner_labels": NER_LABELS,
        "severity_labels": SEVERITY_LABELS,
        "dimensions": {
            "fasttext_general": 300,
            "fasttext_medical": 300,
            "concat": 600,
            "projection_out": 256,
            "lstm_hidden_per_direction": 192,
            "lstm_layers": 2,
            "encoder_output": 384,
            "ner_emissions": 15,
            "severity_attention_pool_concat": 768,
            "severity_mlp_hidden": 256,
            "severity_output": 4,
        },
        "parameter_counts": {
            "total": n_params,
            "encoder": n_encoder,
            "ner_head": n_ner,
            "severity_head": n_sev,
        },
        "all_verification_checks_passed": all(all_checks),
    }
    (LOGS / "model_architecture_summary.json").write_text(
        json.dumps(arch_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nWrote {RESULTS / 'phase7_verification.txt'}")
    print(f"Wrote {LOGS / 'model_architecture_summary.json'}")


if __name__ == "__main__":
    main()
