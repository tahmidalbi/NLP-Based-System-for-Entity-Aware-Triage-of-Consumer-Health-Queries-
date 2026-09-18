# BanglaCare — Next Update: Transformer Variant

**Goal:** replace the frozen-FastText + BiLSTM encoder with a fine-tuned pretrained
transformer, keeping everything else identical, so we can say something defensible
about whether the transformer actually helps on *our* data.

The original guide deliberately excluded transformers from the "one best version"
(§2.3, §10.5). This is not a correction of that decision — it is the controlled
follow-up experiment that makes the decision defensible instead of assumed.

---

## 1. Experimental design: change exactly one thing

```
                    CURRENT (shipped)                    NEW (this plan)
Representation      General FastText 300D                BanglaBERT subword
                    + Medical FastText 300D              embeddings (trainable)
                    = 600D, FROZEN
Encoder             600→256 projection                   12-layer transformer
                    + 2-layer BiLSTM (384D/token)        (768D/token)
─────────────────── ──────────────────────────────────── ────────────────────────
NER head            Linear(→15) + CRF                    ← IDENTICAL
Severity head       entity-aware attention               ← IDENTICAL
                    + max-pool → 768→256→4
Splits              Phase 2 frozen                       ← IDENTICAL
Preprocessing       Phase 3 output                       ← IDENTICAL
Evaluation          ner_eval.py / severity_eval.py       ← IDENTICAL
Calibration         Phase 11 temperature scaling         ← IDENTICAL
```

Everything below the line stays byte-identical. If you change the heads, the
learning-rate schedule *and* the encoder at once, a better number tells you
nothing about why.

`scripts/model.py` already parameterises the head dimensions
(`NERHead(input_dim, ...)`, `EntityAwareSeverityHead(hidden_dim, ...)`), so both
heads can be imported and reused as-is with `hidden_dim=768`. Do that rather than
rewriting them.

---

## 2. The numbers to beat

From the shipped BiLSTM run (seed 42), already in `results/`:

| Metric | Validation | **Test** |
|---|---|---|
| NER entity-level F1 (micro) | 0.6099 | **0.6209** |
| NER precision / recall | — | 0.6208 / 0.6210 |
| NER macro F1 over types | — | 0.6669 |
| Severity Macro-F1 | 0.9182 | **0.9099** |
| Severity accuracy | — | 0.9068 |
| **Emergency Recall** | 0.9118 | **0.9556** |
| ECE (after calibration) | 0.0149 | — |
| Temperature | 1.6358 | — |
| Trainable parameters | 2,088,210 | — |

Per-entity **test** F1 (full table in `results/phase12_ner_test.md`, 8,462 gold
spans over 3,179 sequences):

| Entity type | Test F1 | Gold spans |
|---|---|---|
| Specialist | 0.8852 | 687 |
| Medicine | 0.8458 | 1,852 |
| Age | 0.7829 | 554 |
| Medical Procedure | 0.6336 | 308 |
| Dosage | 0.5510 | 747 |
| Health Condition | 0.5048 | 786 |
| **Symptom** | **0.4649** | **3,528** |

Symptom is both the **worst-performing type and by far the most frequent** — 42%
of all gold spans. It alone drags the micro-F1 down; macro-F1 is 0.6669 precisely
because the rare-but-easy types (Specialist, Medicine) pull the unweighted mean
up. Symptoms are multi-token, fuzzy-boundary phrases where exact-span matching
gives zero credit for a one-word miss, so this is where contextual embeddings
should beat a fixed vector per word.

**If the transformer wins anywhere, it should win here.** Track Symptom F1 as the
headline diagnostic, not just overall F1.

Report parameter count too. A transformer that costs ~55× the parameters for
+2 F1 is a real finding, not a failure.

---

## 3. What must not change (and why)

| Do not touch | Why |
|---|---|
| `data/splits/` | Frozen in Phase 2, duplicate-aware. Reshuffling invalidates every comparison. |
| `data/processed/` | Phase 3 output. Feed the transformer the *same* cleaned tokens. |
| `data/splits/held_out_fingerprints.json` | Val/test texts that must never enter any training corpus. |
| `scripts/ner_eval.py` | Entity-level span F1, 38 passing tests. Reuse verbatim. |
| `scripts/severity_eval.py` | Macro-F1, Emergency Recall, ECE, temperature fitting. |
| `scripts/labels.py` | The 15 NER labels and 4 severity classes, in this exact order. |
| Test sets | Run **once**, at the very end, after every design decision is frozen. |

The test set is the one thing you cannot get back. If you look at test numbers and
then tune anything, the comparison is dead and you have to say so in the report.

---

## 4. Prerequisites

```bash
pip install transformers>=4.40 tokenizers sentencepiece
# BanglaBERT (csebuetnlp) additionally requires their normalizer:
pip install git+https://github.com/csebuetnlp/normalizer
```

`torch` and `pytorch-crf` are already in `requirements.txt`. Add the new lines to
it when you commit, with a comment noting they are for the transformer variant
only.

---

## 5. Step 0 — Subword length audit (do this first, it takes 5 minutes)

The BiLSTM ran on **words**. A transformer runs on **subwords**, and Bangla
morphology expands badly. Phase 3 reported max 507 *words* and zero examples over
the 512 cap — but at subword level a meaningful fraction may blow past 512.

Write `scripts/audit_subword_lengths.py`:

```python
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
tok = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")

for split in ["train", "valid", "test"]:
    data = json.load(open(ROOT / "data/processed/healthner" / f"{split}.json", encoding="utf-8"))
    lens = [len(tok(ex["tokens"], is_split_into_words=True)["input_ids"]) for ex in data]
    lens = np.array(lens)
    print(f"{split}: median={np.median(lens):.0f} p95={np.percentile(lens,95):.0f} "
          f"p99={np.percentile(lens,99):.0f} max={lens.max()} "
          f"over_512={(lens > 512).sum()} ({(lens > 512).mean():.2%})")
```

**Decide from the output, and write the decision down:**

- `over_512` under ~1% → plain truncation at 512 is fine. Log the count.
- `over_512` above ~1% → use a sliding window with overlap (guide §5.5 says the
  same thing), or you are silently deleting gold entities from long queries and
  your recall drop will be an artefact, not a finding.

Severity texts max out at 52 words, so they are never a problem.

---

## 6. Step 1 — Pick the encoder

| Model | HF id | Notes |
|---|---|---|
| **BanglaBERT** (recommended) | `csebuetnlp/banglabert` | ELECTRA discriminator, Bangla-only, strongest published Bangla NLU baseline. Needs their normalizer. |
| BanglishBERT | `csebuetnlp/banglishbert` | Bangla + English. Worth trying *because* our data is code-switched. |
| XLM-RoBERTa base | `xlm-roberta-base` | Multilingual, robust, no special normalizer. Good fallback. |
| MuRIL | `google/muril-base-cased` | Indic-focused, handles transliteration. |

**Start with BanglaBERT.** If you have GPU budget left, BanglishBERT is the most
interesting second run given how much Banglish is in the severity set — that is a
one-line change and a genuinely publishable comparison.

`AutoModel.from_pretrained(...)` works for all four; `.last_hidden_state` is
`(batch, seq, 768)` in every case.

### Normalization decision (read this, it is easy to get wrong)

csebuetnlp models expect their `normalizer` applied to input text. But our tokens
are already cleaned by Phase 3, and we need **token count preserved** or the IOB
labels break.

Apply the normalizer **per token**, never to the joined string:

```python
from normalizer import normalize
clean = [normalize(t) or t for t in ex["tokens"]]   # fall back to original if it empties
assert len(clean) == len(ex["labels"])
```

This mirrors what `text_utils.clean_token()` already does for exactly the same
reason. If you normalize the whole sentence and re-split, alignment dies silently
and your NER F1 will be mysteriously terrible.

---

## 7. Step 2 — Tokenization and label alignment (the hard part)

This is where transformer NER implementations usually go wrong. Read it twice.

HealthNER gives one IOB label **per word**. The tokenizer produces several
subwords per word. We train and evaluate at **word level** so the numbers stay
comparable with the BiLSTM run.

The standard trick is to label the first subword and set the rest to `-100`. That
works for plain cross-entropy — **but we are using a CRF**, and `pytorch-crf`
requires `mask[:, 0] == True` for every sequence. If `[CLS]` sits at position 0
and you mask it out, the CRF raises.

**Solution: gather the first-subword positions into a compact word-level tensor**
instead of masking in place. Cleaner, and it sidesteps the CRF constraint entirely.

```python
from labels import NER_LABEL2ID

def encode_example(tokens, labels, tokenizer, max_len=512):
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
    )
    word_ids = enc.word_ids()

    first_pos, word_labels = [], []
    seen = set()
    for pos, wid in enumerate(word_ids):
        if wid is None or wid in seen:   # special token, or a continuation subword
            continue
        seen.add(wid)
        first_pos.append(pos)
        word_labels.append(NER_LABEL2ID[labels[wid]])

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "first_pos": first_pos,        # (W,) index into the subword sequence
        "word_labels": word_labels,    # (W,) one label per surviving word
    }
```

Truncation is handled for free: `word_ids()` only covers words that survived, so
`word_labels` is automatically the matching prefix.

**Two assertions worth keeping in the code permanently:**

```python
assert len(first_pos) == len(word_labels)
assert len(first_pos) <= len(tokens)      # equal unless truncated
```

The collator pads `input_ids`/`attention_mask` to the batch's subword max, and
`first_pos`/`word_labels` to the batch's **word** max, with a separate
`word_mask`. Two different padding lengths in one batch — do not conflate them.

---

## 8. Step 3 — The model

New file `scripts/model_transformer.py`. Import the existing heads; do not
reimplement them.

```python
import torch
import torch.nn as nn
from transformers import AutoModel

from labels import ENTITY_TYPES, NER_LABELS
from model import EntityAwareSeverityHead, NERHead


class BanglaCareTransformer(nn.Module):
    """Same two heads as BanglaCareModel, transformer encoder instead of
    frozen-FastText + BiLSTM. Operates at WORD level above the encoder so the
    NER decode and the severity attention see the same granularity as the
    BiLSTM version did."""

    def __init__(self, model_name="csebuetnlp/banglabert", num_severity=4, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size          # 768
        self.dropout = nn.Dropout(dropout)

        self.ner_head = NERHead(hidden, num_labels=len(NER_LABELS))
        self.severity_head = EntityAwareSeverityHead(
            hidden, num_entity_types=len(ENTITY_TYPES), num_severity=num_severity
        )

        # Same (15, 7) B/I -> entity-type matrix as BanglaCareModel (guide 10.3).
        bi = torch.zeros(len(NER_LABELS), len(ENTITY_TYPES))
        label2id = {l: i for i, l in enumerate(NER_LABELS)}
        for ei, et in enumerate(ENTITY_TYPES):
            bi[label2id[f"B-{et}"], ei] = 1.0
            bi[label2id[f"I-{et}"], ei] = 1.0
        self.register_buffer("bi_combine_matrix", bi)

    def encode_words(self, input_ids, attention_mask, first_pos):
        """(B, S, H) subword states -> (B, W, H) first-subword-per-word states."""
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.dropout(h)
        idx = first_pos.unsqueeze(-1).expand(-1, -1, h.size(-1))
        return torch.gather(h, 1, idx)                     # (B, W, H)

    def ner_forward(self, input_ids, attention_mask, first_pos):
        word_h = self.encode_words(input_ids, attention_mask, first_pos)
        return self.ner_head.emissions(word_h), word_h

    def severity_forward(self, input_ids, attention_mask, first_pos, word_mask):
        word_h = self.encode_words(input_ids, attention_mask, first_pos)
        emissions = self.ner_head.emissions(word_h)
        q = torch.softmax(emissions, dim=-1) @ self.bi_combine_matrix
        q = q.detach()                                     # protect the NER head (guide 10.4)
        return self.severity_head(word_h, q, word_mask), word_h
```

The `q.detach()` is the same "protect the NER head" rule as the BiLSTM version
(guide §10.4). Keep it. Without it, severity gradients corrupt the entity
extractor and your NER guardrail in the joint phase will start failing.

**Padding note:** pad `first_pos` with `0` (not `-1`) so `torch.gather` never sees
an invalid index. Those positions are masked out by `word_mask` anyway, so the
garbage value is discarded — but an out-of-range index crashes the gather.

---

## 9. Step 4 — Training phases

Mirror Phases 8 → 9 → 10 exactly so the comparison holds. Transformers converge
far faster than the BiLSTM did: **3–5 epochs, not 20.**

### Phase T8 — NER fine-tuning → `checkpoints/best_ner_transformer.pt`

| Setting | Value | vs. BiLSTM |
|---|---|---|
| Encoder LR | **2e-5** | was 1e-3 (a transformer at 1e-3 will diverge) |
| Head LR (NER linear + CRF) | 1e-3 | same |
| Weight decay | 0.01 | was 1e-4 |
| Scheduler | linear, 10% warmup | was none |
| Batch size | 16 | was 32 (768D × 512 tokens needs the headroom) |
| Grad clip | 1.0 | same |
| Max epochs | 5 | was 20 |
| Early stopping | patience 2 on val entity-F1 | was patience 3 |
| Mixed precision | on | same |

Use **discriminative learning rates** — one AdamW param group for
`model.encoder` at 2e-5, another for the heads at 1e-3. `train_utils.py` already
has the helpers (`set_requires_grad`, `trainable_parameters`, `clip_and_step`,
`EarlyStopping`, `save_checkpoint`).

Keep the CRF loss in **fp32** even with AMP on, exactly as
`phase8_ner_pretrain.py` does (`emissions.float()`). The partition function is a
long logsumexp chain and degrades to `nan` in fp16. This bit us already.

### Phase T9 — Severity warm-up → `checkpoints/best_warmup_transformer.pt`

Freeze the transformer entirely, train only the attention + severity MLP.
2–3 epochs, LR 1e-3, plain cross-entropy, **no class weights** (classes are
22–27%, guide §12.2). Select on validation Macro-F1.

Put the frozen encoder in `.eval()` during warm-up training so its dropout is off
— `phase9_severity_warmup.py` does this and the reason is in the comments there.

### Phase T10 — Joint fine-tuning → `checkpoints/best_joint_transformer.pt`

1:1 alternating NER/severity batches, everything unfrozen.

| Param group | LR |
|---|---|
| Transformer encoder | 1e-5 (half the T8 rate) |
| NER head | 5e-4 |
| Severity attention + MLP | 1e-3 |

3–5 epochs. **Keep the NER guardrail**: select on severity Macro-F1, but reject
any checkpoint whose NER validation F1 falls more than 1 point below the T8 best.
`phase10_joint.py` measures that baseline at epoch 0 rather than reading it from a
file — copy that approach, it survives re-running phases out of order.

### Phase T11 / T12 — Calibration and final evaluation

**No new code needed.** Both scripts take `--joint-checkpoint`:

```bash
python scripts/phase11_calibrate.py --joint-checkpoint checkpoints/best_joint_transformer.pt
python scripts/phase12_evaluate.py  --joint-checkpoint checkpoints/best_joint_transformer.pt
```

You will need to make the model construction in those two scripts switchable
(a `--arch {bilstm,transformer}` flag is enough). Everything downstream —
temperature fitting, Youden's-J threshold, ECE, span F1, error analysis — is
architecture-agnostic and should be reused untouched. Write the outputs to
`results/transformer_*` so the BiLSTM results are not overwritten.

---

## 10. Kaggle runtime plan

| Stage | Hardware | Estimate |
|---|---|---|
| Subword audit | CPU | 5 min |
| Phase T8 (NER, 5 epochs) | T4 GPU | 45–90 min |
| Phase T9 (warm-up) | T4 GPU | 5–10 min |
| Phase T10 (joint) | T4 GPU | 45–90 min |
| Calibration + final eval | GPU | 10 min |

Roughly **2–4 hours** per complete run once the code is stable — comparable to the
BiLSTM pipeline, because the transformer needs far fewer epochs even though each
epoch is slower. Weekly Kaggle GPU quota is ~30h, so one architecture plus a
BanglishBERT comparison is affordable.

**No feature cache this time.** The transformer trains its own embeddings, so
`feature_cache.npz` is irrelevant — you need the tokenizer instead, which
downloads in seconds. Internet must be ON for the first run to pull the model from
HuggingFace; save it as a Kaggle Dataset afterwards if you want offline runs.

**Debug on CPU with a 200-example subset before spending GPU quota.** Every
existing phase script has a `--smoke` flag for exactly this; add one here too.
Shape bugs in the gather/padding logic will surface in 30 seconds on CPU.

---

## 11. Results table to fill in

Put this in the report. Empty cells are the deliverable.

| | BiLSTM (shipped) | BanglaBERT | BanglishBERT (optional) |
|---|---|---|---|
| NER entity-F1, micro (test) | 0.6209 | | |
| NER precision / recall | 0.6208 / 0.6210 | | |
| NER macro F1 over types | 0.6669 | | |
| **Symptom F1** (the one to watch) | **0.4649** | | |
| Health Condition F1 | 0.5048 | | |
| Dosage F1 | 0.5510 | | |
| Medical Procedure F1 | 0.6336 | | |
| Age F1 | 0.7829 | | |
| Medicine F1 | 0.8458 | | |
| Specialist F1 | 0.8852 | | |
| Severity Macro-F1 (test) | 0.9099 | | |
| Severity accuracy | 0.9068 | | |
| **Emergency Recall** | **0.9556** | | |
| ECE after calibration | 0.0149 | | |
| Fitted temperature | 1.6358 | | |
| Trainable parameters | 2,088,210 | ~110M | ~110M |
| Training time (full pipeline) | ~30 min | | |

Severity is already at 0.91 Macro-F1 and 0.96 Emergency Recall, so there is not
much headroom there — **the honest place to look for a transformer win is NER**,
and specifically the Symptom / Dosage / Health Condition types. Say so up front in
the report rather than presenting a 0.3-point severity move as a result.

---

## 12. Pitfalls checklist

- [ ] Labels aligned via `word_ids()`, **never** by re-splitting normalized text
- [ ] `first_pos` padded with `0`, not `-1` (gather crashes on negative indices)
- [ ] Two padding lengths per batch: subword-level and word-level, kept separate
- [ ] CRF loss computed in fp32 under AMP (`emissions.float()`)
- [ ] `q.detach()` kept in `severity_forward` — the NER-head protection rule
- [ ] Encoder LR ~2e-5, heads ~1e-3 — a single flat LR will either diverge or crawl
- [ ] Frozen encoder in `.eval()` during the warm-up phase
- [ ] Subword-length audit run and the truncation decision written down
- [ ] Per-token normalization, token count asserted unchanged
- [ ] Results written to `results/transformer_*`, not over the BiLSTM files
- [ ] Test sets touched exactly once, after everything is frozen
- [ ] Same seed (42) as the BiLSTM run; report the seed

---

## 13. What to add to the report

The write-up already has a structure (guide §21). This adds:

- **Method**: one paragraph on the transformer variant, framed as a controlled
  substitution of the encoder, and the subword/word alignment scheme.
- **Results**: the table from §11, with parameter counts and wall-clock beside the
  accuracy numbers. A 55× parameter increase is part of the result.
- **Discussion**: whether the transformer justifies its cost *for this use case*.
  If the gain is small, that is a legitimate finding and directly supports the
  original guide's decision to ship the lightweight model — say that plainly
  rather than burying it.
- **Limitations**: single seed unless you run three; no ablations on the
  transformer side; BanglaBERT's pretraining corpus is not health-domain.

---

## 14. Suggested order of work

1. Step 0 subword audit → decide truncation strategy (**30 min**)
2. `scripts/dataset_transformer.py` — encoding + collator + the two assertions
3. Unit-test the alignment on 20 hand-checked examples before training anything.
   Verify: token count preserved, `first_pos` length equals label count, and
   decoding the gathered positions reproduces the original words. **This is the
   single highest-value hour in the whole plan** — a silent alignment bug produces
   plausible-looking but meaningless F1, and you will not notice for days.
4. `scripts/model_transformer.py` → forward-pass shape test on CPU
5. `scripts/phase_t8_ner.py` with `--smoke`, CPU first, then GPU
6. T9, T10 (copy the structure from `phase9_*` / `phase10_*`)
7. Add `--arch` to phase 11 / 12, run calibration + final eval
8. Fill in the results table, write the discussion
