# BanglaCare — NLP-Based System for Entity-Aware Triage of Consumer Health Queries

BanglaCare reads a health message written in Bangla and does two things at once:

1. **Entity extraction (NER)** — finds the medical information in the text: symptoms, health conditions, medicines, age, dosage, specialists and medical procedures.
2. **Severity triage** — classifies how urgent the message is: `Emergency`, `Urgent`, `Routine` or `General Query`, with a calibrated confidence score and a "needs human review" flag.

The severity decision is **entity-aware**: the severity head attends to the words the NER head believes are medical entities, so both tasks share one encoder and help each other.

> **Disclaimer.** BanglaCare prioritizes messages; it does not diagnose or prescribe. It is a research prototype and must not replace professional medical advice. In an emergency, seek immediate medical care.

---

## Two architectures, one interface

The same task is solved with two interchangeable encoders. Both expose the same `predict(text)` output, so the UI never branches on which model is loaded.

| | **BiLSTM + FastText** | **Fine-tuned BanglaBERT** |
|---|---|---|
| Word representation | Frozen General Bengali FastText (300D) + custom Medical FastText (300D) = 600D | `csebuetnlp/banglabert` subword states (768D) |
| Encoder | Projection + 2-layer BiLSTM (384D/token) | 12-layer transformer, fine-tuned |
| Heads | NER: Linear + CRF · Severity: entity-aware attention + max-pool + MLP | identical |
| Parameters | ~2.1 M trainable | ~111 M |
| Runtime memory | ~8 GB (FastText binaries) | ~450 MB |
| Checkpoint | `checkpoints/best_joint.pt` | `checkpoints/best_joint_transformer.pt` |
| Calibration | `checkpoints/calibration.json` | `checkpoints/calibration_transformer.json` |

### Test-set results (each test set evaluated once, with the frozen checkpoint)

| Metric | BiLSTM + FastText | BanglaBERT |
|---|---|---|
| Severity Macro-F1 | 0.910 | **0.922** |
| Severity accuracy | 0.907 | **0.918** |
| Emergency recall | **0.956** | 0.941 |
| Severity ECE (calibrated) | **0.025** | 0.032 |
| NER entity-level F1 (micro) | 0.621 | **0.625** |

Per-entity NER F1 is highest for Specialist (~0.88) and Medicine (~0.85) and lowest for Symptom (~0.47). Full reports, confusion matrices, calibration plots and error analysis are in [`results/`](results/) and [`figures/`](figures/).

---

## Repository layout

```
app/                    Gradio demo (app.py) and inference pipelines (inference.py)
scripts/                Data pipeline, models, training phases, evaluation
  phase1_audit.py … phase5_medical_fasttext.py     data audit, splits, cleaning, corpus, Medical FastText
  build_feature_cache.py                           offline 600D FastText feature cache
  phase8_ner_pretrain.py, phase9_severity_warmup.py, phase10_joint.py     BiLSTM training
  phase_t8_ner.py, phase_t9_warmup.py, phase_t10_joint.py                 BanglaBERT training
  phase11_calibrate.py, phase12_evaluate.py        temperature scaling + final test evaluation
  model.py, model_transformer.py                   architectures
data/raw, data/splits   source datasets and frozen, leakage-safe splits
corpus/                 decontaminated medical corpus for FastText
checkpoints/            trained weights + calibration files (large files are git-ignored)
results/, figures/      metrics, reports, plots
PIPELINE_EXPLAINED.md   step-by-step explanation of training and inference
SETUP.md                collaborator setup notes
TRANSFORMER_PLAN.md     design plan for the BanglaBERT variant
```

For a plain-language walkthrough of everything from data preparation to the final output, read [PIPELINE_EXPLAINED.md](PIPELINE_EXPLAINED.md).

---

## How it works (short version)

**Training (both models)**

1. Audit the raw datasets and build leakage-safe splits (exact and near-duplicate groups never cross splits).
2. Apply conservative text cleaning (Unicode NFC, invisible characters, whitespace, lowercase Latin only). NER text is cleaned per token so labels stay aligned.
3. *BiLSTM only:* build a medical corpus, train Medical FastText, and combine it with the general Bengali FastText.
4. **NER training** on HealthNER with a CRF loss.
5. **Severity warm-up:** train only the new severity head with everything else frozen.
6. **Joint training:** NER and severity batches alternate 1:1, with per-component learning rates. The NER head is protected from severity gradients, and a checkpoint is rejected if NER F1 drops by more than 1 point.
7. **Calibration:** fit a single temperature on the validation set and choose a data-driven human-review threshold.
8. **Final evaluation** on the held-out test sets, once.

**Inference**

`text → clean → words → encoder (FastText+BiLSTM or BanglaBERT) → NER head + CRF → entities; severity head → temperature-scaled softmax → severity, confidence, review flag`

---

## Setup

### Requirements

- **Python 3.11 is recommended.** The `fasttext-wheel` package used by the BiLSTM model has no prebuilt wheel for Python 3.13 on Windows.
- Windows, macOS or Linux. A GPU is optional; inference runs on CPU.
- BiLSTM inference needs roughly **8–9 GB of RAM** for the two FastText models. BanglaBERT inference needs far less.

### Install

```bash
git clone https://github.com/tahmidalbi/NLP-Based-System-for-Entity-Aware-Triage-of-Consumer-Health-Queries-.git
cd NLP-Based-System-for-Entity-Aware-Triage-of-Consumer-Health-Queries-

python -m venv .venv311            # use a Python 3.11 interpreter
.venv311\Scripts\activate          # Windows   (source .venv311/bin/activate on macOS/Linux)

pip install "numpy<2" torch pytorch-crf transformers tokenizers sentencepiece pandas gradio fasttext-wheel
pip install git+https://github.com/csebuetnlp/normalizer   # required by the BanglaBERT model
```

`requirements.txt` lists the pinned versions used for training (including the CUDA build of PyTorch); see [SETUP.md](SETUP.md) for details.

### Required model files

The trained weights and embeddings are large and not stored in git. Place them as follows:

| File | Used by |
|---|---|
| `checkpoints/best_joint.pt`, `checkpoints/calibration.json` | BiLSTM |
| `checkpoints/best_joint_transformer.pt`, `checkpoints/calibration_transformer.json` | BanglaBERT |
| `embeddings/cc.bn.300.bin` (official Bengali FastText, [download](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.bin.gz)) | BiLSTM |
| `embeddings/medical_fasttext.bin` (built by `scripts/phase5_medical_fasttext.py`) | BiLSTM |

The BanglaBERT tokenizer and config are fetched from the Hugging Face Hub on first run (internet required once).

---

## Running the demo

```bash
python app/app.py --arch both
```

Then open <http://127.0.0.1:7860>, type a Bangla health query, and choose a model.

| Option | Meaning |
|---|---|
| `--arch {bilstm,transformer,both}` | Which model the UI starts on (default `transformer`) |
| `--preload` | Load every available model at startup instead of on first use |
| `--cpu` | Force CPU |
| `--port N` | Port (default 7860) |
| `--share` | Create a temporary public Gradio link |
| `--no-normalize` | Skip the csebuetnlp normalizer (**not recommended** — the checkpoint was trained with it) |

The first BiLSTM query is slow because both FastText models are loaded lazily.

### Output

For each query the app shows the extracted entities with their types, the predicted severity, the calibrated confidence, the probability of each severity class, a **"needs human review"** flag when confidence is below the model's validation-derived threshold, and the safety disclaimer.

### Using the pipelines from Python

```python
import sys; sys.path.insert(0, "app")
from inference import PipelineRegistry

registry = PipelineRegistry()
result = registry.predict("transformer", "আমার বুকে ব্যথা হচ্ছে এবং শ্বাস নিতে কষ্ট হচ্ছে")
print(result["severity"], result["confidence"], result["entities"])
```

---

## Reproducing the results

Run from the repository root, in order (see [SETUP.md](SETUP.md) for prerequisites and timing):

```bash
python scripts/phase3_preprocess.py          # clean the frozen splits
python scripts/phase5_medical_fasttext.py    # Medical FastText (BiLSTM only)
python scripts/build_feature_cache.py        # 600D feature cache (BiLSTM only)

# BiLSTM
python scripts/phase8_ner_pretrain.py
python scripts/phase9_severity_warmup.py
python scripts/phase10_joint.py

# BanglaBERT
python scripts/phase_t8_ner.py
python scripts/phase_t9_warmup.py
python scripts/phase_t10_joint.py

python scripts/phase11_calibrate.py          # temperature + review threshold
python scripts/phase12_evaluate.py           # final test evaluation (run once)
```

Training was done on Kaggle GPUs; every training script supports `--smoke --cpu` for a quick local check.

---

## Data

| Dataset | Use |
|---|---|
| Bangla-HealthNER | NER training/evaluation (7 entity types, IOB labels) |
| Bangla Healthcare Severity Dataset | 4-class severity training/evaluation |
| BanglaCHQ-Summ, BanglaHealth | Unlabeled medical text for Medical FastText |

Please respect each dataset's own license and citation requirements (see the files inside `data/raw/`).

---

## Known limitations

- Symptom and Health Condition entities are the hardest to extract (entity F1 ≈ 0.47 and ≈ 0.53).
- The severity data is short and fairly clean; performance on long, noisy, code-switched real-world messages is not measured.
- Confidence is calibrated on the validation split only; the review threshold is a screening aid, not a guarantee.
- The BiLSTM demo needs about 8 GB of RAM for its embeddings; machines with less will page heavily.
- The system is a prioritization aid for research and demonstration, not a clinical tool.
