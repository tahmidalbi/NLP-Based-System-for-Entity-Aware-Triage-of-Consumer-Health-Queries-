# BanglaCare — Setup for a new collaborator

This repo is pushed to GitHub at `origin` (`tahmidalbi/NLP-Based-System-for-Entity-Aware-Triage-of-Consumer-Health-Queries-`). Almost everything needed is already in the repo — raw datasets, split files, the built medical corpus, and all pipeline scripts are committed. Only a few large/generated files are excluded via `.gitignore` and need to be downloaded or regenerated locally. **Nothing currently requires the owner to manually send a file** — see the table below for exactly why.

## 1. Clone and install

```bash
git clone https://github.com/tahmidalbi/NLP-Based-System-for-Entity-Aware-Triage-of-Consumer-Health-Queries-.git
cd NLP-Based-System-for-Entity-Aware-Triage-of-Consumer-Health-Queries-
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Two things to know about `requirements.txt` before you run it:

- **torch**: pinned to `2.5.1+cu121` via a CUDA 12.1 wheel index. If your machine has no NVIDIA GPU, or a different CUDA version, edit that line — remove `+cu121` and the `--extra-index-url` line, or point it at [the wheel matching your setup](https://pytorch.org/get-started/locally/). Kaggle notebooks (the project's actual target training platform) ship PyTorch preinstalled, so on Kaggle you can usually skip the torch line entirely.
- **fasttext**: we install `fasttext-wheel`, not the official `fasttext` package — the official one fails to compile on Windows against current MSVC/pybind11. `fasttext-wheel` is a maintained fork with the same `import fasttext` API, prebuilt wheels included. Don't `pip install fasttext` on top of it.

## 2. What's in git vs. what you need to get yourself

| Path | In git? | What it is | What you need to do |
|---|---|---|---|
| `data/raw/` | ✅ yes | HealthNER, Severity dataset (csv+xlsx), BanglaCHQ-Summ | Nothing — already there. |
| `data/splits/` | ✅ yes | Phase 2 frozen splits + held-out fingerprints | Nothing — already there. |
| `corpus/medical_corpus.txt` | ✅ yes (65MB) | Phase 4 decontaminated training corpus | Nothing — already there. |
| `results/`, `.gitignore`, `requirements.txt`, all `scripts/*.py` | ✅ yes | Reports + all pipeline code | Nothing — already there. |
| `data/processed/` | ❌ gitignored | Phase 3 cleaned split files | **Regenerate**, don't download: `python scripts/phase3_preprocess.py` (seconds, needs nothing external). |
| `embeddings/cc.bn.300.bin` | ❌ gitignored (5.6GB) | Official pretrained Bengali FastText | **Download** from the exact link below. |
| `embeddings/medical_fasttext.bin` + `.vec` | ❌ gitignored (2.6GB + 198MB) | Our custom-trained Medical FastText | **Regenerate** (recommended, ~15 min) or ask the owner to send the files directly (see §4). |
| `logs/*.json` | ❌ gitignored | Training-config/verification metadata | Auto-created the moment you re-run the script that produces each one (see table in §3). Not required for the pipeline to function — informational only. |
| `checkpoints/` | ❌ gitignored | Not created yet | Nothing yet — Phase 8+ will populate this. |

## 3. Exact rebuild order

Run these from the repo root, in this order, after installing dependencies:

```bash
# 1. Rebuild Phase 3's cleaned files (fast, no downloads needed - data/splits/ is already in git)
python scripts/phase3_preprocess.py

# 2. Download the official pretrained Bengali FastText (see exact link below), then:
#    gunzip it and place the result at embeddings/cc.bn.300.bin

# 3. Rebuild the custom Medical FastText from the corpus already in git (~15 min on a multi-core CPU)
python scripts/phase5_medical_fasttext.py

# 4. (Optional) re-verify the Phase 6 pretrained-model load + Phase 7 data pipeline + model architecture
python scripts/phase7_verify.py
```

You do **not** need to re-run `phase1_audit.py`, `phase2_split.py`, or `phase4_corpus.py` — their outputs (`data/raw/`, `data/splits/`, `corpus/medical_corpus.txt`) are already committed. Re-run them only if you want to rebuild those artifacts from scratch (e.g. `phase4_corpus.py` needs internet access to re-fetch BanglaHealth from Hugging Face — see §5).

## 4. Exact download link

**Official pretrained Bengali FastText (`cc.bn.300.bin`)** — required, ~5.6GB after extraction:

```
https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.bin.gz
```

Download that `.gz`, extract it, and place the resulting `cc.bn.300.bin` at `embeddings/cc.bn.300.bin`. This is a public Facebook AI Research file — no request to the owner needed.

## 5. What else touches the internet (informational, not a manual-download step)

`scripts/phase1_audit.py` and `scripts/phase4_corpus.py` pull **BanglaHealth** on the fly from Hugging Face (`faisal4590aziz/bangla-health-related-paraphrased-dataset`) via the `datasets` library — it's never saved as a file in this repo, it's only used transiently to build `corpus/medical_corpus.txt`, which is already committed. You only need internet access for this if you deliberately re-run `phase4_corpus.py` to rebuild the corpus from scratch; otherwise it's irrelevant.

## 6. What the owner may want to send manually (optional convenience only)

Nothing is *required* — everything above is either already in git, publicly downloadable, or regenerable from files already in git. The one optional convenience:

- **`embeddings/medical_fasttext.bin` + `.vec`** (2.6GB + 198MB): regenerating takes ~15 minutes via `phase5_medical_fasttext.py`. If a collaborator wants to skip that wait, the owner can send these two files directly (e.g. via a shared drive link — they're too large for GitHub/email as-is). Not necessary otherwise.

## 7. Note on file size

`corpus/medical_corpus.txt` is 65MB and is committed directly (under GitHub's 100MB hard limit, though GitHub may show a "large file" warning above 50MB on push — this is just a warning, the push succeeds). If the corpus grows significantly in a future phase, consider Git LFS at that point.
