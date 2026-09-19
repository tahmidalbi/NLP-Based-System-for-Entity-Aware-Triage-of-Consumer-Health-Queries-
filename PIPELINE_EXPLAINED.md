# BanglaCare — The Whole Pipeline in Easy Words

BanglaCare reads a Bangla health message (like *"আমার বুকে ব্যথা হচ্ছে এবং শ্বাস নিতে কষ্ট হচ্ছে"*) and does **two jobs at once**:

1. **NER (Named Entity Recognition)** – finds the medical things inside the text: symptoms, medicines, age, dosage, and so on.
2. **Severity triage** – decides how urgent the message is: `Emergency`, `Urgent`, `Routine`, or `General Query`.

There are **two versions of the system**. They share the same idea and the same output. Only the "brain" that reads the text is different:

| | **BiLSTM version** | **BanglaBERT version** |
|---|---|---|
| Word understanding | FastText word vectors (frozen, not changed) | BanglaBERT (a big pretrained language model, fine-tuned) |
| Reader | 2-layer BiLSTM (small, ~2.1 million parameters) | BERT transformer (~111 million parameters) |
| Speed / memory | Small model, but needs ~8 GB of FastText files in RAM | Needs ~450 MB, loads in seconds |
| Test severity Macro-F1 | 0.910 | 0.922 |
| Test NER entity-F1 | 0.621 | 0.625 |

The 7 things NER looks for: **Symptom, Health Condition, Medicine, Age, Dosage, Specialist, Medical Procedure**.
The NER labels use the IOB style (15 labels total): `O` (not an entity), `B-Symptom` (a symptom *begins* here), `I-Symptom` (a symptom *continues*), and so on.

---

# PART 0 — Data preparation (same for both models)

This happens once, before any training. Scripts are in `scripts/`.

**Step 1 – Audit the raw data (Phase 1).** Look at the datasets and count/inspect them. Nothing is changed.
- *Bangla-HealthNER* – sentences with a label for every word (used for NER).
- *Severity dataset* – messages with one of the 4 severity labels.
- *BanglaCHQ-Summ* and *BanglaHealth* – plain medical text (used only to teach word meanings for FastText).

**Step 2 – Split the data safely (Phase 2).**
- HealthNER already has train / valid / test files, so they are copied as they are.
- The Severity dataset is split ~80 / 10 / 10 into train / validation / test. The split puts exact duplicates and *near*-duplicates (very similar messages) into the **same** split, so the test set can never contain a copy of something the model trained on. This prevents "cheating by memorising".
- A list of "fingerprints" of all validation + test texts is saved so they can be kept out of the FastText corpus later.

**Step 3 – Clean the text (Phase 3).** Light, careful cleaning only: fix Unicode form, remove invisible characters, collapse extra spaces, lowercase English letters. Nothing is stemmed, no words removed, numbers and medicine names are not touched. For NER, each word is cleaned **one by one**, so the number of words never changes and every word keeps its label.

**Step 4 – Build a medical text corpus (Phase 4).** Collect lots of medical Bangla text (BanglaHealth, training text of HealthNER, BanglaCHQ-Summ, training text of the Severity set). Remove duplicates and remove anything that matches or nearly matches validation/test text. Result: `corpus/medical_corpus.txt`.

---

# PART 1 — BiLSTM + FastText model

## 1A. Training the BiLSTM model

**Step 1 – Train "Medical FastText" (Phase 5).**
FastText learns a 300-number vector for every word, from the medical corpus (skip-gram, 15 epochs). Words used in similar ways get similar vectors. FastText also breaks words into small pieces (3–6 letter chunks), so it can make a sensible vector even for a **spelling mistake or a word it never saw**. This is important because people misspell words in health messages.

**Step 2 – Combine two FastText models.**
Each word becomes **600 numbers**:

```
word  →  [ General Bengali FastText (300) ; Medical FastText (300) ]  →  600 numbers
```
The general model (`cc.bn.300`, trained by Facebook on huge web text) knows everyday Bangla. The medical one knows health words. Both are **frozen** — they are never changed again.

**Step 3 – Cache the vectors (`build_feature_cache.py`).**
So training doesn't need to load 8 GB of FastText, the 600-number vectors for every word in the training data are computed once and saved to a small `.npz` file.

**Step 4 – The model that will be trained.**
```
600 numbers per word
  → Projection (Linear 600→256, LayerNorm, GELU, Dropout)       shrinks + mixes features
  → 2-layer BiLSTM (192 per direction → 384 per word)           reads the sentence left→right AND right→left,
                                                                so each word "knows" its context
        ├─ NER head:      Linear(384→15) + CRF                  label for every word
        └─ Severity head: entity-aware attention + max-pool     one urgency label for the whole message
                          → Dense 256 → GELU → Dropout → Dense 4
```
- **CRF** is a layer that makes sure label sequences make sense (e.g. `I-Symptom` can't appear straight after `O`).
- **Entity-aware attention** means the severity head pays extra attention to the words that NER thinks are medical entities (a chest-pain word matters more than a greeting).
- The BiLSTM and projection are the **shared encoder**: both jobs use it, so the entity knowledge helps the severity decision.

**Step 5 – Phase 8: NER pretraining.**
Train the shared encoder + NER head + CRF on HealthNER train data. Loss = CRF negative log-likelihood (it rewards getting the *whole* label sequence right). AdamW, lr 1e-3, batch 32, up to 20 epochs; stop early if validation entity-F1 doesn't improve for 3 epochs; keep the best checkpoint (`best_ner.pt`).
*Why first?* The Severity dataset is smaller. Teaching the encoder medical structure from the larger NER data first gives it a good start.

**Step 6 – Phase 9: Severity warm-up.**
Attach the new severity head. **Freeze** everything else (projection, BiLSTM, NER head, CRF) and train only the severity head for 2–3 epochs (lr 1e-3, plain cross-entropy). *Why?* A brand-new head gives big random error signals at the start; if they flowed into the encoder they would damage what Phase 8 learned. The best epoch by validation Macro-F1 is saved (`best_warmup.pt`).

**Step 7 – Phase 10: Joint training.**
Unfreeze everything. Now the two tasks **take turns, 1 : 1**:
- A HealthNER batch → NER loss → updates encoder + NER head.
- A Severity batch → severity loss → updates encoder + severity head. The NER head's output is **detached** here, so the severity loss can never damage the NER layers ("protect the NER head").
- Different learning rates: encoder 2e-4, NER head 5e-4, severity head 1e-3.
- Which checkpoint is kept? The one with the best **validation severity Macro-F1** — **but** it is thrown away if validation NER-F1 dropped by more than 1 point compared with the Phase 8 level. Result: `best_joint.pt`.

**Step 8 – Phase 11: Calibration.**
A model's confidence is often too high or too low. On the **validation** set only, find one number **T (temperature)** so that `softmax(logits / T)` gives honest probabilities. For the BiLSTM T = 1.6358. Then choose a **"needs human review" threshold** from validation data (BiLSTM: 0.9196) — predictions with confidence below it will be flagged. Saved to `calibration.json`.

**Step 9 – Phase 12: Final test (once).**
Run the frozen model on the **test** sets one time and record the results (`results/phase12_*`): severity Macro-F1 0.910, Emergency recall 0.956; NER entity-F1 0.621.

## 1B. Inference with the BiLSTM model (what happens when you press Submit)

Example input: `আমার বাচ্চার জ্বর, প্যারাসিটামol খাওয়ানো যাবে?`

1. **You type text** in the interface (Gradio app, `app/app.py`) and choose the BiLSTM model.
2. **Model is loaded (first time only).** The app loads `best_joint.pt`, `calibration.json`, and both FastText models (~8 GB, takes a while).
3. **Empty check.** If the box is empty, you get "Please enter a health query."
4. **Clean the text.** The same light cleaning used in training (`normalize_text`): fix Unicode, remove invisible characters, collapse spaces, lowercase Latin letters.
5. **Split into words.** By spaces. Cut at 512 words if extremely long.
6. **Turn each word into 600 numbers.** General FastText (300) + Medical FastText (300). Unknown or misspelled words still get a vector because of the FastText letter-chunks.
7. **Projection layer** shrinks 600 → 256 numbers per word.
8. **BiLSTM reads the whole sentence** in both directions → 384 numbers per word, now aware of context.
9. **NER head.** Linear layer gives a score for each of the 15 labels per word; the **CRF (Viterbi decoding)** picks the best consistent label sequence, e.g. `জ্বর → B-Symptom`, `প্যারাসিটামol → B-Medicine`.
10. **Group labels into entities.** `B-` starts an entity, following `I-` labels extend it. Result: a list like `Symptom: জ্বর`, `Medicine: প্যারাসিটামol`.
11. **Severity head.** The word-level entity probabilities (combined per type, 7 numbers) guide the attention over the 384-number word states; attention-average + max-pool are joined (768 numbers) → Dense layers → 4 scores (Emergency / Urgent / Routine / General Query).
12. **Calibrate.** Divide the 4 scores by T = 1.6358, apply softmax → 4 probabilities that add to 1.
13. **Pick the answer.** Highest probability = predicted severity; its probability = **confidence**.
14. **Review flag.** If confidence < 0.9196, mark **"Needs human review"**.
15. **Show output.** Entities highlighted in the text, severity label, confidence, probability for each class, review flag, and the safety disclaimer ("this system prioritizes messages; it does not diagnose or prescribe").

---

# PART 2 — Fine-tuned BanglaBERT model

BanglaBERT (`csebuetnlp/banglabert`) is a transformer already trained by others on a huge amount of Bangla text. It already understands Bangla grammar and meaning much better than a small BiLSTM built from scratch. We only **fine-tune** it (gently adjust it) for our two tasks. No FastText is needed.

## 2A. Training the BanglaBERT model

**Step 1 – Use the same cleaned data** from Part 0 (same splits, same labels, same cleaning).

**Step 2 – Extra normalizer.** Each word also goes through the `csebuetnlp` normalizer (made for BanglaBERT's own training), applied **word by word** so the word count never changes.

**Step 3 – Sub-word tokenizing.** BERT cuts words into smaller pieces ("subwords"), e.g. an unusual word becomes 3 pieces. Special tokens `[CLS]` and `[SEP]` are added. Labels stay at the **word** level: each word is represented by the state of its **first piece**, so NER and severity work exactly like in the BiLSTM version (one label per word).

**Step 4 – The model that will be trained.**
```
subword pieces → BanglaBERT (12 transformer layers) → 768 numbers per piece
   → take the FIRST piece of each word → 768 numbers per word
        ├─ NER head:      Linear(768→15) + CRF               (same as BiLSTM version, just 768 inputs)
        └─ Severity head: entity-aware attention + max-pool  (same as BiLSTM version)
```
The two heads have the exact same design as in the BiLSTM model. Only the reader changed. The "protect the NER head" detach is also kept.

**Step 5 – Phase T8: NER fine-tuning.**
Train BanglaBERT + NER head + CRF on HealthNER train. Because BERT is already smart, it is trained with a **small learning rate** (BERT 2e-5, heads 1e-3), a warm-up-then-decay schedule, batch 16, up to 5 epochs, early stopping after 2 epochs with no validation gain. Best checkpoint → `best_ner_transformer.pt`.

**Step 6 – Phase T9: Severity warm-up.**
Freeze BanglaBERT and the NER head. Train only the new severity head for 2–3 epochs, pick the best by validation Macro-F1 → `best_warmup_transformer.pt`. (Same reason as before: don't let random early errors hurt the encoder.)

**Step 7 – Phase T10: Joint training.**
Unfreeze everything; alternate NER and severity batches 1 : 1. Learning rates: BERT 1e-5 (even gentler), NER head 5e-4, severity head 1e-3. The best checkpoint is chosen on validation severity Macro-F1, and rejected if NER-F1 drops by more than 1 point. Result: `best_joint_transformer.pt` (validation Macro-F1 0.933).

**Step 8 – Phase T11: Calibration.**
Fit temperature on the validation set: T = 1.2142, review threshold 0.9392 → `calibration_transformer.json`.

**Step 9 – Phase 12: Final test (once).**
Severity: Macro-F1 0.922, accuracy 0.918, Emergency recall 0.941. NER: entity-F1 0.625 (best types: Specialist 0.88, Medicine 0.86; hardest: Symptom 0.47).

## 2B. Inference with the BanglaBERT model

Example input: `আমার বুকে ব্যথা হচ্ছে এবং শ্বাস নিতে কষ্ট হচ্ছে`

1. **You type text** in the interface and choose the BanglaBERT model.
2. **Model is loaded (first time only).** Tokenizer + BanglaBERT structure, then the fine-tuned weights from `best_joint_transformer.pt` and the temperature from `calibration_transformer.json`. Needs the `csebuetnlp` normalizer package installed.
3. **Empty check.** No text → "Please enter a health query."
4. **Clean the text** (same light cleaning as training) and **split into words**.
5. **csebuetnlp normalizer** on each word (same as training).
6. **Sub-word tokenizing.** Words → pieces → numeric IDs, with `[CLS]` at the start and `[SEP]` at the end. Remember which piece is the first piece of each word. If the message is too long (over 512 pieces), cut it and keep only the words that fit.
7. **BanglaBERT reads everything at once.** Every piece looks at every other piece (self-attention), giving 768 numbers per piece that reflect the full context.
8. **Pick one vector per word** (its first piece).
9. **NER head + CRF (Viterbi)** → the best label for each word (`বুকে → B-Symptom`, `ব্যথা → I-Symptom`, …).
10. **Group labels into entities.** e.g. `Symptom: বুকে ব্যথা`, `Symptom: শ্বাস নিতে কষ্ট`.
11. **Severity head.** Entity probabilities steer the attention over the word vectors; attention-average + max-pool → Dense layers → 4 scores.
12. **Calibrate.** Divide by T = 1.2142, softmax → 4 probabilities.
13. **Pick the answer.** Highest probability = severity, and its value = confidence.
14. **Review flag.** Confidence < 0.9392 → **"Needs human review"**.
15. **Show output.** Entities, severity, confidence, class probabilities, review flag, disclaimer.

---

# Quick side-by-side of the two flows

**Training**
```
BiLSTM :  clean data → train Medical FastText → 600D vectors (cached) → NER pretrain → severity warm-up
          → joint training → calibrate (T) → test once
BERT   :  clean data → normalizer + subwords → NER fine-tune → severity warm-up
          → joint training → calibrate (T) → test once
```

**Inference**
```
BiLSTM :  text → clean → words → FastText 600D → projection → BiLSTM → NER(CRF) + severity → calibrate → output
BERT   :  text → clean → words → normalizer → subwords → BanglaBERT → first-subword per word → NER(CRF) + severity → calibrate → output
```

**Output (identical shape for both):** the entities found, the severity level, a confidence score, the probability of each severity class, a "needs human review" flag, and a disclaimer.

> **Important:** this is a *prioritisation aid*, not a doctor. It does not diagnose or prescribe. In any real emergency, get medical help immediately.
