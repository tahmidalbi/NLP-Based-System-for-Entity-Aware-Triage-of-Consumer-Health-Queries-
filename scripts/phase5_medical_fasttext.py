"""
Phase 5 - Train the custom Medical FastText.

Trains a 300D skip-gram FastText model on corpus/medical_corpus.txt (Phase 4
output) with the exact fixed configuration from the guide (7.1). FastText is
not the final classifier - it only produces the domain-specific 300D word
vectors later concatenated with the general Bengali FastText (Phase 6/7).

Fixed configuration (guide 7.1, not tunable):
  model=skipgram, dim=300, ws=5, epoch=15, minCount=2,
  minn=3, maxn=6, neg=10, lr=0.05
"""

import json
import time
from pathlib import Path

import fasttext

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "corpus" / "medical_corpus.txt"
EMBEDDINGS = ROOT / "embeddings"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
DATA_PROCESSED = ROOT / "data" / "processed"

EMBEDDINGS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "model": "skipgram",
    "dim": 300,
    "ws": 5,          # context window
    "epoch": 15,
    "minCount": 2,
    "minn": 3,
    "maxn": 6,
    "neg": 10,
    "lr": 0.05,
}

# Nearest-neighbor audit set, pulled from real single-token entity mentions in
# HealthNER TRAIN (data/processed/healthner/train.json) plus a couple of
# deliberately Banglish/rare picks, per guide 7.3: "common symptoms, medicine
# names, procedure terms, Banglish spellings, and rare medical words."
AUDIT_WORDS = {
    "Symptom": ["জ্বর", "কাশি", "ব্যথা", "jor"],  # jor = Banglish spelling of জ্বর (fever)
    "Medicine": ["নাপা", "প্যারাসিটামল", "napa", "এন্টিবায়োটিক"],
    "Health Condition": ["ডায়াবেটিস", "diabetes", "pregnancy"],
    "Specialist": ["gynaecologist", "dermatologist", "ডেন্টিস্ট"],
    "Medical Procedure": ["এক্সরে", "অপারেশন", "ecg"],
    "Rare/long medical word": ["আল্ট্রাসনোগ্রাম"],  # ultrasonogram - tests subword generalization
}


def extract_vocab_stats(model):
    words = model.get_words()
    return len(words)


def write_vec_export(model, path):
    """Optional .vec text export (word2vec format): first line 'n_words dim',
    then one 'word v1 v2 ... vN' line per word."""
    words = model.get_words()
    dim = model.get_dimension()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(words)} {dim}\n")
        for w in words:
            vec = model.get_word_vector(w)
            f.write(w + " " + " ".join(f"{x:.6f}" for x in vec) + "\n")


def main():
    line_count = sum(1 for _ in open(CORPUS_PATH, encoding="utf-8"))
    print(f"Training corpus: {CORPUS_PATH} ({line_count} lines)")
    print(f"Config: {CONFIG}")

    t0 = time.time()
    model = fasttext.train_unsupervised(
        input=str(CORPUS_PATH),
        model=CONFIG["model"],
        dim=CONFIG["dim"],
        ws=CONFIG["ws"],
        epoch=CONFIG["epoch"],
        minCount=CONFIG["minCount"],
        minn=CONFIG["minn"],
        maxn=CONFIG["maxn"],
        neg=CONFIG["neg"],
        lr=CONFIG["lr"],
        thread=12,
        verbose=2,
    )
    train_seconds = time.time() - t0
    print(f"Training finished in {train_seconds:.1f}s")

    bin_path = EMBEDDINGS / "medical_fasttext.bin"
    model.save_model(str(bin_path))
    print(f"Saved {bin_path}")

    vec_path = EMBEDDINGS / "medical_fasttext.vec"
    write_vec_export(model, vec_path)
    print(f"Saved {vec_path}")

    vocab_size = extract_vocab_stats(model)

    config_log = dict(CONFIG)
    config_log.update({
        "corpus_path": str(CORPUS_PATH.relative_to(ROOT)),
        "corpus_line_count": line_count,
        "learned_vocab_size": vocab_size,
        "training_seconds": round(train_seconds, 1),
        "output_bin": str(bin_path.relative_to(ROOT)),
        "output_vec": str(vec_path.relative_to(ROOT)),
    })
    (LOGS / "medical_fasttext_config.json").write_text(
        json.dumps(config_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {LOGS / 'medical_fasttext_config.json'}")

    # Qualitative nearest-neighbor audit (guide 7.3)
    lines = ["BanglaCare - Medical FastText nearest-neighbor audit", "=" * 60, ""]
    lines.append(f"Vocabulary size: {vocab_size}")
    lines.append(f"Training time: {train_seconds:.1f}s on corpus of {line_count} lines")
    lines.append("")
    lines.append("NOTE: not every neighbor will be medically perfect - FastText learns")
    lines.append("usage similarity from the corpus, not an ontology (guide 7.3).")
    lines.append("")

    for category, words in AUDIT_WORDS.items():
        lines.append(f"## {category}")
        for w in words:
            in_vocab = w in model.get_words()
            neighbors = model.get_nearest_neighbors(w, k=10)
            lines.append(f"  '{w}' (in learned vocab: {in_vocab})")
            for score, neighbor in neighbors:
                lines.append(f"      {score:.3f}  {neighbor}")
            lines.append("")
        lines.append("")

    audit_path = RESULTS / "embedding_neighbor_audit.txt"
    audit_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()
