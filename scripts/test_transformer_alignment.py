"""
Offline tests for the transformer pipeline: label alignment, collation, and
model shapes. Run with:  python scripts/test_transformer_alignment.py

Needs torch, transformers, pytorch-crf - but NO network and no GPU: the
tokenizer is a tiny WordPiece vocab built in a temp dir and the encoder is a
tiny random ELECTRA. This is the "single highest-value hour" check from
TRANSFORMER_PLAN.md section 14: a silent alignment bug gives plausible-looking
but meaningless F1.
"""

import sys
import tempfile
import traceback
from pathlib import Path

import torch

from dataset_transformer import TransformerCollator, WordEncoder
from labels import NER_ID2LABEL, NER_LABEL2ID

_passed = _failed = 0


def check(name):
    def deco(fn):
        global _passed, _failed
        try:
            fn()
            _passed += 1
            print(f"PASS  {name}")
        except Exception:
            _failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
        return fn
    return deco


VOCAB = (
    ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    + ["fever", "head", "##ache", "take", "para", "##ceta", "##mol", "and", "ok",
       "a", "b", "c", "d", "e", "f", "g", "h"]
)


def make_tokenizer():
    from transformers import BertTokenizerFast

    d = Path(tempfile.mkdtemp())
    (d / "vocab.txt").write_text("\n".join(VOCAB), encoding="utf-8")
    return BertTokenizerFast(str(d / "vocab.txt"), do_lower_case=True)


TOK = make_tokenizer()
ENC = WordEncoder(TOK, max_len=16)

TOKENS = ["take", "paracetamol", "and", "headache"]
LABELS = ["O", "B-Medicine", "O", "B-Symptom"]


@check("multi-subword words: first_pos hits first subword, one label per word")
def _():
    item = ENC.encode(TOKENS, LABELS)
    pieces = TOK.convert_ids_to_tokens(item["input_ids"])
    # [CLS] take para ##ceta ##mol and head ##ache [SEP]
    assert pieces == ["[CLS]", "take", "para", "##ceta", "##mol", "and", "head", "##ache", "[SEP]"], pieces
    assert item["first_pos"] == [1, 2, 5, 6], item["first_pos"]
    assert [pieces[p] for p in item["first_pos"]] == ["take", "para", "and", "head"]
    assert len(item["first_pos"]) == len(item["ner_label_ids"]) == len(TOKENS)
    assert [NER_ID2LABEL[i] for i in item["ner_label_ids"]] == LABELS
    assert item["tokens"] == TOKENS and not item["truncated"]


@check("truncation keeps a matching prefix of words and labels")
def _():
    tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "a", "b", "c", "d", "e", "f", "g", "h", "a", "b"]
    labels = ["O"] * len(tokens)
    labels[0] = "B-Symptom"
    item = ENC.encode(tokens, labels)
    assert len(item["input_ids"]) == 16
    assert item["truncated"] and item["n_words"] == 14  # 16 - [CLS] - [SEP]
    assert len(item["ner_label_ids"]) == len(item["first_pos"]) == len(item["tokens"]) == 14
    assert item["ner_label_ids"][0] == NER_LABEL2ID["B-Symptom"]


@check("word that yields zero subwords is replaced by [UNK], alignment intact")
def _():
    tokens = ["take", "​", "fever"]  # zero-width space tokenizes to nothing
    item = ENC.encode(tokens, ["O", "O", "B-Symptom"])
    assert item["n_words"] == 3
    assert item["ner_label_ids"][-1] == NER_LABEL2ID["B-Symptom"]
    pieces = TOK.convert_ids_to_tokens(item["input_ids"])
    assert pieces[item["first_pos"][1]] == "[UNK]"
    assert pieces[item["first_pos"][2]] == "fever"


@check("per-token normalizer cannot change token count")
def _():
    enc = WordEncoder(TOK, normalize_fn=lambda t: t.upper(), max_len=16)
    item = enc.encode(TOKENS, LABELS)
    assert item["n_words"] == 4


@check("collator: two separate padding lengths, first_pos pad 0, CRF mask[:,0] True")
def _():
    a = ENC.encode(TOKENS, LABELS)                       # 9 subwords, 4 words
    b = ENC.encode(["fever"], ["B-Symptom"])             # 3 subwords, 1 word
    batch = TransformerCollator(TOK.pad_token_id)([a, b])
    assert batch["input_ids"].shape == (2, 9)
    assert batch["first_pos"].shape == (2, 4)
    assert batch["word_mask"][:, 0].all()                # pytorch-crf requirement
    assert batch["word_mask"].tolist() == [[True] * 4, [True, False, False, False]]
    assert batch["first_pos"][1].tolist() == [1, 0, 0, 0]
    assert (batch["attention_mask"][1] == torch.tensor([1, 1, 1] + [0] * 6)).all()
    assert batch["lengths"].tolist() == [4, 1]
    assert batch["ner_tags"][1, 0] == NER_LABEL2ID["B-Symptom"]
    # gathered positions reproduce the original words
    for i, item in enumerate([a, b]):
        pieces = TOK.convert_ids_to_tokens(batch["input_ids"][i].tolist())
        got = [pieces[p] for p in batch["first_pos"][i, : item["n_words"]].tolist()]
        want = [TOK.tokenize(w)[0] for w in item["tokens"]]
        assert got == want, (got, want)


@check("model: shapes, gather, and severity gradients never reach the NER head")
def _():
    from transformers import ElectraConfig, ElectraModel

    from model_transformer import BanglaCareTransformer

    cfg = ElectraConfig(vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64,
                        embedding_size=32, max_position_embeddings=32)
    model = BanglaCareTransformer(encoder=ElectraModel(cfg))
    a = ENC.encode(TOKENS, LABELS)
    b = ENC.encode(["fever"], ["B-Symptom"])
    batch = TransformerCollator(TOK.pad_token_id)([a, b])

    emissions, word_h = model.ner_forward(
        batch["input_ids"], batch["attention_mask"], batch["first_pos"]
    )
    assert emissions.shape == (2, 4, 15) and word_h.shape == (2, 4, 32)
    loss = model.ner_head.loss(emissions.float(), batch["ner_tags"], batch["word_mask"])
    assert torch.isfinite(loss)
    preds = model.ner_head.decode(emissions, batch["word_mask"])
    assert [len(p) for p in preds] == [4, 1]

    model.zero_grad()
    logits, _ = model.severity_forward(
        batch["input_ids"], batch["attention_mask"], batch["first_pos"], batch["word_mask"]
    )
    assert logits.shape == (2, 4)
    logits.sum().backward()
    assert model.ner_head.emission.weight.grad is None or \
        model.ner_head.emission.weight.grad.abs().sum() == 0, "severity leaked into NER emission"
    assert model.ner_head.crf.transitions.grad is None
    assert any(p.grad is not None for p in model.encoder.parameters())


if __name__ == "__main__":
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
