"""
Architecture switch for phases 11 and 12 (`--arch {bilstm,transformer}`).

Everything downstream of the logits/emissions - temperature fitting, Youden's-J
threshold, ECE, span F1, error analysis - is architecture-agnostic. The only
things that differ are how the model is built, how batches are produced, and
which tensors go into the forward pass; this module owns exactly those three.
"""

from pathlib import Path

from train_utils import load_featurizer, ner_loader as bilstm_ner_loader
from train_utils import severity_loader as bilstm_severity_loader

ARCHS = ("bilstm", "transformer")


def result_prefix(arch):
    """Transformer outputs go to results/transformer_* so BiLSTM files survive."""
    return "" if arch == "bilstm" else "transformer_"


class Arch:
    def __init__(self, arch, device, cache=None, model_name=None, no_normalize=False):
        assert arch in ARCHS, arch
        self.arch = arch
        self.device = device
        self.prefix = result_prefix(arch)

        if arch == "bilstm":
            from model import BanglaCareModel

            self.featurizer = load_featurizer(cache) if cache else load_featurizer()
            self.model = BanglaCareModel().to(device)
        else:
            from dataset_transformer import DEFAULT_MODEL, WordEncoder, load_tokenizer, make_normalizer
            from model_transformer import BanglaCareTransformer

            model_name = model_name or DEFAULT_MODEL
            self.encoder = WordEncoder(load_tokenizer(model_name), make_normalizer(not no_normalize))
            # Weights come from the fine-tuned checkpoint, not the hub.
            self.model = BanglaCareTransformer(model_name, pretrained=False).to(device)

    # ------------------------------------------------------------ loaders
    def ner_loader(self, split, batch_size, shuffle=None, limit=None):
        if self.arch == "bilstm":
            return bilstm_ner_loader(self.featurizer, split, batch_size, shuffle=shuffle, limit=limit)
        from dataset_transformer import ner_loader
        return ner_loader(self.encoder, split, batch_size, shuffle=shuffle, limit=limit)

    def severity_loader(self, split, batch_size, shuffle=None, limit=None):
        if self.arch == "bilstm":
            return bilstm_severity_loader(self.featurizer, split, batch_size, shuffle=shuffle, limit=limit)
        from dataset_transformer import severity_loader
        return severity_loader(self.encoder, split, batch_size, shuffle=shuffle, limit=limit)

    # ------------------------------------------------------------ forward
    def _t(self, batch, key):
        return batch[key].to(self.device, non_blocking=True)

    def ner_forward(self, batch):
        """-> (emissions, mask) where mask is the CRF mask for this batch."""
        if self.arch == "bilstm":
            mask = self._t(batch, "mask")
            emissions, _ = self.model.ner_forward(self._t(batch, "features"), batch["lengths"], mask)
        else:
            mask = self._t(batch, "word_mask")
            emissions, _ = self.model.ner_forward(
                self._t(batch, "input_ids"), self._t(batch, "attention_mask"),
                self._t(batch, "first_pos"),
            )
        return emissions, mask

    def severity_forward(self, batch):
        if self.arch == "bilstm":
            logits, _ = self.model.severity_forward(
                self._t(batch, "features"), batch["lengths"], self._t(batch, "mask")
            )
        else:
            logits, _ = self.model.severity_forward(
                self._t(batch, "input_ids"), self._t(batch, "attention_mask"),
                self._t(batch, "first_pos"), self._t(batch, "word_mask"),
            )
        return logits
