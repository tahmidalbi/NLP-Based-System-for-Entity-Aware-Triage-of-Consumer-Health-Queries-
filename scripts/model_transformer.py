"""
Transformer variant of the BanglaCare model (TRANSFORMER_PLAN.md section 8).

Only the encoder changes: frozen FastText + BiLSTM is replaced by a fine-tuned
pretrained transformer. NERHead and EntityAwareSeverityHead are imported from
model.py unchanged (hidden_dim = 768), and the "protect the NER head" detach
(guide 10.4) is kept.

Everything above the heads works at SUBWORD level; the heads see WORD level -
each word represented by its first subword's state (see dataset_transformer).
"""

import torch
import torch.nn as nn

from labels import ENTITY_TYPES, NER_LABELS
from model import EntityAwareSeverityHead, NERHead

DEFAULT_MODEL = "csebuetnlp/banglabert"


class BanglaCareTransformer(nn.Module):
    def __init__(self, model_name=DEFAULT_MODEL, num_severity=4, dropout=0.1,
                 pretrained=True, encoder=None):
        """
        pretrained=False builds the encoder from its config only (random
        weights) - used by phases 11/12, which load a fine-tuned state dict
        anyway and should not re-download pretrained weights.
        encoder: inject a ready-made module (unit tests use a tiny random one).
        """
        super().__init__()
        if encoder is None:
            from transformers import AutoConfig, AutoModel

            if pretrained:
                encoder = AutoModel.from_pretrained(model_name)
            else:
                encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        self.encoder = encoder
        hidden = self.encoder.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)

        self.ner_head = NERHead(hidden, num_labels=len(NER_LABELS))
        self.severity_head = EntityAwareSeverityHead(
            hidden, num_entity_types=len(ENTITY_TYPES), num_severity=num_severity
        )

        # (15, 7) B/I -> entity-type matrix, identical to BanglaCareModel (guide 10.3).
        bi = torch.zeros(len(NER_LABELS), len(ENTITY_TYPES))
        label2id = {l: i for i, l in enumerate(NER_LABELS)}
        for ei, et in enumerate(ENTITY_TYPES):
            bi[label2id[f"B-{et}"], ei] = 1.0
            bi[label2id[f"I-{et}"], ei] = 1.0
        self.register_buffer("bi_combine_matrix", bi)

    def encode_words(self, input_ids, attention_mask, first_pos):
        """(B, S) subwords -> (B, W, H): the first-subword state of each word."""
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.dropout(h)
        idx = first_pos.unsqueeze(-1).expand(-1, -1, h.size(-1))
        return torch.gather(h, 1, idx)

    def ner_forward(self, input_ids, attention_mask, first_pos):
        word_h = self.encode_words(input_ids, attention_mask, first_pos)
        return self.ner_head.emissions(word_h), word_h

    def severity_forward(self, input_ids, attention_mask, first_pos, word_mask):
        word_h = self.encode_words(input_ids, attention_mask, first_pos)
        emissions = self.ner_head.emissions(word_h)
        q = torch.softmax(emissions, dim=-1) @ self.bi_combine_matrix
        q = q.detach()  # protects the NER head (guide 10.4)
        return self.severity_head(word_h, q, word_mask), word_h


def param_groups(named_groups, weight_decay):
    """AdamW param groups with discriminative LRs.

    named_groups: list of (module, lr). Biases and LayerNorm weights get no
    weight decay (standard transformer fine-tuning practice).
    """
    groups = []
    for module, lr in named_groups:
        decay, no_decay = [], []
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


def linear_warmup_schedule(optimizer, total_steps, warmup_frac=0.10):
    from transformers import get_linear_schedule_with_warmup

    return get_linear_schedule_with_warmup(
        optimizer, int(warmup_frac * total_steps), total_steps
    )
