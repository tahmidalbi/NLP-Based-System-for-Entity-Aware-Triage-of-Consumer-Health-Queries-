"""
Final neural architecture (guide section 10):

  [General FastText 300D ; Medical FastText 300D] -> 600D per token
    -> Projection: Linear(600,256) -> LayerNorm -> GELU -> Dropout(0.20)
    -> Shared 2-layer BiLSTM (192 hidden/direction) -> 384D per token
    -> NER head:      Linear(384,15) -> CRF
    -> Severity head:  entity-aware attention(384D) + max-pool(384D) -> 768D
                        -> Dense 256 -> GELU -> Dropout(0.40) -> Dense 4

"Protect the NER head" (10.4): on severity batches, the 7D soft entity
probabilities derived from the NER emissions are used as attention features,
but detached first, so severity loss can never backpropagate into the NER
emission Linear layer or the CRF - only into the shared projection/BiLSTM,
via the severity head's own attention+max-pool path operating on the
(non-detached) encoder states.
"""

import torch
import torch.nn as nn
from torchcrf import CRF

from labels import ENTITY_TYPES, NER_LABELS


class SharedEncoder(nn.Module):
    """600D fixed FastText features -> 256D trainable projection -> 384D/token shared BiLSTM."""

    def __init__(self, input_dim=600, proj_dim=256, lstm_hidden=192, lstm_layers=2,
                 proj_dropout=0.20, lstm_dropout=0.30):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.output_dim = lstm_hidden * 2  # 384

    def forward(self, x, lengths):
        """x: (B, T, 600) fixed FastText features. lengths: (B,) real token counts."""
        h = self.projection(x)  # (B, T, 256)
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bilstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )
        return out  # (B, T, 384)


class NERHead(nn.Module):
    """384D token state -> 15 emissions -> CRF (guide 10.2)."""

    def __init__(self, input_dim, num_labels=len(NER_LABELS)):
        super().__init__()
        self.emission = nn.Linear(input_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def emissions(self, h):
        return self.emission(h)  # (B, T, 15)

    def loss(self, emissions, tags, mask):
        """CRF negative log-likelihood (guide 11.2), mean-reduced over the batch."""
        return -self.crf(emissions, tags, mask=mask, reduction="mean")

    def decode(self, emissions, mask):
        """Viterbi-decode the most likely label-id sequence per example."""
        return self.crf.decode(emissions, mask=mask)


class EntityAwareSeverityHead(nn.Module):
    """
    Entity-aware attention + global max-pool severity head (guide 10.3).

    score_i = v^T tanh(W_h h_i + W_q q_i + b)
    alpha   = softmax(score)
    h_att   = sum_i alpha_i * h_i
    z       = [h_att ; h_max] -> Dense 256 -> GELU -> Dropout(0.40) -> Dense 4
    """

    def __init__(self, hidden_dim, num_entity_types=len(ENTITY_TYPES),
                 num_severity=4, mlp_hidden=256, mlp_dropout=0.40):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_q = nn.Linear(num_entity_types, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, num_severity),
        )

    def forward(self, h, q, mask):
        """
        h: (B, T, hidden_dim) shared encoder states (NOT detached - severity
           gradients are allowed to flow into the shared encoder through here).
        q: (B, T, num_entity_types) soft entity probabilities - the caller is
           responsible for detaching this before calling (see BanglaCareModel).
        mask: (B, T) bool, True at real token positions.
        """
        scores = self.v(torch.tanh(self.W_h(h) + self.W_q(q))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)  # (B, T)
        h_att = torch.einsum("bt,btd->bd", alpha, h)  # (B, hidden_dim)

        h_masked = h.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        h_max, _ = h_masked.max(dim=1)  # (B, hidden_dim)

        z = torch.cat([h_att, h_max], dim=-1)  # (B, 2*hidden_dim)
        return self.mlp(z)  # (B, num_severity)


class BanglaCareModel(nn.Module):
    """Full fixed multi-task architecture: shared encoder + NER head + severity head."""

    def __init__(self, input_dim=600, proj_dim=256, lstm_hidden=192, lstm_layers=2,
                 num_severity=4):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, proj_dim, lstm_hidden, lstm_layers)
        self.ner_head = NERHead(self.encoder.output_dim, num_labels=len(NER_LABELS))
        self.severity_head = EntityAwareSeverityHead(
            self.encoder.output_dim, num_entity_types=len(ENTITY_TYPES), num_severity=num_severity
        )

        # (15, 7) 0/1 matrix combining each entity type's B- and I- probability
        # (guide 10.3 step 13: "For each entity type, add its B-type and
        # I-type probabilities").
        bi_combine = torch.zeros(len(NER_LABELS), len(ENTITY_TYPES))
        label2id = {l: i for i, l in enumerate(NER_LABELS)}
        for ei, etype in enumerate(ENTITY_TYPES):
            bi_combine[label2id[f"B-{etype}"], ei] = 1.0
            bi_combine[label2id[f"I-{etype}"], ei] = 1.0
        self.register_buffer("bi_combine_matrix", bi_combine)

    def encode(self, x, lengths):
        return self.encoder(x, lengths)  # (B, T, 384)

    def ner_forward(self, x, lengths, mask):
        """HealthNER batch: full normal gradient flow into projection/BiLSTM/NER head."""
        h = self.encode(x, lengths)
        emissions = self.ner_head.emissions(h)
        return emissions, h

    def severity_forward(self, x, lengths, mask):
        """
        Severity batch: NER head runs forward (its soft probabilities are used
        as attention features) but its output is detached before reaching the
        severity head, so severity loss cannot update the NER emission/CRF
        parameters (guide 10.4, "Protect the NER head"). The severity head
        still updates the shared projection/BiLSTM through `h`, which is not
        detached.
        """
        h = self.encode(x, lengths)
        emissions = self.ner_head.emissions(h)
        probs15 = torch.softmax(emissions, dim=-1)  # (B, T, 15)
        q = probs15 @ self.bi_combine_matrix  # (B, T, 7)
        q = q.detach()  # <-- protects the NER head
        severity_logits = self.severity_head(h, q, mask)
        return severity_logits, h
