from typing import Dict

import torch
import torch.nn as nn


class MusicFusionModel(nn.Module):
    """Backbone used by the public MusicFusion training entry point."""

    def __init__(
        self,
        mel_dim: int,
        coch_dim: int,
        hidden_mel: int,
        hidden_coch: int,
        fusion_hidden: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.mel_encoder = nn.Sequential(
            nn.Linear(mel_dim, hidden_mel),
            nn.BatchNorm1d(hidden_mel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_mel, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
        )
        self.coch_encoder = nn.Sequential(
            nn.Linear(coch_dim, hidden_coch),
            nn.BatchNorm1d(hidden_coch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_coch, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
        )

        self.gating = nn.Sequential(
            nn.Linear(fusion_hidden * 2, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.Sigmoid(),
        )

        self.mel_head = nn.Linear(fusion_hidden, num_classes)
        self.coch_head = nn.Linear(fusion_hidden, num_classes)
        self.fusion_head = nn.Linear(fusion_hidden, num_classes)

    def forward(self, mel: torch.Tensor, coch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return features and logits required by training and evaluation."""
        batch_size = mel.size(0)
        mel_flat = mel.view(batch_size, -1)
        coch_flat = coch.view(batch_size, -1)

        mel_feat = self.mel_encoder(mel_flat)
        coch_feat = self.coch_encoder(coch_flat)

        gate = self.gating(torch.cat([mel_feat, coch_feat], dim=-1))
        fusion_feat = gate * mel_feat + (1.0 - gate) * coch_feat

        mel_logits = self.mel_head(mel_feat)
        coch_logits = self.coch_head(coch_feat)
        fusion_logits = self.fusion_head(fusion_feat)

        return {
            "mel_feat": mel_feat,
            "coch_feat": coch_feat,
            "fusion_feat": fusion_feat,
            "mel_logits": mel_logits,
            "coch_logits": coch_logits,
            "fusion_logits": fusion_logits,
        }
