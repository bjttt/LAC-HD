"""Simple probe baselines for token-level hallucination detection.

These heads share the same forward API as the main confidence heads:

    logits, layer_weights = head(hidden_states, attention_mask)

where hidden_states is [B, T, L, D].  They are intentionally simple so they can
serve as strong, easy-to-interpret baselines in ablation tables.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LastLayerLinearProbe(nn.Module):
    """Use only the last transformer layer and train a linear token classifier.

    Architecture:
        hidden_states[:, :, -1, :] -> LayerNorm -> Linear(D, 1)
    """

    def __init__(
        self,
        hidden_size: int = 2560,
        num_layers: int = 36,
        dropout: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 4:
            raise ValueError(f"hidden_states must be [B,T,L,D], got {hidden_states.shape}")

        z = self.layer_norm(hidden_states[:, :, -1, :])
        if attention_mask is not None:
            mask = attention_mask.to(device=z.device, dtype=z.dtype).unsqueeze(-1)
            z = z * mask
        logits = self.classifier(self.dropout(z)).squeeze(-1)

        # Return deterministic "layer attention" for API compatibility:
        # all mass is on the last layer.
        weights = torch.zeros(
            (*hidden_states.shape[:3],),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        weights[:, :, -1] = 1.0
        return logits, weights


class MeanLayerMLPProbe(nn.Module):
    """Average all transformer layers and train a 3-layer MLP classifier.

    Architecture:
        mean(LayerNorm(hidden_states), dim=layer) -> MLP(D -> H1 -> H2 -> 1)
    """

    def __init__(
        self,
        hidden_size: int = 2560,
        num_layers: int = 36,
        mlp_hidden1: int = 1024,
        mlp_hidden2: int = 256,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden1, mlp_hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden2, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 4:
            raise ValueError(f"hidden_states must be [B,T,L,D], got {hidden_states.shape}")

        h = self.layer_norm(hidden_states)
        z = h.mean(dim=2)
        if attention_mask is not None:
            mask = attention_mask.to(device=z.device, dtype=z.dtype).unsqueeze(-1)
            z = z * mask
        logits = self.mlp(z).squeeze(-1)

        # Uniform layer weights for API compatibility / interpretability.
        weights = torch.full(
            (*hidden_states.shape[:3],),
            fill_value=1.0 / hidden_states.shape[2],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return logits, weights
