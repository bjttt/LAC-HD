"""Ablation variant 1: NO Conv1D, keep 3-layer MLP.

Architecture:
Frozen LLM hidden states [B, T, L, D]
  -> LayerNorm
  -> cross-layer attention over L layers
  -> 3-layer MLP
  -> token hallucination logits [B, T]

This is the "no context smoothing" ablation against the base head
(Conv1D + 3-layer MLP in confidence_head.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerAttentionConfidenceHeadNoConv(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2560,
        num_layers: int = 36,
        attn_hidden: int = 256,
        mlp_hidden1: int = 1024,
        mlp_hidden2: int = 256,
        dropout: float = 0.1,
        # Accepted for API compatibility with the base head; ignored.
        use_context_conv: bool = False,
        conv_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_context_conv = False

        # Normalize hidden states from different transformer layers.
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Token-wise attention over layers.
        # Input:  [B, T, L, D]
        # Output: [B, T, L, 1]
        self.layer_attn = nn.Sequential(
            nn.Linear(hidden_size, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

        # 3-layer MLP, same as base head.
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
        """Forward.

        Args:
            hidden_states: Tensor with shape [B, T, L, D]. L should be transformer
                layers only. If your HF model returns embeddings + layers, pass
                `outputs.hidden_states[1:]` after stacking.
            attention_mask: Optional Tensor with shape [B, T]. Accepted for API
                compatibility with Conv1D heads; no convolution is used here.

        Returns:
            logits: [B, T], raw BCE logits.
            layer_weights: [B, T, L], attention weights for interpretability.
        """
        if hidden_states.ndim != 4:
            raise ValueError(f"hidden_states must be [B,T,L,D], got {hidden_states.shape}")

        h = self.layer_norm(hidden_states)
        scores = self.layer_attn(h)                 # [B, T, L, 1]
        weights = torch.softmax(scores, dim=2)      # [B, T, L, 1]
        z = (weights * h).sum(dim=2)                # [B, T, D]

        # No Conv1D context smoothing in this ablation.
        logits = self.mlp(z).squeeze(-1)            # [B, T]
        return logits, weights.squeeze(-1)
