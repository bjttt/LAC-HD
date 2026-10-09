"""LAC-HD detection head for a frozen Llama-2-7b backbone.

Input: [batch, tokens, 32 transformer layers, 4096 hidden features].
Embedding states are excluded. The learned module is identical to the shared
LAC-HD architecture, so its state dictionary remains compatible with the
existing training, evaluation and structural-control interfaces.
"""
from __future__ import annotations
from typing import Any
import torch
from confidence_head import LayerAttentionConfidenceHead


class Llama7BConfidenceHead(LayerAttentionConfidenceHead):
    HIDDEN_SIZE = 4096
    NUM_LAYERS = 32

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        attn_hidden: int = 256,
        mlp_hidden1: int = 1024,
        mlp_hidden2: int = 256,
        dropout: float = 0.1,
        use_context_conv: bool = True,
        conv_kernel_size: int = 3,
    ) -> None:
        if (num_layers, hidden_size) != (self.NUM_LAYERS, self.HIDDEN_SIZE):
            raise ValueError('Llama-2-7b requires 32 layers and hidden size 4096')
        super().__init__(hidden_size=hidden_size, num_layers=num_layers,
                         attn_hidden=attn_hidden, mlp_hidden1=mlp_hidden1,
                         mlp_hidden2=mlp_hidden2, dropout=dropout,
                         use_context_conv=use_context_conv,
                         conv_kernel_size=conv_kernel_size)

    @staticmethod
    def validate_model_config(config: Any) -> None:
        expected = {'model_type': 'llama', 'hidden_size': 4096,
                    'num_hidden_layers': 32, 'vocab_size': 32000}
        actual = {key: getattr(config, key, None) for key in expected}
        if actual != expected:
            raise ValueError(f'Expected Llama-2-7b architecture {expected}, got {actual}. '
                             'Use the exact Llama-2-7b model/tokenizer snapshot.')

    @classmethod
    def from_model_config(cls, config: Any, **kwargs: Any) -> 'Llama7BConfidenceHead':
        cls.validate_model_config(config)
        return cls(**kwargs)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 4 or hidden_states.shape[2:] != (self.NUM_LAYERS, self.HIDDEN_SIZE):
            raise ValueError(f'Llama-2-7b states must be [B,T,32,4096], got {tuple(hidden_states.shape)}')
        if attention_mask is not None and attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError('attention_mask must match the [B,T] input dimensions')
        return super().forward(hidden_states, attention_mask=attention_mask)
