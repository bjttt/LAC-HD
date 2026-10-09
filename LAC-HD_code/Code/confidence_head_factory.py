"""Build shared LAC-HD and structural-control heads for supported backbones."""
from __future__ import annotations
import argparse
from typing import Any
import torch.nn as nn
from confidence_head import LayerAttentionConfidenceHead
from llama7b_confidence_head import Llama7BConfidenceHead
from confidence_head_1 import LayerAttentionConfidenceHeadNoConv
from confidence_head_2 import LayerAttentionConfidenceHeadDeepMLP
from confidence_head_baselines import LastLayerLinearProbe, MeanLayerMLPProbe


def build_head(args_or_cfg: argparse.Namespace | dict[str, Any], hidden_size: int, num_layers: int) -> nn.Module:
    """Build base or ablation confidence head from CLI args/checkpoint config."""
    get = args_or_cfg.get if isinstance(args_or_cfg, dict) else lambda k, d=None: getattr(args_or_cfg, k, d)
    head_type = get("head_type", "base")

    if head_type == "no_conv":
        return LayerAttentionConfidenceHeadNoConv(
            hidden_size=hidden_size,
            num_layers=num_layers,
            attn_hidden=get("attn_hidden", 256),
            mlp_hidden1=get("mlp_hidden1", 1024),
            mlp_hidden2=get("mlp_hidden2", 256),
            dropout=get("dropout", 0.1),
        )

    if head_type == "deep_mlp":
        return LayerAttentionConfidenceHeadDeepMLP(
            hidden_size=hidden_size,
            num_layers=num_layers,
            attn_hidden=get("attn_hidden", 256),
            mlp_hidden1=get("mlp_hidden1", 2048),
            mlp_hidden2=get("mlp_hidden2", 1024),
            mlp_hidden3=get("mlp_hidden3", 512),
            mlp_hidden4=get("mlp_hidden4", 128),
            dropout=get("dropout", 0.1),
            use_context_conv=not get("no_context_conv", False),
            conv_kernel_size=get("conv_kernel_size", 3),
        )

    if head_type == "last_layer_linear":
        return LastLayerLinearProbe(
            hidden_size=hidden_size,
            num_layers=num_layers,
            # Keep this baseline a true linear probe; --dropout is ignored here.
            dropout=0.0,
        )

    if head_type == "mean_layer_mlp":
        return MeanLayerMLPProbe(
            hidden_size=hidden_size,
            num_layers=num_layers,
            mlp_hidden1=get("mlp_hidden1", 1024),
            mlp_hidden2=get("mlp_hidden2", 256),
            dropout=get("dropout", 0.1),
        )

    head_class = (Llama7BConfidenceHead
                  if get("model_type", "") == "llama" and (num_layers, hidden_size) == (32, 4096)
                  else LayerAttentionConfidenceHead)
    return head_class(
        hidden_size=hidden_size,
        num_layers=num_layers,
        attn_hidden=get("attn_hidden", 256),
        mlp_hidden1=get("mlp_hidden1", 1024),
        mlp_hidden2=get("mlp_hidden2", 256),
        dropout=get("dropout", 0.1),
        use_context_conv=not get("no_context_conv", False),
        conv_kernel_size=get("conv_kernel_size", 3),
    )


