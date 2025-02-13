from typing import Sequence, Union

import torch
from torch import nn


def seq_weight_init(weight_init_fn, bias_init_fn=None):
    if bias_init_fn is None:
        bias_init_fn = nn.init.zeros_

    def _apply(m):
        if isinstance(m, nn.Linear):
            weight_init_fn(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                bias_init_fn(m.bias)

    return _apply


class MLP(nn.Module):
    def __init__(
        self,
        latents: Sequence[int],
        act_fn: nn.Module = nn.GELU,
        bias: Union[bool, Sequence[bool]] = True,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        if isinstance(bias, bool):
            bias = [bias] * (len(latents) - 1)
        dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        mlp = []
        for i, (lat_i, lat_i2) in enumerate(zip(latents, latents[1:])):
            mlp.append(nn.Linear(lat_i, lat_i2, bias=bias[i]))
            mlp.append(dropout)
            if i != len(latents) - 2:
                mlp.append(act_fn())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
