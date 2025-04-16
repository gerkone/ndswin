from typing import Type, Optional, Sequence, Union, Tuple

from itertools import product
from math import ceil
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn

from ndswin.utils import seq_weight_init, MLP


def pad_to_blocks(
    x: Union[Tuple[int], torch.Tensor], blocks: Sequence[int]
) -> Tuple[torch.Tensor, Sequence[int]]:
    """Pad a tensor or a shape to a block size so that the shape is divisible by blocks.

    Args:
        x (tuple | torch.Tensor): Input shape or tensor.
        blocks (tuple): Block sizes per axis.

    Returns:
        tuple (torch.Tensor, tuple): Padded tensor and padding sequence.
    """
    x_shp = x.shape if isinstance(x, torch.Tensor) else (None, *x)
    pad_axes = []
    for i, w in enumerate(blocks):
        pad_axes.append((w - x_shp[1 + i] % w) % w)  # +1 for batch
        pad_axes.append(0)

    if isinstance(x, torch.Tensor):
        # pad tensor
        # last tuple first in
        pad_axes = pad_axes[::-1]
        if any([p > 0 for p in pad_axes]):
            x = F.pad(x, (0, 0, *pad_axes))
    else:
        # compute padded shape
        x = tuple([x[i] + pad_axes[i * 2] for i in range(len(x))])

    return x, pad_axes


def unpad(
    x: torch.Tensor, pad_axes: Sequence[int], base_grid: Sequence[int]
) -> torch.Tensor:
    """Unpads a tensor to a base resolution, given the padding sequence.

    Args:
        x (torch.Tensor): Input tensor.
        pad_axes (tuple): Padding sequence.
        base_grid (tuple): base grid resolution.

    Returns:
        x (torch.Tensor): Unpadded tensor.
    """
    if any([p > 0 for p in pad_axes]):
        # unpad to original resolution
        x = x[:, *[slice(0, r) for r in base_grid], :].contiguous()

    return x


class PatchEmbed(nn.Module):
    """ViT-style patch embedding for n- dimensional grid data.

    Args:
        space (int): Number of input/output dimensions.
        in_resolution (tuple(int)): Input image size.
        patch_size (int | tuple(int)): Patch size. Default is 5 (across all dimensions).
        embed_dim (int): Latent dimension.
        in_channels (int): Number of input channels. Default is 2.
        norm_layer (nn.Module): Normalization layer type. Default is nn.LayerNorm.
        flatten (bool): Flatten output patches. Default is False.
        use_conv (bool): Use convolutions to patch (only 2D or 3D). Default is False.
        mlp_ratio (float): Expansion rate for patching MLPs. Default is 8.0
        mlp_depth (int): Depth of the patching MLPs. Default is 2
        act_fn (callable): Activation function. Default is nn.LeakyReLU.
    """

    def __init__(
        self,
        space: int,
        in_resolution: Sequence[int],
        patch_size: Sequence[int],
        embed_dim: int,
        in_channels: int = 2,
        norm_layer: nn.Module = None,
        flatten: bool = True,
        use_conv: bool = False,
        act_fn: nn.Module = nn.LeakyReLU,
        mlp_ratio: float = 8.0,
        mlp_depth: int = 2,
        init_weights: Optional[str] = None,
    ):
        assert len(in_resolution) == space, f"Image size must be {space}D"
        assert len(patch_size) == space, f"Patch size must be {space}D"

        super().__init__()
        self.in_resolution = in_resolution
        self.patch_size = patch_size
        self.grid_size = [ceil(in_resolution[i] / patch_size[i]) for i in range(space)]
        self.embed_dim = embed_dim
        self.flatten = flatten
        self.use_conv = use_conv
        self.space = space
        self.in_channels = in_channels

        if use_conv:
            if space == 2:
                Conv = nn.Conv2d
            elif space == 3:
                Conv = nn.Conv3d
            else:
                raise NotImplementedError

            self.patch = Conv(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            )
        else:
            input_lat = [in_channels * np.prod(patch_size)]
            output_lat = [embed_dim]
            hidden_lat = [int(embed_dim * mlp_ratio)] * (mlp_depth - 1)
            self.patch = MLP(
                input_lat + hidden_lat + output_lat,
                act_fn=act_fn,
                bias=False,
            )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        if init_weights:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if init_weights == "torch" or init_weights is None:
            pass
        elif init_weights == "xavier_uniform":
            self.patch.apply(seq_weight_init(nn.init.xavier_uniform_))
        elif init_weights in ["truncnormal", "truncnormal002"]:
            self.patch.apply(seq_weight_init(nn.init.trunc_normal_))
        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Args:
            x: Tensor (B, C, ...)

        Returns:
            Tensor (B, ..., C)
        """
        assert all([xs == res for xs, res in zip(x.shape[1:-1], self.in_resolution)])
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, "b ... c -> b (...) c")
        x = self.norm(x)
        return x

    def proj(self, x):
        if self.use_conv:
            x = rearrange(x, "b ... c -> b c ...")
            x = self.patch(x)
            x = rearrange(x, "b c ... -> b ... c")
        else:
            patch_view = []
            patch_permute_even = []
            patch_permute_odd = []
            for i in range(self.space):
                # + 1 for batch dimension
                patch_view.append(ceil(x.shape[1 + i] / self.patch_size[i]))
                patch_view.append(self.patch_size[i])
                patch_permute_even.append(1 + i * 2)
                patch_permute_odd.append(1 + i * 2 + 1)

            patch_permute = patch_permute_even + patch_permute_odd

            b, c = x.shape[0], x.shape[-1]
            # TODO check if permute is correct or transposes image?
            x = x.view(b, *patch_view, c).permute(0, *patch_permute, -1)
            # flatten patches
            x = x.flatten(1 + self.space)
            # linear flat-patch projection
            x = self.patch(x)

        return x


class PatchMerging(nn.Module):
    """Swin-like patch merging layer for n- dimensional grid data.

    Concatenates odd/even patches across all dimensions and projects them down, reducing
    the grid size by 1/2.

    Args:
        space (int): Number of input/output dimensions.
        dim (int): Latent dimension.
        in_resolution (tuple(int)): Input grid size.
        norm_layer (nn.Module): Normalization layer type. Default is nn.LayerNorm.
        c_multiplier (int): Latent dimensions expansions after merging. Default is 2.
        mask_spaces (tuple(bool)): Whether nth axis is merged. Sequence of length space.
    """

    def __init__(
        self,
        space: int,
        dim: int,
        in_resolution: Sequence[int],
        norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        c_multiplier: int = 2,
        mask_spaces: Optional[Sequence[bool]] = None,
        init_weights: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mask_spaces = mask_spaces if mask_spaces else (True,) * space

        assert len(self.mask_spaces) == space
        n_merges = sum(self.mask_spaces)

        self.in_resolution = in_resolution
        # grid resolution after patch merging
        self.grid_size = [
            ceil(g / 2) if m else g for g, m in zip(in_resolution, self.mask_spaces)
        ]

        self.norm = norm_layer(2**n_merges * dim) if norm_layer else nn.Identity()
        self.out_dim = c_multiplier * dim
        self.reduction = nn.Linear(2**n_merges * dim, self.out_dim, bias=False)

        if init_weights:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if init_weights == "torch" or init_weights is None:
            pass
        elif init_weights == "xavier_uniform":
            self.reduction.apply(seq_weight_init(nn.init.xavier_uniform_))
        elif init_weights in ["truncnormal", "truncnormal002"]:
            self.reduction.apply(seq_weight_init(nn.init.trunc_normal_))
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # must pad to even shape
        padding = [2 if m else 1 for m in self.mask_spaces]
        x, _ = pad_to_blocks(x, padding)

        subspaces = []
        for sub in product(*[[0, 1] if sub else [0] for sub in self.mask_spaces]):
            merge_axis = []
            for i, s in enumerate(sub):
                if self.mask_spaces[i]:
                    # alternated slice
                    merge_axis.append(slice(s, None, 2))
                else:
                    # masked axis
                    merge_axis.append(slice(0, None))
            subspaces.append(merge_axis)

        x = torch.cat([x[:, *merge_ax, :] for merge_ax in subspaces], -1)

        x = self.norm(x)
        x = self.reduction(x)

        return x
