"""
based on: "Liu et al., 
Swin Transformer: Hierarchical Vision Transformer using Shifted Windows 
<https://arxiv.org/abs/2103.14030>"
"""

from typing import Type, Sequence, Union, Optional

import numpy as np
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from math import ceil
from functools import partial

from ndswin.layers.drop import DropPath
from ndswin.layers.patching import unpad, pad_to_blocks
from ndswin.utils import seq_weight_init, MLP


def window_partition(x, window_size):
    """Window partition operation in N dimensions.

    Partition tokens into their respective windows

     Args:
        x: input tensor (B, H, W, ..., C)

        window_size: local window size.


    Returns:
        windows: (B * num_windows, window_size ** space, C)
    """

    win_split = []
    win_permute_even = []
    win_permute_odd = []
    for i, w in enumerate(window_size):
        # number of windows in ith dimension
        win_split.append(x.shape[1 + i] // w)  # + 1 for batch dimension
        # window size in ith dimension
        win_split.append(w)
        win_permute_even.append(1 + i * 2)
        win_permute_odd.append(1 + i * 2 + 1)

    win_permute = win_permute_even + win_permute_odd

    b, c = x.shape[0], x.shape[-1]
    windows = x.view(b, *win_split, c)
    windows = windows.permute(0, *win_permute, -1)
    windows = windows.flatten(0, len(window_size))  # flatten batch and num windows
    windows = windows.flatten(1, -2)  # flatten windows
    return windows  # (b * num windows, prod(window_size), c)


def window_reverse(windows, window_size, dims):
    """Window reconstruction/reverse operation in N dimensions.

     Args:
        windows: windows tensor (B * num_windows, window_size[0], ..., C)
        window_size: local window size.
        dims: dimension (B, H, W, ...).

    Returns:
        x: (B, H, W, ..., C)
    """

    space = len(window_size)
    win_split_count = []
    win_split_win = []
    win_permute = []
    for i, (w, d) in enumerate(zip(window_size, dims[1:])):
        # number of windows in ith dimension
        win_split_count.append(d // w)
        # window size in ith dimension
        win_split_win.append(w)
        win_permute.append(1 + i)
        win_permute.append(1 + i + space)

    b = dims[0]
    x = windows.view(b, *win_split_count, *win_split_win, -1)
    x = x.permute(0, *win_permute, -1)
    # recompose by flattening window count and size in pairs
    for i in range(space):
        x = x.flatten(i + 1, i + 2)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Window size correction.

    Args:
       x_size: input size.
       window_size: local window size.
       shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def compute_mask(dims, window_size, shift_size):
    """Computing region masks for n- dimension shifted windows.

    Args:
       dims: dimension values.
       window_size: local window size.
       shift_size: shift size.

    Returns:
        attn_mask: (B * num_windows, window_size ** space, C), window-wise attention
                    masks to account for shifted rolled masks outside the frame.
    """

    img_mask = torch.zeros((1, *dims, 1))
    cnt = 0

    axis_slices = [
        (slice(-s), slice(-w, -s), slice(-s, None))
        for w, s in zip(window_size, shift_size)
    ]

    for ax_idx in product(*axis_slices):
        img_mask[:, *ax_idx, :] = cnt
        cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias.

    Args:
        dim (int): Number of latent channels.
        num_heads (int): Number of attention heads.
        window_size (int | tuple(int)): Local window size.
        qkv_bias (bool): Add a learnable bias to query, key, value. Default is False.
        attn_drop (float): Attention dropout rate. Default is 0.
        proj_drop (float): Dropout rate of output. Default is 0.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_weights: Optional[str] = None,
    ) -> None:

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.qkv_bias = qkv_bias
        space = len(window_size)

        # relative position bias (RPB) to learn token distances
        self.cpb_mlp = MLP(
            [space, 512, num_heads],
            act_fn=partial(nn.ReLU, inplace=True),
            bias=[True, False],
        )
        # get relative_coords_table
        coords_nd = []
        for w in window_size:
            coords_nd.append(torch.arange(-(w - 1), w, dtype=torch.float32))

        rpb = torch.stack(torch.meshgrid(*coords_nd, indexing="ij"))
        for i in range(space):
            rpb[i] = rpb[i] / (window_size[i] - 1)
        rpb = rearrange(rpb, "d ... -> ... d").unsqueeze(0)
        # normalize to -8, 8
        rpb = 8 * rpb
        rpb = torch.sign(rpb) * torch.log2(torch.abs(rpb) + 1.0) / np.log2(8)
        self.register_buffer("rpb", rpb)  # NOTE: fsdp does not shard buffer

        # index with distances
        grid = torch.stack(
            torch.meshgrid(*[torch.arange(w) for w in window_size], indexing="ij")
        )  # (space, wD, wH, wW, wU, wV)
        dists = grid.flatten(1).unsqueeze(-1) - grid.flatten(1).unsqueeze(1)

        # TODO why?
        for i in range(space):
            center = max(np.prod([(2 * w - 1) for w in window_size[(i + 1) :]]), 1)
            dists[i] = (dists[i] + window_size[i] - 1) * center

        self.register_buffer("rpb_idx", dists.sum(0))

        # for swinv2 cosine similarity attention
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.register_buffer("max_logits", torch.log(torch.tensor(1.0 / 0.01)))
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if init_weights:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if init_weights == "torch" or init_weights is None:
            return
        elif init_weights == "xavier_uniform":
            init_weights_fn = nn.init.xavier_uniform_
        elif init_weights in ["truncnormal", "truncnormal002"]:
            init_weights_fn = nn.init.trunc_normal_
        else:
            raise NotImplementedError

        self.cpb_mlp.apply(seq_weight_init(init_weights_fn))
        init_weights_fn(self.qkv.weight)
        if self.qkv_bias:
            nn.init.zeros_(self.qkv.bias)
        init_weights_fn(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, mask):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        qkv = rearrange(
            self.qkv(x),
            "b n (three heads c) -> three b heads n c",
            three=3,
            heads=self.num_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        sl = q.shape[2]

        # compute relative position bias
        rpb = self.cpb_mlp(self.rpb).view(-1, self.num_heads)
        rpb = rpb[self.rpb_idx.flatten()].view(sl, sl, self.num_heads)
        rpb = 16 * torch.sigmoid(rpb)

        rpb = rearrange(rpb, "slx sly h -> h slx sly").unsqueeze(0).contiguous()

        # swinv2 cosine similarity attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=self.max_logits).exp()
        attn = attn * logit_scale
        attn = attn + rpb
        if mask is not None:
            mask = mask.unsqueeze(1)  # head dimension
            mask = mask.repeat(q.shape[0] // mask.shape[0], 1, 1, 1)  # batch dimension
            attn = attn + mask
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = attn @ v

        # attention readout
        x = rearrange(x, "b k n c -> b n (k c)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block.

    Handles shifts and window partitions, applies window-wise attention with the
    configured roll attention mask, reverses the windows and undoes the roll and shift,
    and applies the output MLP.


    Args:
        dim (int): Number of hidden channels.
        num_heads (int): Number of attention heads.
        window_size (tuple(int)): Local window size.
        shift_size (tuple(int)): Window shift size (for each dimension).
        mlp_ratio (float): Expansion ratio of the mlp hidden dimension. Default is 2.
        qkv_bias (bool): Add a learnable bias to query, key, value. Default is False.
        drop (float): Attention output dropout rate. Detault is 0.
        attn_drop (float): Attention dropout rate. Default is 0.
        drop_path (float): Stochastic depth drop rate. Default is 0.
        norm_layer (nn.Module): Normalization layer type. Default is nn.LayerNorm.
        use_checkpoint (bool): Gradient checkpointing (saves memory). Default is False.
        act_fn (callable): Activation function. Default is nn.GELU.
    """

    def __init__(
        self,
        space: int,
        dim: int,
        num_heads: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        mlp_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
        act_fn: nn.Module = nn.GELU,
        init_weights: Optional[str] = None,
    ) -> None:

        super().__init__()
        self.space = space
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.window_size, self.shift_size = get_window_size(
            grid_size, window_size, shift_size
        )

        assert len(self.window_size) == len(self.shift_size) == space

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=mlp_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = MLP([dim, mlp_hidden_dim, dim], act_fn=act_fn, dropout_prob=mlp_drop)

        if init_weights:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if init_weights == "torch" or init_weights is None:
            pass
        elif init_weights == "xavier_uniform":
            self.mlp.apply(seq_weight_init(nn.init.xavier_uniform_))
        elif init_weights in ["truncnormal", "truncnormal002"]:
            self.mlp.apply(seq_weight_init(nn.init.trunc_normal_))
        else:
            raise NotImplementedError

        self.attn.reset_parameters(init_weights)

    def forward_part1(self, x, mask_matrix):
        grid_size = x.shape[1:-1]

        x, pad_axes = pad_to_blocks(x, self.window_size)

        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(
                x,
                shifts=[-s for s in self.shift_size],
                dims=list(range(1, self.space + 1)),
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        # shifted window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)
        # reshape to window/grid
        attn_windows = attn_windows.view(-1, *(self.window_size + (x.shape[-1],)))
        shifted_x = window_reverse(attn_windows, self.window_size, x.shape[:-1])
        if any(i > 0 for i in self.shift_size):
            x = torch.roll(
                shifted_x,
                shifts=[s for s in self.shift_size],
                dims=list(range(1, self.space + 1)),
            )
        else:
            x = shifted_x

        x = unpad(x, pad_axes, grid_size)

        x = self.norm1(x)
        return x

    def forward_part2(self, x):
        x = self.drop_path(self.norm2(self.mlp(x)))
        return x

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(
                self.forward_part1, x, mask_matrix, use_reentrant=False
            )
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        return x


class SwinLayer(nn.Module):
    """
    Basic Swin Transformer layer.

    Pre-computes attention mask, applies swin attention and applies an optional down/up-
    sample layer.

    Args:
        space (int): Number of input/output dimensions.
        dim (int): Number of hidden channels.
        depth (int): Number of swin transformer layers.
        num_heads (int): Number of attention heads.
        grid_size (tuple(int)): Input resolution.
        window_size (tuple(int)): Local window size.
        mlp_ratio (float): Expansion ratio of the mlp hidden dimension. Default is 2.
        qkv_bias (bool): Add a learnable bias to query, key, value. Default is False.
        drop_path (float | tuple(float)): Stochastic depth drop rate. Default is 0.
        mlp_drop (float): Attention output dropout rate. Detault is 0.
        attn_drop (float): Attention dropout rate. Default is 0.
        use_checkpoint (bool): Gradient checkpointing (saves memory). Default is False.
        act_fn (callable): Activation function. Default is nn.GELU.
    """

    def __init__(
        self,
        space: int,
        dim: int,
        depth: int,
        num_heads: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_path: Union[Sequence[float], float] = 0.0,
        mlp_drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
        act_fn: nn.Module = nn.GELU,
        init_weights: Optional[str] = None,
    ) -> None:

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = (0,) * len(window_size)

        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth

        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.grid_size = grid_size

        assert dim % num_heads == 0

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    space,
                    dim=dim,
                    num_heads=num_heads,
                    grid_size=grid_size,
                    window_size=window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    mlp_drop=mlp_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    act_fn=act_fn,
                )
                for i in range(depth)
            ]
        )

        # precompute attention mask
        if any([s > 0 for s in self.shift_size]):
            window_size, shift_size = get_window_size(
                grid_size, self.window_size, self.shift_size
            )
            mask_dims = [ceil(grid_size[i] / w) * w for i, w in enumerate(window_size)]
            attn_mask = compute_mask(mask_dims, window_size, shift_size)
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

        if init_weights:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        for blk in self.blocks:
            blk.reset_parameters(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, self.attn_mask)

        return x
