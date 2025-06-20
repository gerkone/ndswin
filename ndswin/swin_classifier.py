from typing import Sequence, Optional
from einops import rearrange
from torch import nn

from ndswin.layers import SwinLayer, PatchEmbed, PatchMerging, PositionalEmbedding


class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


class SwinClassifier(nn.Module):
    """Simple N-dimensional shifted window transformer with classifier head.

    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    (arxiv.org/pdf/2103.14030)
    """

    def __init__(
        self,
        dim: int,
        resolution: Sequence[int],
        space: int = 2,
        in_channels: int = 3,
        num_classes: int = 3,
        patch_size: Sequence[int] = (4, 4),
        window_size: Sequence[int] = (7, 7),
        depth: Sequence[int] = [2, 2, 2],
        num_heads: Sequence[int] = [3, 6, 12],
        drop_path: float = 0.1,
        head_drop_p: float = 0.3,
        use_conv: bool = False,
        use_abs_pe: bool = False,
        merge_mask: Optional[bool] = None,
    ):
        super().__init__()

        self.num_layers = len(depth)
        assert len(depth) == len(num_heads)

        self.grid_sizes = []

        self.patch_embed = PatchEmbed(
            space,
            in_resolution=resolution,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
            flatten=False,
            use_conv=use_conv,
            mlp_depth=1,
            norm_layer=nn.LayerNorm,
        )
        grid_size = self.patch_embed.grid_size
        self.grid_sizes.append(grid_size)

        if use_abs_pe:
            self.abs_pe = PositionalEmbedding(dim, grid_size, True, init_weights="rand")

        stack = []
        for d, h in zip(depth, num_heads):
            blk = SwinLayer(
                space,
                dim,
                depth=d,
                num_heads=h,
                grid_size=grid_size,
                window_size=window_size,
                drop_path=drop_path,
                init_weights="xavier_uniform",
            )
            stack.append(blk)
            merge = PatchMerging(space, dim, blk.grid_size, mask_spaces=merge_mask)
            stack.append(merge)
            # next latent size and resolution
            dim = merge.out_dim
            grid_size = merge.grid_size
            self.grid_sizes.append(grid_size)

        self.stack = nn.Sequential(*stack)

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            GlobalAvgPool(),
            nn.Dropout(head_drop_p) if head_drop_p > 0 else nn.Identity(),
            nn.Linear(dim, num_classes),
        )

    def forward_features(self, x):
        # embed into patches
        x = rearrange(x, "b c ... -> b ... c")
        x = self.patch_embed(x)
        # swin
        if hasattr(self, "abs_pe"):
            x = self.abs_pe(x)
        x = self.stack(x)
        return x

    def forward_classify(self, z):
        # classifier head
        z = rearrange(z, "b ... c-> b (...) c")
        return self.head(z)

    def forward(self, x):
        z = self.forward_features(x)
        c = self.forward_classify(z)
        return c
