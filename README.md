# Generalized n-dimensional Swin
`ndswin` extends and generalizes [Swin Transformer](https://arxiv.org/pdf/2103.14030) and its follow ups, [Video Swin](https://arxiv.org/pdf/2106.13230) and [SwIFT](https://arxiv.org/pdf/2307.05916), to inputs of arbitrary dimensionsality. This allows flexible application of vision transformers beyond images and videos, making it useful for tasks involving volumetric, spatiotemporal, and other high-dimensional data.


## Content
- [`classifier.py`](ndswin/swin_classifier.py) Simple swin classifier.
- [`layers/swin.py`](ndswin/layers/swin.py) N-dimensional Shifted Window Transformer layer.
- [`layers/patching.py`](ndswin/layers/patching.py) Nd patch embed and patch merging.
- [`layers/positional.py`](ndswin/layers/positional.py) Positional encodings.
- [`layers/vit.py`](ndswin/layers/vit.py) Nd ViT layer.


## Usage
```python
import torch
import ndswin

# 4D binary classifier
model = ndswin.SwinClassifier(
    space=4,
    dim=32,
    in_channels=3,
    resolution=[16] * 4,
    num_classes=2,
    patch_size=[2] * 4,
    window_size=[4] * 4,
    depth=[4, 4],
    num_heads=[4, 8],
)

x = torch.randn((1, 3, 16, 16, 16, 16,))  # (b, c, t, x, y, z)
model(x)  # (b, 2)
```


## Experiments
Simple experiments of `ndswin` are performed for image classification on CIFAR-10 ([cifar.ipynb](/cifar.ipynb)) and for video classification on HMDB-10 with masked frame prediction / MAE pretraining ([hmdb.ipynb](/hmdb.ipynb)).
