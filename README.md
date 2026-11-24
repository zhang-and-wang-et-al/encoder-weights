# Encoder weights (timm / segmentation-models-pytorch), usable offline

Unmodified copies of five pretrained ImageNet encoders from the Hugging Face Hub, stored directly in this git
repository in the layout of the Hugging Face cache. After a clone, `timm` and `segmentation-models-pytorch` load
them with their normal calls and no connection to huggingface.co. This mirror does not relicense anything.

| Model | Load it with |
|---|---|
| `timm/resnet18d.ra2_in1k` | `timm.create_model("resnet18d.ra2_in1k", pretrained=True)` |
| `timm/convnext_tiny.in12k_ft_in1k` | `timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True)` |
| `smp-hub/resnet34.imagenet` | `smp.Unet("resnet34", encoder_weights="imagenet")` (any smp architecture) |
| `smp-hub/resnet50.imagenet` | `smp.Unet("resnet50", encoder_weights="imagenet")` |
| `smp-hub/mit_b2.imagenet` | `smp.Unet("mit_b2", encoder_weights="imagenet")` |

## Use

```
git clone --depth 1 https://github.com/zhang-and-wang-et-al/encoder-weights.git
cd encoder-weights
python check.py
```

`check.py` loads all five models and runs one forward pass, with the Hugging Face address pointed at nothing.
It needs `torch`, `timm` and `segmentation-models-pytorch` installed; the Python packages are not part of this repo.

In your own code, import `offline_weights` before `timm` / `segmentation_models_pytorch`:

```python
import sys; sys.path.insert(0, r"C:\path\to\encoder-weights")
import offline_weights
import timm, segmentation_models_pytorch as smp

backbone = timm.create_model("resnet18d.ra2_in1k", pretrained=True)
model = smp.Unet("resnet34", encoder_weights="imagenet")
```

On its first import `offline_weights` joins the two split files (see below), then points the Hugging Face cache at
the `hub` folder of this repo and switches the Hugging Face libraries to offline mode.

Without the import, the same thing by hand: run `python join.py` once, then set two environment variables before
starting Python:

```
:: Windows (cmd)
set HF_HUB_CACHE=C:\path\to\encoder-weights\hub
set HF_HUB_OFFLINE=1

# Linux / macOS
export HF_HUB_CACHE=/path/to/encoder-weights/hub HF_HUB_OFFLINE=1
```

While these are set, Hugging Face models that are not in this repo cannot be loaded.

## Tested

A fresh clone was checked with `check.py` on Windows, Python 3.12, CPU, with torch 2.15.0, timm 1.0.31,
segmentation-models-pytorch 0.6.0, huggingface_hub 2.4.0.

segmentation-models-pytorch asks for one fixed revision per encoder. The snapshot folders of the three `smp-hub`
models are named after the revisions that version 0.6.0 asks for. A version of the library that pins other revisions
will not find them; then load the file directly:

```python
from safetensors.torch import load_file
model = smp.Unet("resnet34", encoder_weights=None)
model.encoder.load_state_dict(load_file(r"hub\models--smp-hub--resnet34.imagenet\snapshots\7a57b34f723329ff020b3f8bc41771163c519d0c\model.safetensors"))
```

## Files

No releases, no Git LFS. GitHub allows at most 100 MB per file, so the two largest files are stored as
`model.safetensors.part000` / `.part001`; `join.py` (standard library only) joins them and checks the SHA256.

| File (under `hub/models--<org>--<name>/snapshots/<revision>/`) | Upstream revision | Size | Parts | SHA256 |
|---|---|---|---|---|
| `timm/resnet18d.ra2_in1k` `model.safetensors` | `ad962375b19d6416bf7e99bfcad25c9510a3d96c` | 47 MB | stored whole | `0ad5d924c4269ad863590b6ccdd19f5fb48d0bae695c1dd6751de0b0282d8361` |
| `timm/convnext_tiny.in12k_ft_in1k` `model.safetensors` | `aa096f03029c7f0ec052013f64c819b34f8ad790` | 114 MB | 2 | `a1aefa409b513cf209b085424eb3efffe4e4a9f511491bc2c12ea35209e6bb95` |
| `smp-hub/resnet34.imagenet` `model.safetensors` | `7a57b34f723329ff020b3f8bc41771163c519d0c` | 87 MB | stored whole | `6549ccd46a2d20a0a74fca866cfc5c6f9b3672d468b0a92aa94a58545a61103e` |
| `smp-hub/resnet50.imagenet` `model.safetensors` | `00cb74e366966d59cd9a35af57e618af9f88efe9` | 102 MB | 2 | `df1aad85e18536504a4c8597118364e291ff3a9c4b56dd9b3a4900642e4c3a7c` |
| `smp-hub/mit_b2.imagenet` `model.safetensors` | `868ab6f13871dcf8c3d9f90ee4519403475b65ef` | 99 MB | stored whole | `ab18b54f81748427bae1183160822903470bcca0e3121944c0d185b4b1816a95` |

The SHA256 values are the ones the Hugging Face Hub lists; every download was checked against them. For the three
`smp-hub` models the weights and `config.json` are identical at the listed revision and at the current `main`; the
model card (`README.md`) in the folder is the one from `main`. The `pytorch_model.bin` duplicates of the two timm
models are not included (same weights, pickle format).

## Licences and origin

- `timm/resnet18d.ra2_in1k`, `timm/convnext_tiny.in12k_ft_in1k`: Apache License 2.0 according to their model cards.
- `smp-hub/*`: the model cards say `license: other` and name the original files:
  - `resnet34`, `resnet50`: torchvision weights (see the "Original weights URL" in each card).
  - `mit_b2`: the Mix Transformer encoder of SegFormer (NVIDIA), republished by segmentation-models-pytorch. SegFormer
    is under the NVIDIA Source Code License, which allows research and evaluation use only. Check it before any
    other use.
