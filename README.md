# Encoder weights (timm / segmentation-models-pytorch)

Unmodified copies of five pretrained ImageNet encoders from the Hugging Face Hub, stored directly in this git
repository, for machines that cannot reach huggingface.co. Each folder holds `model.safetensors`, `config.json`
and the upstream model card (`README.md`). This mirror does not relicense anything.

## Get the weights

Everything is in the repo itself (no releases, no Git LFS). GitHub allows at most 100 MB per file,
so the two largest files are stored as `<name>.part000`, `<name>.part001` next to their config files.

```
git clone --depth 1 https://github.com/zhang-and-wang-et-al/encoder-weights.git
cd encoder-weights
python join.py
```

`join.py` works offline (standard library only): it joins the parts into `model.safetensors` and checks the SHA256.
`python join.py --only <folder>` joins one model, `--delete-parts` frees the space afterwards.

Without Python, join by hand and compare the SHA256 with the table:

```
:: Windows (cmd)
copy /b "timm__convnext_tiny.in12k_ft_in1k\model.safetensors.part*" "timm__convnext_tiny.in12k_ft_in1k\model.safetensors"
certutil -hashfile "timm__convnext_tiny.in12k_ft_in1k\model.safetensors" SHA256

# Linux / macOS
cat smp-hub__resnet50.imagenet/model.safetensors.part* > smp-hub__resnet50.imagenet/model.safetensors && sha256sum smp-hub__resnet50.imagenet/model.safetensors
```

## Contents

- `timm__resnet18d.ra2_in1k/`: https://huggingface.co/timm/resnet18d.ra2_in1k, revision `ad962375b19d6416bf7e99bfcad25c9510a3d96c`
- `smp-hub__resnet34.imagenet/`: https://huggingface.co/smp-hub/resnet34.imagenet, revision `fa54a662fb7bb5149049a30c11fac0deb86cf7e7`
- `smp-hub__resnet50.imagenet/`: https://huggingface.co/smp-hub/resnet50.imagenet, revision `ea4ecdd751d7c8bf8a3e35546cedfd02dd006f42`
- `smp-hub__mit_b2.imagenet/`: https://huggingface.co/smp-hub/mit_b2.imagenet, revision `2ab56e8487d613b0a6964c068bbc966801097db4`
- `timm__convnext_tiny.in12k_ft_in1k/`: https://huggingface.co/timm/convnext_tiny.in12k_ft_in1k, revision `aa096f03029c7f0ec052013f64c819b34f8ad790`

| File | Size | Parts | SHA256 |
|---|---|---|---|
| `timm__resnet18d.ra2_in1k/model.safetensors` | 47 MB | stored whole | `0ad5d924c4269ad863590b6ccdd19f5fb48d0bae695c1dd6751de0b0282d8361` |
| `smp-hub__resnet34.imagenet/model.safetensors` | 87 MB | stored whole | `6549ccd46a2d20a0a74fca866cfc5c6f9b3672d468b0a92aa94a58545a61103e` |
| `smp-hub__resnet50.imagenet/model.safetensors` | 102 MB | 2 | `df1aad85e18536504a4c8597118364e291ff3a9c4b56dd9b3a4900642e4c3a7c` |
| `smp-hub__mit_b2.imagenet/model.safetensors` | 99 MB | stored whole | `ab18b54f81748427bae1183160822903470bcca0e3121944c0d185b4b1816a95` |
| `timm__convnext_tiny.in12k_ft_in1k/model.safetensors` | 114 MB | 2 | `a1aefa409b513cf209b085424eb3efffe4e4a9f511491bc2c12ea35209e6bb95` |

The SHA256 values are the ones the Hugging Face Hub lists for these files; every download was checked against them.
The `pytorch_model.bin` duplicates of the two timm models are not included (same weights, pickle format).

## Licences and origin

- `timm/resnet18d.ra2_in1k`, `timm/convnext_tiny.in12k_ft_in1k`: Apache License 2.0 according to their model cards.
- `smp-hub/*`: the model cards say `license: other` and name the original files:
  - `resnet34`, `resnet50`: torchvision weights (`download.pytorch.org/models/resnet34-333f7ec4.pth`; see the card in the
    `resnet50` folder for its file).
  - `mit_b2`: the Mix Transformer encoder of SegFormer (NVIDIA), republished by segmentation-models-pytorch. SegFormer
    is under the NVIDIA Source Code License, which allows research and evaluation use only. Check it before any
    other use.

## Use offline

timm:

```python
import timm
m = timm.create_model("resnet18d", pretrained=True,
                      pretrained_cfg_overlay=dict(file="timm__resnet18d.ra2_in1k/model.safetensors"))
```

segmentation-models-pytorch (build without download, then load the encoder):

```python
import segmentation_models_pytorch as smp
from safetensors.torch import load_file
model = smp.Unet("resnet34", encoder_weights=None)
model.encoder.load_state_dict(load_file("smp-hub__resnet34.imagenet/model.safetensors"))
```

These two snippets were written from the libraries' documentation and not run here.
