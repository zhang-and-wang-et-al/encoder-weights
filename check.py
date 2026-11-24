"""Load every model of this repo offline and run one forward pass.  python check.py"""
import os
os.environ["HF_ENDPOINT"] = "http://127.0.0.1:9"   # nothing listens there: any download attempt fails at once
import offline_weights
import torch, timm
import segmentation_models_pytorch as smp

x = torch.zeros(1, 3, 224, 224)
with torch.no_grad():
    for name in ["resnet18d.ra2_in1k", "convnext_tiny.in12k_ft_in1k"]:
        m = timm.create_model(name, pretrained=True).eval()
        print("ok timm", name, tuple(m(x).shape))
    for enc in ["resnet34", "resnet50", "mit_b2"]:
        m = smp.Unet(enc, encoder_weights="imagenet").eval()
        print("ok smp ", enc, tuple(m(x).shape))
print("all five loaded without a connection")
