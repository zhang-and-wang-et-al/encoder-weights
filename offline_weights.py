"""Make timm and segmentation-models-pytorch load the weights of this repo, with no connection to huggingface.co.

    import sys; sys.path.insert(0, r"<path to this repo>")
    import offline_weights          # before timm / segmentation_models_pytorch / huggingface_hub
    import timm
    model = timm.create_model("resnet18d.ra2_in1k", pretrained=True)

On import it joins the split files once (see join.py), points the Hugging Face cache at the `hub` folder of this
repo and switches the Hugging Face libraries to offline mode. Standard library only.
"""
import json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUB = HERE / "hub"


def activate():
    man = json.loads((HERE / "manifest.json").read_text())
    if any(not (HERE / e["path"]).exists() or (HERE / e["path"]).stat().st_size != e["size"] for e in man["files"]):
        r = subprocess.run([sys.executable, str(HERE / "join.py")], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("offline_weights: joining the split weight files failed:\n" + r.stdout + r.stderr)
    os.environ["HF_HUB_CACHE"] = str(HUB)
    os.environ["HF_HUB_OFFLINE"] = "1"
    hub = sys.modules.get("huggingface_hub")
    if hub is not None:   # imported before us: its settings were read already, set them directly
        hub.constants.HF_HUB_CACHE = str(HUB)
        hub.constants.HF_HUB_OFFLINE = True


activate()
