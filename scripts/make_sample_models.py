"""Writes a few sample checkpoints to ./samples for trying out the viewer.

Requires PyTorch (and optionally torchvision / safetensors):
    pip install torch torchvision safetensors
    python scripts/make_sample_models.py
"""
import os

import torch
import torch.nn as nn

os.makedirs("samples", exist_ok=True)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


mlp = MLP()
torch.save(mlp.state_dict(), "samples/mlp_state_dict.pt")
torch.save(mlp, "samples/mlp_full_model.pt")
torch.save({"epoch": 3, "model_state_dict": mlp.state_dict()}, "samples/mlp_checkpoint.pt")
torch.jit.script(mlp).save("samples/mlp_torchscript.pt")

try:
    import torchvision

    torch.save(torchvision.models.resnet18(weights="DEFAULT").state_dict(), "samples/resnet18.pt")
except Exception as e:  # torchvision missing or offline
    print("skipping resnet18:", e)

try:
    from safetensors.torch import save_file

    save_file(mlp.state_dict(), "samples/mlp.safetensors")
except ImportError:
    print("skipping safetensors (pip install safetensors)")

print("wrote:", sorted(os.listdir("samples")))
