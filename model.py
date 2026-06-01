"""
nst/model.py
Loads VGG19 once at startup and exposes the shared device, model, mean & std.
"""

import torch
import torch.nn as nn
from torchvision import models

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[model] Using device: {device}")
if device.type == "cuda":
    print(f"[model] GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

# ── VGG19 (loaded once, frozen) ───────────────────────────────────────────────
print("[model] Loading VGG19…")
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
for p in vgg.parameters():
    p.requires_grad_(False)

vgg_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
vgg_std  = torch.tensor([0.229, 0.224, 0.225], device=device)
print("[model] VGG19 ready.")


# ── Loss modules ──────────────────────────────────────────────────────────────
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss   = torch.tensor(0.0, device=device)

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x


def gram_matrix(x):
    b, c, h, w = x.size()
    F = x.view(b * c, h * w)
    return torch.mm(F, F.t()).div(b * c * h * w)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss   = torch.tensor(0.0, device=device)

    def forward(self, x):
        self.loss = nn.functional.mse_loss(gram_matrix(x), self.target)
        return x


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std  = std.view(-1, 1, 1)

    def forward(self, x):
        return (x - self.mean) / self.std
