"""
nst/transfer.py
Builds the style-transfer model and runs the optimisation loop.
"""

import copy
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T

from nst.model import (
    device, vgg, vgg_mean, vgg_std,
    ContentLoss, StyleLoss, Normalization,
)


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(style_img, content_img,
                content_layers=None, style_layers=None):
    if content_layers is None:
        content_layers = ["conv_4"]
    if style_layers is None:
        style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    cnn   = copy.deepcopy(vgg)
    norm  = Normalization(vgg_mean, vgg_std).to(device)

    content_losses, style_losses = [], []
    model = nn.Sequential(norm)
    i = 0

    for layer in cnn.children():
        if   isinstance(layer, nn.Conv2d):      i += 1; name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):        name = f"relu_{i}"; layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):   name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d): name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            cl = ContentLoss(model(content_img).detach())
            model.add_module(f"content_loss_{i}", cl)
            content_losses.append(cl)

        if name in style_layers:
            sl = StyleLoss(model(style_img).detach())
            model.add_module(f"style_loss_{i}", sl)
            style_losses.append(sl)

    # trim everything after the last loss layer
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            break
    return model[:j + 1], style_losses, content_losses


# ── Tensor helpers ────────────────────────────────────────────────────────────
def pil_to_tensor(pil_img, imsize=256):
    loader = T.Compose([T.Resize((imsize, imsize)), T.ToTensor()])
    return loader(pil_img.convert("RGB")).unsqueeze(0).to(device, torch.float)


def tensor_to_pil(tensor):
    return T.ToPILImage()(tensor.squeeze(0).cpu().detach().clamp(0, 1))


# ── Main transfer function ────────────────────────────────────────────────────
def run_neural_style_transfer(
    content_pil,
    style_pil,
    num_steps=200,
    style_weight=1_000_000,
    content_weight=1,
    imsize=256,
    progress_cb=None,
):
    """
    Optimised NST:
      • Adam optimizer  (faster per-step than LBFGS, GPU-friendly)
      • AMP / FP16      (on CUDA only – ~2× speedup on RTX cards)
      • Default 256 px  (4× fewer pixels than 512 → ~4× faster)
    """
    ct  = pil_to_tensor(content_pil, imsize)
    st  = pil_to_tensor(style_pil,   imsize)
    inp = ct.clone().requires_grad_(True)

    model, s_losses, c_losses = build_model(st, ct)
    model.requires_grad_(False)

    optimizer = optim.Adam([inp], lr=0.05)
    use_amp   = (device.type == "cuda")
    scaler    = torch.cuda.GradScaler() if use_amp else None

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model(inp)
                loss = (style_weight   * sum(sl.loss for sl in s_losses) +
                        content_weight * sum(cl.loss for cl in c_losses))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            model(inp)
            loss = (style_weight   * sum(sl.loss for sl in s_losses) +
                    content_weight * sum(cl.loss for cl in c_losses))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            inp.clamp_(0, 1)

        if progress_cb and step % 10 == 0:
            progress_cb(step / num_steps)

    result = tensor_to_pil(inp)

    del ct, st, inp, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return result
