# 🎨 Artistic Image Stylization

**Using Digital Image Processing & Neural Style Transfer**

An interactive Gradio web app that turns your photos into art. Six presets are built on classical Digital Image Processing (DIP) techniques — edge detection, spatial filtering, histogram equalization, and color transforms — and a seventh option uses a VGG19-based **Neural Style Transfer (NST)** engine to fuse your photo's content with the style of any reference artwork.

Built with **Python**, **OpenCV**, **PyTorch**, and **Gradio**.

---

## ✨ Features

- 🖼️ **7 artistic rendering styles** — 6 classical DIP pipelines + 1 deep-learning style transfer
- 🎛️ **Real-time, interactive controls** — tune edge method, color tone, smoothing, and more
- ⚡ **GPU-accelerated NST** — runs on CUDA with Automatic Mixed Precision (FP16) when a GPU is available, falls back to CPU otherwise
- 📊 **Live progress feedback** — a Gradio progress bar tracks preprocessing, pipeline application, and (for NST) percentage-complete during optimization
- 🎨 **Custom themed UI** — a lilac/purple design system defined entirely in CSS, with an about section, a technique showcase grid, and a pipeline-flow diagram baked into the page

---

## 🖌️ Supported Styles & How They Work

| Style | Actual Pipeline |
|---|---|
| **Cartoon** | Mean-shift smoothing (`cv2.pyrMeanShiftFiltering`) → K-means color quantization → edge detection (Canny/Sobel/Laplacian, optionally dilated for thickness) → edges composited as black outlines over the quantized image |
| **Oil Painting** | Repeated bilateral filtering → mean-shift smoothing → color-tone shift → unsharp-mask sharpening |
| **Pencil Sketch** | Grayscale color-dodge blend (inverted Gaussian-blurred image divided into the grayscale original) combined with inverted edge-detection output |
| **Watercolor** | Strong repeated bilateral filtering → K-means quantization (boosted color levels) → color-tone shift → CLAHE histogram equalization |
| **Pop Art** | CLAHE contrast boost → K-means quantization → saturation boost (HSV) → edge-enhance convolution kernel → color-tone shift |
| **Emboss / Relief** | Bilateral smoothing → directional emboss convolution kernel, blended with the original → CLAHE → color-tone shift |
| **Reference Style Transfer** | VGG19-based Neural Style Transfer (Gatys et al. approach) — see below |

### Neural Style Transfer details

- Uses a frozen, pretrained **VGG19** (`torchvision.models.vgg19`, ImageNet weights), loaded once at startup.
- **Content layer:** `conv_4`. **Style layers:** `conv_1` through `conv_5`. Losses are computed via MSE (content) and Gram-matrix MSE (style).
- Optimized with **Adam** (lr = 0.05) directly on the pixels of a clone of the content image, rather than the classic LBFGS — chosen for speed.
- On CUDA, runs under **Automatic Mixed Precision (FP16)** for roughly a 2× speedup on supported GPUs; on CPU it runs in standard precision.
- Progress is reported every 10 optimization steps via a callback.

---

## 🎛️ Configurable Parameters

| Parameter | Range / Default | Purpose | Applies To |
|---|---|---|---|
| **Artistic Style** | Dropdown — default `Cartoon` | Selects the active rendering pipeline | All styles |
| **Edge Detection Method** | `canny` \| `sobel` \| `laplacian` — default `canny` | Canny uses fixed thresholds (50/150); Sobel and Laplacian use gradient magnitude + Otsu auto-thresholding | Cartoon, Pencil Sketch |
| **Colour Tone** | `warm` \| `cool` \| `vintage` \| `none` — default `warm` | Scales the R/G/B channels to shift color temperature | Oil Painting, Watercolor, Pop Art, Emboss / Relief |
| **Colour Levels (K-means)** | 4–20 (step 1) — default `8` | Number of color clusters (K) used for quantization | Cartoon, Watercolor, Pop Art |
| **Smoothing Strength** | 1–15 (step 1) — default `7` | Scales blur radius / filter iterations depending on the pipeline | Cartoon, Oil Painting, Watercolor, Emboss / Relief |
| **Edge Thickness** | 1–5 (step 1) — default `2` | Dilates detected edges by an NxN kernel before compositing | Cartoon, Pencil Sketch |
| **NST Steps** | 50–500 (step 50) — default `200` | Number of Adam optimization iterations | Reference Style Transfer |
| **NST Image Size** | 128 / 256 / 384 / 512 (step 128) — default `256` | Square resolution used during optimization — larger is higher quality but slower | Reference Style Transfer |

> For **Reference Style Transfer**, upload a second image in the "Style Reference" panel — its texture and color are fused onto your content image. Internally the style weight is fixed at `1,000,000` and content weight at `1`.

---

## 📂 Project Structure

Each source file documents which package it belongs in, and `app.py` imports accordingly:

```
├── nst/
│   ├── model.py         # Loads VGG19 once at startup; exposes device, model, mean/std, loss modules
│   └── transfer.py      # Builds the per-image loss model and runs the Adam/AMP optimization loop
├── styles/
│   └── pipelines.py     # The 6 classical DIP style functions (Cartoon, Oil Painting, etc.)
├── ui/
│   └── layout.py        # Gradio CSS theme + HTML for header, about section, feature grid, footer
├── utils/
│   └── image_ops.py     # Shared OpenCV/NumPy primitives: edge detection, filters, color transforms
├── app.py               # Gradio Blocks UI + process_image() orchestration
└── requirements.txt
```

> ⚠️ **Current repo state:** as of this writing, the actual files in this repository sit flat at the root (`model.py`, `transfer.py`, `pipelines.py`, `layout.py`, `image_ops.py`) rather than inside the `nst/`, `styles/`, `ui/`, and `utils/` folders that `app.py`'s imports expect. Running `python app.py` as-is will raise a `ModuleNotFoundError`. To fix this, either:
> - Move each file into the folder shown above (and add an empty `__init__.py` to each new folder), **or**
> - Update the `import` statements in `app.py`, `pipelines.py`, and `transfer.py` to import from the flat root instead.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip
- (Optional) an NVIDIA GPU with CUDA for faster Neural Style Transfer

### Installation

```bash
git clone https://github.com/Ayushi081005/NeuralStyleTransfer.git
cd NeuralStyleTransfer
pip install -r requirements.txt
```

`requirements.txt`:

```
torch==2.3.1
torchvision==0.18.1
opencv-python
numpy<2
Pillow
gradio
```

### Running the App

```bash
python app.py
```

Once initialized, the terminal prints a local URL (e.g. `http://127.0.0.1:7860`). Open it in your browser.

### How to Use

1. **Upload an image** in the "Upload Your Image" panel.
2. If you selected **Reference Style Transfer**, also upload a second image in the "Style Reference" panel.
3. Pick an **Artistic Style** and adjust the sliders/options relevant to it — irrelevant controls are simply ignored by that pipeline.
4. Click **✨ Generate Stylized Image** and watch the progress bar track preprocessing, pipeline application, and (for NST) live optimization percentage.
5. View the result in the **Stylized Output** panel.

---

## 🧠 Behind the Scenes

- **Device Acceleration** — `model.py` resolves `device` to `cuda` when available (and enables `cudnn.benchmark`), otherwise `cpu`. `app.py` labels progress messages accordingly (`GPU 🚀` / `CPU`).
- **Image Preprocessing** — `preprocess_image()` resizes the long edge of the uploaded image to 512 px and applies a bilateral filter before any style pipeline runs, keeping memory and runtime predictable.
- **VGG19 loaded once** — the network is loaded and frozen a single time at import, so repeated Reference Style Transfer runs don't reload it.
- **Memory cleanup** — after each NST run, tensors are deleted, the CUDA cache is cleared (if applicable), and `gc.collect()` is called to avoid memory creep across repeated runs in the same session.
- **Unified output path** — regardless of which pipeline runs, the result is converted back to RGB and returned as a `PIL.Image` so the Gradio output component always gets a consistent type.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Check the [issues page](https://github.com/Ayushi081005/NeuralStyleTransfer/issues) or open a pull request.

## 📄 License

This project is available under the [MIT License](LICENSE).
