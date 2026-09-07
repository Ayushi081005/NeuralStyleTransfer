# 🎨 Artistic Image Stylization Studio

Turn ordinary photos into extraordinary art. This interactive web app combines classic computer vision techniques with deep-learning-based Neural Style Transfer (NST) to render your images in seven distinct artistic styles — all through a clean, real-time Gradio interface.

Built with **Python**, **OpenCV**, **PyTorch**, and **Gradio**.

---

## ✨ Features

- 🖼️ **7 artistic rendering pipelines** — from cartoons to fine-art style transfer
- 🎛️ **Real-time, interactive controls** — tweak parameters and see results instantly
- ⚡ **Automatic GPU acceleration** — uses CUDA when available, falls back to CPU
- 🧩 **Modular architecture** — clean separation between UI, CV pipelines, and NST engine

---

## 🖌️ Supported Styles

| Style | Description |
|---|---|
| **Cartoon** | Bilateral/median smoothing combined with edge masks (Canny, Sobel, or Laplacian) and K-means color quantization |
| **Oil Painting** | Brush-stroke effects via regional intensity histograms and localized smoothing |
| **Pencil Sketch** | Structural contours and gradients rendered in grayscale for a hand-drawn look |
| **Watercolor** | Soft gradient transitions with customizable color-tone mapping |
| **Pop Art** | Bold, highly saturated chromatic mapping and intense color levels |
| **Emboss / Relief** | Directional convolution kernels that simulate an engraved, 3D-stamped surface |
| **Reference Style Transfer** | Deep-learning optimization that fuses your image's content with a reference artwork's texture and color (Neural Style Transfer) |

---

## 🎛️ Configurable Parameters

| Parameter | Range / Default | Purpose | Applies To |
|---|---|---|---|
| **Artistic Style** | Dropdown — default `Cartoon` | Selects the active rendering pipeline | All styles |
| **Edge Detection Method** | `canny` \| `sobel` \| `laplacian` — default `canny` | Chooses the contour algorithm | Cartoon, Pencil Sketch |
| **Colour Tone** | `warm` \| `cool` \| `vintage` \| `none` — default `warm` | Shifts the palette temperature | Oil Painting, Watercolor, Pop Art, Emboss / Relief |
| **Colour Levels (K-means)** | 4–20 (step 1) — default `8` | Controls quantization bins (K) for flatter or more detailed shading | Cartoon, Watercolor, Pop Art |
| **Smoothing Strength** | 1–15 (step 1) — default `7` | Sets the spatial blur radius to reduce photographic noise | Cartoon, Oil Painting, Watercolor, Emboss / Relief |
| **Edge Thickness** | 1–5 (step 1) — default `2` | Adjusts the stroke weight of detected boundaries | Cartoon, Pencil Sketch |
| **NST Steps** | 50–500 (step 50) — default `200` | Number of optimization iterations — higher means better quality but slower | Reference Style Transfer |
| **NST Image Size** | 128 or 256 or 384 or 512 (step 128) — default `256` | Resolution used during style transfer optimization — 256 is fast, 512 is higher quality | Reference Style Transfer |

> For **Reference Style Transfer**, you also upload a second image — the **Style Reference** — whose textures and colors are fused onto your content image. Internally this uses a fixed style weight of `1,000,000` and a content weight of `1`.

---

## 📂 Project Structure

```
├── nst/
│   ├── model.py         # NST network setup and device (CPU/GPU) configuration
│   └── transfer.py      # Optimization loop for running Neural Style Transfer
├── styles/
│   └── pipelines.py     # Core CV logic for Cartoon, Oil Painting, Sketch, etc.
├── ui/
│   └── layout.py        # Custom HTML sections and CSS styling
├── utils/
│   └── image_ops.py     # Image loading, preprocessing, and resizing utilities
├── app.py               # Main application orchestration and Gradio interface
└── requirements.txt      # Project dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

`requirements.txt` includes:

```
opencv-python
numpy
Pillow
gradio
torch
torchvision
```

### Running the App

Launch the Gradio interface locally:

```bash
python app.py
```

Once initialized, the terminal will print a local URL (e.g. `http://127.0.0.1:7860`). Open it in your browser to start creating.

### How to Use

1. **Upload an image** in the "Upload Your Image" panel.
2. If you selected **Reference Style Transfer**, also upload a second image in the "Style Reference" panel — this is the artwork whose style will be transferred onto your image.
3. Choose an **Artistic Style** from the dropdown and adjust the relevant sliders/options (only the ones applicable to your chosen style will affect the output).
4. Click **✨ Generate Stylized Image** and watch the live progress bar as the image is processed.
5. The result appears in the **Stylized Output** panel.

---

## 🧠 Behind the Scenes

- **Device Acceleration** — `nst/model.py` exposes a `device` object that automatically resolves to `cuda` when an NVIDIA GPU is available, falling back to `cpu` otherwise. `app.py` reads this to label progress messages (`GPU 🚀` or `CPU`) during Neural Style Transfer.
- **Image Preprocessing** — Uploaded images are converted from RGB (PIL) to BGR (OpenCV) and passed through `preprocess_image()` in `utils/image_ops.py`, which resizes them to a maximum of 512×512 pixels to keep memory usage in check for both the CV pipelines and the NST engine.
- **Live Progress Feedback** — `app.py` uses Gradio's `Progress` API to report status at each stage: preprocessing, style application, and (for Neural Style Transfer) live percentage updates streamed from `run_neural_style_transfer()`'s optimization loop via a callback.
- **Unified Output Path** — Regardless of which pipeline runs, the final BGR result is converted back to RGB and returned as a `PIL.Image`, so the Gradio output component always receives a consistent image type.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues) or open a pull request.

## 📄 License

This project is available under the [MIT License](LICENSE).
