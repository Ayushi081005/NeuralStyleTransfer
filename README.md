# 🎨 Artistic Image Stylization

### Using Digital Image Processing & Neural Style Transfer

An interactive **image stylization web application** that transforms ordinary photographs into artistic images using a combination of **classical Digital Image Processing (DIP)** techniques and **deep-learning-based Neural Style Transfer (NST)**.

The application provides **7 artistic styles**: six classical image-processing pipelines and a VGG19-based Neural Style Transfer pipeline that allows users to combine the content of an image with the visual style of a reference artwork.

Built with **Python, OpenCV, PyTorch, Torchvision, NumPy, PIL, and Gradio**.

---

## ✨ Features

* 🖼️ **7 Artistic Styles**
  Six classical DIP-based styles and one deep-learning-based Neural Style Transfer option.

* 🎨 **Classical Image Stylization**
  Uses techniques such as edge detection, smoothing, K-means color quantization, histogram equalization, convolution filters, and color transformations.

* 🧠 **Neural Style Transfer**
  Uses a pretrained **VGG19** network to transfer the visual characteristics of a reference artwork onto the uploaded image.

* 🎛️ **Interactive Controls**
  Users can adjust parameters such as edge detection method, color levels, smoothing strength, edge thickness, color tone, NST steps, and NST resolution.

* ⚡ **GPU Acceleration**
  Neural Style Transfer automatically uses CUDA when an NVIDIA GPU is available and falls back to CPU otherwise.

* 🚀 **Automatic Mixed Precision**
  CUDA-based NST can use FP16 Automatic Mixed Precision to improve performance on supported GPUs.

* 📊 **Live Progress Tracking**
  Gradio displays preprocessing, stylization, and Neural Style Transfer optimization progress.

* 💜 **Custom Interactive UI**
  Includes a custom-themed Gradio interface with an information section, feature showcase, pipeline visualization, and responsive layout.

---

## 🖌️ Supported Artistic Styles

| Style                           | Technique                                                                                           |
| ------------------------------- | --------------------------------------------------------------------------------------------------- |
| 🎨 **Cartoon**                  | Mean-shift smoothing → K-means color quantization → edge detection → edge dilation → compositing    |
| 🖼️ **Oil Painting**            | Bilateral filtering → mean-shift smoothing → color-tone transformation → unsharp-mask sharpening    |
| ✏️ **Pencil Sketch**            | Grayscale conversion → color-dodge blending → Gaussian blur → edge detection                        |
| 💧 **Watercolor**               | Bilateral filtering → K-means quantization → color transformation → CLAHE                           |
| 🌈 **Pop Art**                  | CLAHE → K-means quantization → HSV saturation enhancement → edge enhancement → color transformation |
| 🗿 **Emboss / Relief**          | Bilateral filtering → directional convolution → blending → CLAHE → color transformation             |
| 🧠 **Reference Style Transfer** | VGG19-based Neural Style Transfer using content and style feature representations                   |

---

## 🧠 Neural Style Transfer

The **Reference Style Transfer** mode uses a pretrained **VGG19** convolutional neural network to extract content and style representations from the uploaded images.

### Pipeline

```text
             Content Image
                  │
                  ▼
          Image Preprocessing
                  │
                  ▼
              VGG19
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Content Features       Style Features
        │                   │
        │              Gram Matrices
        │                   │
        └─────────┬─────────┘
                  ▼
            Loss Function
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   Content Loss          Style Loss
        │                   │
        └─────────┬─────────┘
                  ▼
          Adam Optimization
                  │
                  ▼
         Stylized Image
```

### Implementation Details

* Uses a pretrained **VGG19** model from `torchvision`.
* The network is frozen during optimization.
* **Content representation:** `conv_4`
* **Style representations:** `conv_1` through `conv_5`
* Content loss is calculated using **Mean Squared Error (MSE)**.
* Style loss is calculated using **Gram-matrix MSE**.
* The generated image is optimized directly using **Adam**.
* Default learning rate: `0.05`
* Default content weight: `1`
* Default style weight: `1,000,000`
* NST steps can be configured from **50 to 500**.
* NST image resolution can be configured from **128 × 128 to 512 × 512**.
* CUDA is automatically selected when available.
* GPU execution can use **FP16 Automatic Mixed Precision**.

---

## 🎛️ Configurable Parameters

| Parameter              | Range / Default                   | Purpose                                        |
| ---------------------- | --------------------------------- | ---------------------------------------------- |
| **Artistic Style**     | 7 options / `Cartoon`             | Selects the stylization pipeline               |
| **Edge Detection**     | `Canny`, `Sobel`, `Laplacian`     | Controls how edges are detected                |
| **Colour Tone**        | `Warm`, `Cool`, `Vintage`, `None` | Changes the overall color temperature          |
| **Colour Levels**      | 4–20 / `8`                        | Controls K-means color quantization            |
| **Smoothing Strength** | 1–15 / `7`                        | Controls smoothing/filter intensity            |
| **Edge Thickness**     | 1–5 / `2`                         | Controls the thickness of detected edges       |
| **NST Steps**          | 50–500 / `200`                    | Controls the number of optimization iterations |
| **NST Image Size**     | 128–512 / `256`                   | Controls NST resolution and processing time    |

> **Tip:** Increasing NST steps and image size can improve the final result but will increase processing time and GPU/CPU memory usage.

---

## 🔬 Digital Image Processing Techniques

The six classical artistic styles demonstrate several important image-processing concepts.

### Edge Detection

The application supports:

* **Canny Edge Detection**
* **Sobel Edge Detection**
* **Laplacian Edge Detection**

These techniques are primarily used by the Cartoon and Pencil Sketch pipelines.

### K-means Color Quantization

K-means clustering reduces the number of distinct colors in an image.

```text
Original Image
      ↓
Pixel Color Values
      ↓
K-means Clustering
      ↓
Reduced Color Palette
      ↓
Stylized Image
```

This creates the simplified color appearance used in styles such as **Cartoon, Watercolor, and Pop Art**.

### Histogram Equalization

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** is used to improve local contrast while limiting excessive amplification of noise.

It contributes to:

* Watercolor
* Pop Art
* Emboss / Relief

### Spatial Filtering

The project uses several spatial filtering techniques including:

* Bilateral filtering
* Gaussian-based operations
* Mean-shift filtering
* Sharpening kernels
* Emboss convolution kernels

These operations create different artistic textures and visual effects.

---

## ⚙️ Application Architecture

```text
                        ┌──────────────────┐
                        │   Gradio UI      │
                        │    (app.py)      │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ process_image()  │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
           Classical DIP                Neural Style
             Pipelines                    Transfer
                    │                         │
                    ▼                         ▼
             OpenCV / NumPy              VGG19 / PyTorch
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                        ┌──────────────────┐
                        │ Stylized Image   │
                        └──────────────────┘
```

---

## 📂 Project Structure

```text
NeuralStyleTransfer/
│
├── nst/
│   ├── model.py              # VGG19 model and device configuration
│   └── transfer.py           # Neural Style Transfer optimization
│
├── styles/
│   └── pipelines.py          # Six classical artistic pipelines
│
├── ui/
│   └── layout.py             # Gradio UI, CSS and HTML components
│
├── utils/
│   └── image_ops.py          # Image preprocessing and shared operations
│
├── app.py                    # Main Gradio application
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have:

* Python **3.9+**
* `pip`
* A working browser
* Optional: NVIDIA GPU with CUDA for faster Neural Style Transfer

---

### 1. Clone the Repository

```bash
git clone https://github.com/Ayushi081005/NeuralStyleTransfer.git
```

Navigate into the project:

```bash
cd NeuralStyleTransfer
```

---

### 2. Create a Virtual Environment

Recommended:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project uses dependencies similar to:

```text
torch==2.3.1
torchvision==0.18.1
opencv-python
numpy<2
Pillow
gradio
```

---

## ▶️ Running the Application

Start the Gradio application:

```bash
python app.py
```

After initialization, Gradio will provide a local URL similar to:

```text
http://127.0.0.1:7860
```

Open the URL in your browser to access the application.

---

## 🎮 How to Use

### Step 1 — Upload an Image

Upload your photograph using:

**Upload Your Image**

### Step 2 — Select a Style

Choose one of the available artistic styles:

* Cartoon
* Oil Painting
* Pencil Sketch
* Watercolor
* Pop Art
* Emboss / Relief
* Reference Style Transfer

### Step 3 — Configure Parameters

Depending on the selected style, adjust:

* Edge detection method
* Colour levels
* Smoothing strength
* Edge thickness
* Colour tone
* NST optimization steps
* NST image size

### Step 4 — Reference Style Transfer

If you select **Reference Style Transfer**, upload a second image under:

**Style Reference**

The second image provides the artistic texture and visual style.

### Step 5 — Generate

Click:

**✨ Generate Stylized Image**

The application will display processing progress and generate the final stylized image.

---

## ⚡ GPU vs CPU

The application automatically determines the available device.

```text
NVIDIA GPU + CUDA available
            ↓
        CUDA / GPU
            ↓
   Faster NST processing
```

Otherwise:

```text
No CUDA available
       ↓
      CPU
       ↓
Standard precision NST
```

The application also displays whether Neural Style Transfer is running on **GPU 🚀** or **CPU**.

> Classical OpenCV-based styles generally require significantly less computation than Neural Style Transfer.

---

## 🧹 Memory Management

Neural Style Transfer can consume substantial memory because the optimization process works with VGG19 feature representations.

The project therefore performs cleanup after NST execution by:

* Releasing intermediate tensors
* Clearing the CUDA cache when applicable
* Calling Python garbage collection

This helps reduce memory accumulation when multiple images are processed during the same application session.

---

## 📊 Processing Pipeline

For classical styles:

```text
Upload Image
     ↓
RGB → BGR Conversion
     ↓
Image Preprocessing
     ↓
Selected DIP Pipeline
     ↓
Post-processing
     ↓
BGR → RGB Conversion
     ↓
Stylized Output
```

For Neural Style Transfer:

```text
Content Image + Style Reference
              ↓
       Image Preprocessing
              ↓
            VGG19
              ↓
     Content + Style Features
              ↓
       Content / Style Loss
              ↓
       Adam Optimization
              ↓
        Generated Image
```

---

## 💡 Learning Outcomes

This project provides practical experience with:

* Digital Image Processing
* Computer Vision
* Image filtering
* Edge detection
* K-means clustering
* Histogram equalization
* Color-space transformations
* Convolution operations
* PyTorch
* Transfer learning
* Convolutional Neural Networks
* Neural Style Transfer
* VGG19 feature extraction
* Gram matrices
* Optimization-based image generation
* GPU acceleration
* Gradio application development

---

## 🔮 Future Improvements

Possible extensions include:

* [ ] 🖼️ **Before/After Image Comparison**
* [ ] 💾 **Download Stylized Artwork**
* [ ] 🎨 **Custom Style Presets**
* [ ] 🧠 **Additional Deep Learning Style Transfer Models**
* [ ] ⚡ **Further GPU Optimization**
* [ ] 📱 **Improved Mobile-Friendly Interface**
* [ ] 🖌️ **Custom Brush / Artistic Controls**
* [ ] 🎚️ **Adjustable Content-to-Style Weight**
* [ ] 📚 **Multiple Reference Images**
* [ ] 🧪 **Quantitative Evaluation of Stylization Quality**

---

## 🤝 Contributing

Contributions, suggestions, issues, and feature requests are welcome.

If you would like to improve the project:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your changes.
5. Open a Pull Request.

---

## 📄 License

This project is available under the **MIT License**.

---

## 👩‍💻 Author

**Ayushi Shrivastava**

GitHub: **[@Ayushi081005](https://github.com/Ayushi081005)**

---

## ⭐ Support

If you found this project interesting or useful, consider giving the repository a ⭐ on GitHub!

**Repository:**
https://github.com/Ayushi081005/NeuralStyleTransfer
