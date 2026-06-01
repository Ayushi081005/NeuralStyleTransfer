Artistic Image Stylization Web App

An interactive web application built with Gradio and OpenCV that transforms standard photos into diverse artistic mediums. The application includes traditional computer vision image processing algorithms (e.g., K-means clustering, edge filters) alongside a Neural Style Transfer (NST) engine backed by Deep Learning to fuse the content of one image with the artistic style of another.

📂 Project Structure
Based on your workspace directory layout, the modular components are organized as follows:


├── nst/

│   ├── model.py          # NST neural network setup and device configuration (CPU/GPU)

│   └── transfer.py       # Optimization loop for running Neural Style Transfer

├── styles/

│   └── pipelines.py      # Core CV logic for Cartoon, Oil Painting, Sketch, etc.

├── ui/

│   └── layout.py         # Custom HTML sections and CSS styling configurations

├── utils/

│   └── image_ops.py      # Image loading, preprocessing, and resizing utilities
 
├── app.py                # Main application orchestration script and Gradio interface

└── requirements.txt      # List of dependencies

🎨 Supported Artistic Styles
The app provides 7 distinct rendering pipelines:

Cartoon: Blends bilateral/median smoothing filters with edge masks (Canny, Sobel, or Laplacian) and color quantization via K-means.

Oil Painting: Emphasizes brush stroke effects using regional intensity histograms and localized smoothing.

Pencil Sketch: Extracts fine structural contours and gradients, removing color channels for a classic hand-drawn look.

Watercolor: Combines softened gradient transitions with custom color toning mappings.

Pop Art: Features intense color levels and highly saturated chromatic mapping.

Emboss / Relief: Uses directional convolution matrices to mimic an engraved or 3D stamped sculpture surface.

Reference Style Transfer: Leverages Deep Learning optimization to synthesize a new image matching the contents of your input with the textures/colors of an explicit reference artwork.

⚙️ Configurable Parameters
You can fine-tune your artwork in real-time using the interactive controls:

Control Parameter	Purpose	Applicable Styles
Artistic Style	Selects the active target rendering pipeline	All
Edge Detection Method	Chooses the contour algorithm (canny, sobel, laplacian)	Cartoon, Pencil Sketch
Colour Tone	Shifts palette temperature (warm, cool, vintage, none)	Oil Painting, Watercolor, Pop Art, Emboss
Colour Levels (K-means)	Controls quantization bins (K) for flatter or more complex shading	Cartoon, Watercolor, Pop Art
Smoothing Strength	Sets spatial blur radius to eliminate fine photographic noise	Cartoon, Oil Painting, Watercolor, Emboss
Edge Thickness	Modifies the structural stroke weight of isolated boundaries	Cartoon, Pencil Sketch
NST Steps & Size	Alters optimization iteration count and execution resolution	Reference Style Transfer

🚀 Getting Started
1. Prerequisites & Installation
Clone the repository and install the required foundational libraries:


pip install -r requirements.txt
Your requirements.txt should contain at least:


opencv-python
numpy
Pillow
gradio
torch
torchvision

2. Running the Web Application
Launch the Gradio interface server locally by executing:


python app.py

After initialization, the terminal will display a local address . Open this URL in your web browser to interact with the application.

🧠 Behind the Scenes
Device Acceleration: The system automatically checks for hardware configurations. If an NVIDIA graphics card is present, Neural Style Transfer runs on CUDA for rapid processing; otherwise, it gracefully drops back to CPU execution.

Image Operations: Input files are standardized to target dimensions (512×512 max boundaries) inside utils/image_ops.py to prevent memory bottlenecks during deep network processing.
