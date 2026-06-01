import cv2
import numpy as np
from PIL import Image
import gradio as gr

# ── Project imports ────────────────────────────────────────────────────────────
from utils.image_ops import preprocess_image
from styles.pipelines import (
    style_cartoon,
    style_oil_painting,
    style_pencil_sketch,
    style_watercolor,
    style_pop_art,
    style_emboss_relief,
)
from nst.transfer import run_neural_style_transfer
from nst.model    import device
from ui.layout    import (
    css, header_html, about_html,
    features_html, pipeline_html, footer_html,
)


# ── Core processing function ───────────────────────────────────────────────────
def process_image(
    input_image,
    style,
    edge_method,
    color_levels,
    smoothing_strength,
    edge_thickness,
    color_tone,
    nst_steps,
    nst_size,
    style_reference=None,
    progress=gr.Progress(),
):
    if input_image is None:
        raise gr.Error("Please upload an image first!")
    if style == "Reference Style Transfer" and style_reference is None:
        raise gr.Error("Please upload a style reference image!")

    progress(0.1, desc="Preprocessing…")
    image_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    processed = preprocess_image(image_bgr, target_size=512)

    progress(0.3, desc=f"Applying {style}…")

    if style == "Cartoon":
        result = style_cartoon(processed, edge_method, color_levels,
                               smoothing_strength, edge_thickness)

    elif style == "Oil Painting":
        result = style_oil_painting(processed, smoothing_strength, color_tone)

    elif style == "Pencil Sketch":
        result = style_pencil_sketch(processed, edge_method, edge_thickness)

    elif style == "Watercolor":
        result = style_watercolor(processed, smoothing_strength,
                                  color_levels, color_tone)

    elif style == "Pop Art":
        result = style_pop_art(processed, color_levels, color_tone)

    elif style == "Emboss / Relief":
        result = style_emboss_relief(processed, color_tone, smoothing_strength)

    elif style == "Reference Style Transfer":
        dev_label = "GPU 🚀" if device.type == "cuda" else "CPU"
        progress(0.2, desc=f"Running Neural Style Transfer on {dev_label}…")
        result_pil = run_neural_style_transfer(
            input_image,
            style_reference,
            num_steps=int(nst_steps),
            style_weight=1_000_000,
            content_weight=1,
            imsize=int(nst_size),
            progress_cb=lambda p: progress(
                0.2 + p * 0.7,
                desc=f"Neural style [{dev_label}]: {int(p * 100)}%",
            ),
        )
        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

    else:
        result = processed

    progress(0.9, desc="Finalising…")
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


# ── Gradio UI ──────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(css=css, title="Artistic Image Stylization") as demo:

        # ── Header & info ──────────────────────────────────────────────────
        gr.HTML(header_html)
        gr.HTML(about_html)
        gr.HTML(features_html)
        gr.HTML('<div class="app-section" style="padding:2rem;text-align:center;">'
                '<h2 style="color:var(--lilac-deep);font-size:1.8rem;">Create Your Artwork</h2>'
                '</div>')
        gr.HTML(pipeline_html)

        # ── Image columns ──────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column():
                input_img = gr.Image(type="pil", label="Upload Your Image")
            with gr.Column():
                style_ref_img = gr.Image(
                    type="pil",
                    label="Style Reference (for Reference Style Transfer only)",
                )
            with gr.Column():
                output_img = gr.Image(label="Stylized Output",
                                      elem_classes="output-container")

        # ── Style & edge controls ──────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                style = gr.Dropdown(
                    choices=["Cartoon", "Oil Painting", "Pencil Sketch",
                             "Watercolor", "Pop Art", "Emboss / Relief",
                             "Reference Style Transfer"],
                    value="Cartoon",
                    label="Artistic Style",
                )
            with gr.Column():
                edge_method = gr.Radio(
                    choices=["canny", "sobel", "laplacian"],
                    value="canny",
                    label="Edge Detection Method",
                )
            with gr.Column():
                color_tone = gr.Radio(
                    choices=["warm", "cool", "vintage", "none"],
                    value="warm",
                    label="Colour Tone",
                )

        # ── Sliders ────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                color_levels = gr.Slider(
                    minimum=4, maximum=20, value=8, step=1,
                    label="Colour Levels (K-means)",
                )
            with gr.Column():
                smoothing_strength = gr.Slider(
                    minimum=1, maximum=15, value=7, step=1,
                    label="Smoothing Strength",
                )
            with gr.Column():
                edge_thickness = gr.Slider(
                    minimum=1, maximum=5, value=2, step=1,
                    label="Edge Thickness",
                )

        # ── NST-specific controls ──────────────────────────────────────────
        with gr.Row(elem_classes="nst-options"):
            with gr.Column():
                nst_steps = gr.Slider(
                    minimum=50, maximum=500, value=200, step=50,
                    label="NST Steps  (↑ quality, ↑ time)",
                )
            with gr.Column():
                nst_size = gr.Slider(
                    minimum=128, maximum=512, value=256, step=128,
                    label="NST Image Size  (256 = fast, 512 = high quality)",
                )

        # ── Generate button ────────────────────────────────────────────────
        run_btn = gr.Button("✨ Generate Stylized Image", elem_id="generate-btn")
        run_btn.click(
            fn=process_image,
            inputs=[
                input_img, style, edge_method, color_levels,
                smoothing_strength, edge_thickness, color_tone,
                nst_steps, nst_size, style_ref_img,
            ],
            outputs=output_img,
        )

        gr.HTML(footer_html)

    return demo


# ── Launch ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
