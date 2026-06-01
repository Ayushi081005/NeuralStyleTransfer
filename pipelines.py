"""
styles/pipelines.py
Six classical DIP artistic-style pipelines.
Each function takes a BGR numpy image and returns a BGR numpy image.
"""

import cv2
import numpy as np

from utils.image_ops import (
    detect_edges,
    apply_mean_shift,
    apply_bilateral,
    apply_sharpening,
    apply_convolution,
    apply_histogram_eq,
    apply_color_tone,
    apply_color_quantization,
    adjust_saturation,
)


def style_cartoon(img, edge_method, color_levels, smooth_str, edge_thick):
    """
    Pipeline:
      Mean-shift smoothing → K-means colour quantisation
      → Edge detection → Overlay edges as black outlines
    """
    smoothed  = apply_mean_shift(img,
                                 sp=max(5, smooth_str * 3),
                                 sr=max(10, smooth_str * 5))
    quantized = apply_color_quantization(smoothed, k=color_levels)
    edges     = detect_edges(img, method=edge_method)
    if edge_thick > 1:
        edges = cv2.dilate(edges,
                           np.ones((edge_thick, edge_thick), np.uint8),
                           iterations=1)
    mask = cv2.cvtColor(cv2.bitwise_not(edges), cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(quantized, mask)


def style_oil_painting(img, smooth_str, color_tone):
    """
    Pipeline:
      Bilateral smoothing → Mean-shift → Sharpening → Colour tone
    """
    smoothed = apply_bilateral(img, iterations=max(2, smooth_str // 2))
    paint    = cv2.pyrMeanShiftFiltering(smoothed, sp=15, sr=40)
    return apply_sharpening(apply_color_tone(paint, tone=color_tone),
                            amount=0.5)


def style_pencil_sketch(img, edge_method, edge_thick):
    """
    Pipeline:
      Grayscale dodge blend → Edge detection → Weighted composite
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv  = cv2.bitwise_not(gray)
    bs   = 21 + edge_thick * 4
    bs   = bs + 1 if bs % 2 == 0 else bs          # ensure odd kernel
    blurred   = cv2.GaussianBlur(inv, (bs, bs), 0)
    sketch    = cv2.divide(gray, cv2.bitwise_not(blurred), scale=256.0)
    edges_inv = cv2.bitwise_not(detect_edges(img, method=edge_method))
    combined  = cv2.addWeighted(sketch, 0.7, edges_inv, 0.3, 0)
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)


def style_watercolor(img, smooth_str, color_levels, color_tone):
    """
    Pipeline:
      Strong bilateral smoothing → K-means quantisation
      → Colour tone → CLAHE histogram equalisation
    """
    smoothed  = apply_bilateral(img, iterations=3, sc=100, ss=100)
    quantized = apply_color_quantization(smoothed, k=max(12, color_levels + 4))
    toned     = apply_color_tone(quantized, tone=color_tone)
    return apply_histogram_eq(toned, method="clahe", clip_limit=2.0)


def style_pop_art(img, color_levels, color_tone):
    """
    Pipeline:
      CLAHE contrast boost → K-means quantisation
      → Saturation boost → Edge-enhance kernel → Colour tone
    """
    contrast  = apply_histogram_eq(img, method="clahe", clip_limit=4.0)
    quantized = apply_color_quantization(contrast, k=max(4, color_levels))
    enhanced  = apply_convolution(
        adjust_saturation(quantized, factor=1.8), "edge_enhance"
    )
    return apply_color_tone(enhanced, tone=color_tone)


def style_emboss_relief(img, color_tone, smooth_str):
    """
    Pipeline:
      Bilateral smoothing → Emboss convolution → Blend with original
      → CLAHE → Colour tone
    """
    smoothed = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
    blended  = cv2.addWeighted(img, 0.4,
                               apply_convolution(smoothed, "emboss"), 0.6, 0)
    equalized = apply_histogram_eq(blended, method="clahe", clip_limit=2.5)
    return apply_color_tone(equalized, tone=color_tone)
