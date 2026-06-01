"""
utils/image_ops.py
Low-level OpenCV / NumPy image operations shared by all style pipelines.
"""

import cv2
import numpy as np


def preprocess_image(image, target_size=512):
    """Resize to target_size on the long edge, then bilateral-smooth."""
    h, w  = image.shape[:2]
    scale = target_size / max(h, w)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return cv2.bilateralFilter(resized, d=9, sigmaColor=75, sigmaSpace=75)


# ── Edge detection ────────────────────────────────────────────────────────────
def detect_edges(image, method="canny", low=50, high=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "canny":
        return cv2.Canny(gray, low, high)
    elif method == "sobel":
        gx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.uint8(np.clip(np.sqrt(gx**2 + gy**2), 0, 255))
        _, edges = cv2.threshold(mag, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return edges
    elif method == "laplacian":
        lap     = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        lap_abs = np.uint8(np.clip(np.abs(lap), 0, 255))
        _, edges = cv2.threshold(lap_abs, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return edges
    return cv2.Canny(gray, low, high)


# ── Spatial filters ───────────────────────────────────────────────────────────
def apply_mean_shift(image, sp=10, sr=30):
    return cv2.pyrMeanShiftFiltering(image, sp=sp, sr=sr)


def apply_bilateral(image, iterations=3, d=9, sc=75, ss=75):
    result = image.copy()
    for _ in range(iterations):
        result = cv2.bilateralFilter(result, d=d, sigmaColor=sc, sigmaSpace=ss)
    return result


def apply_sharpening(image, amount=1.0):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return np.clip(
        cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0),
        0, 255
    ).astype(np.uint8)


def apply_convolution(image, kernel_type="emboss"):
    kernels = {
        "emboss":       np.array([[-2,-1,0],[-1,1,1],[0,1,2]],     dtype=np.float32),
        "sharpen":      np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],     dtype=np.float32),
        "edge_enhance": np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=np.float32),
    }
    return cv2.filter2D(image, -1, kernels.get(kernel_type, kernels["emboss"]))


# ── Colour transforms ─────────────────────────────────────────────────────────
def apply_histogram_eq(image, method="clahe", clip_limit=3.0):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    if method == "clahe":
        l = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(8, 8)).apply(l)
    else:
        l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def apply_color_tone(image, tone="warm"):
    r = image.copy().astype(np.float32)
    if tone == "warm":
        r[:, :, 2] = np.clip(r[:, :, 2] * 1.15, 0, 255)
        r[:, :, 0] = np.clip(r[:, :, 0] * 0.90, 0, 255)
    elif tone == "cool":
        r[:, :, 0] = np.clip(r[:, :, 0] * 1.15, 0, 255)
        r[:, :, 2] = np.clip(r[:, :, 2] * 0.90, 0, 255)
    elif tone == "vintage":
        r[:, :, 2] = np.clip(r[:, :, 2] * 1.10, 0, 255)
        r[:, :, 1] = np.clip(r[:, :, 1] * 0.95, 0, 255)
        r[:, :, 0] = np.clip(r[:, :, 0] * 0.85, 0, 255)
    return np.uint8(np.clip(r, 0, 255))


def apply_color_quantization(image, k=8):
    data     = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10,
                                    cv2.KMEANS_PP_CENTERS)
    return np.uint8(centers)[labels.flatten()].reshape(image.shape)


def adjust_saturation(image, factor=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(np.uint8(hsv), cv2.COLOR_HSV2BGR)
