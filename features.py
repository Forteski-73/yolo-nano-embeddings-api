import cv2
import numpy as np

from PIL import Image

from skimage.feature import local_binary_pattern

"""
# =====================================================
# HSV HISTOGRAM
# =====================================================
def hsv_histogram(img: Image.Image):

    img_np = np.array(
        img.convert("RGB")
    )

    hsv = cv2.cvtColor(
        img_np,
        cv2.COLOR_RGB2HSV
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )

    hist = cv2.normalize(
        hist,
        hist
    ).flatten()

    return hist.tolist()

"""

# =====================================================
# HSV HISTOGRAM
# =====================================================
def hsv_histogram(img: Image.Image):

    # =================================================
    # RGB -> NUMPY
    # =================================================
    img_np = np.array(
        img.convert("RGB")
    )

    # =================================================
    # HSV
    # =================================================
    hsv = cv2.cvtColor(
        img_np,
        cv2.COLOR_RGB2HSV
    )

    # =================================================
    # CANAIS
    # =================================================
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # =================================================
    # IGNORA:
    # - branco
    # - cinza
    # - preto
    # - sombra
    # =================================================
    # baixa saturação = sem cor relevante
    # =================================================
    mask = (
        (s > 40)
        &
        (v > 40)
    ).astype(np.uint8)

    # =================================================
    # HISTOGRAMA HUE
    # =================================================
    hist_h = cv2.calcHist(

        [h],

        [0],

        mask,

        [36],

        [0, 180]
    )

    # =================================================
    # NORMALIZA
    # =================================================
    hist_h = cv2.normalize(
        hist_h,
        hist_h
    ).flatten()

    return hist_h.tolist()

# =====================================================
# HISTOGRAMA RGB
# (compatibilidade legado)
# =====================================================
def color_histogram(img: Image.Image):

    img_np = np.array(
        img.convert("RGB")
    )

    hist = cv2.calcHist(
        [img_np],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )

    hist = cv2.normalize(
        hist,
        hist
    ).flatten()

    return hist.tolist()

# =====================================================
# LBP TEXTURE
# =====================================================
def texture_lbp(img: Image.Image):

    gray = np.array(
        img.convert("L")
    )

    lbp = local_binary_pattern(
        gray,
        P=24,
        R=3,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 27),
        range=(0, 26)
    )

    hist = hist.astype("float")

    hist /= (
        hist.sum() + 1e-6
    )

    return hist.tolist()

# =====================================================
# EDGE DENSITY
# =====================================================
def edge_density(img: Image.Image):

    gray = np.array(
        img.convert("L")
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    density = (
        np.sum(edges > 0)
        / edges.size
    )

    return float(density)

# =====================================================
# SYMMETRY SCORE
# =====================================================
def symmetry_score(img: Image.Image):

    gray = np.array(
        img.convert("L")
    )

    flipped = np.fliplr(gray)

    diff = np.mean(
        np.abs(gray - flipped)
    )

    score = 1 - (
        diff / 255.0
    )

    return float(score)