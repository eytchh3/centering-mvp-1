import gradio as gr
import cv2
import numpy as np
from PIL import Image


# =============================
# CONFIG
# =============================

PSA_LIMIT = 0.55
CLEAR_PASS_MAX = 0.54
BORDERLINE_HIGH = 0.56

INNER_PAD_FRAC = 0.006
INNER_HEIGHT_SCALE = 0.90
INNER_H_SCALES = [0.97, 1.00, 1.03]

Y_SCAN_TOP = 0.12
Y_SCAN_BOTTOM = 0.88
Y_SCAN_STEP_FRAC = 0.01
EDGE_BAND = 3


# =============================
# Helpers
# =============================

def _smooth1d(a, k=31):
    k = max(5, k | 1)
    pad = k // 2
    ap = np.pad(a.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(ap, kernel, mode="valid")


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# =============================
# OUTER CARD DETECTION
# =============================

def find_outer_card_quad(img_bgr):
    def _select_best_quad_from_contours(cnts, H, W):
        best = None

        best_aspect_error = None

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 0.05 * H * W:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) != 4:
                continue

            quad = approx.reshape(4, 2)
            xs = quad[:, 0]
            ys = quad[:, 1]
            if (xs < 10).any() or (ys < 10).any() or (xs > (W - 11)).any() or (ys > (H - 11)).any():
                continue

            _, _, w_box, h_box = cv2.boundingRect(quad.astype(np.float32))
            if w_box <= 0:
                continue

            aspect_error = abs((h_box / float(w_box)) - 1.4)
            if best_aspect_error is None or aspect_error < best_aspect_error:
                best_aspect_error = aspect_error
                best = quad

        return best

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img_bgr.shape[:2]
    best = _select_best_quad_from_contours(cnts, H, W)
    if best is not None:
        return best

    gray_fb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_fb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts_fb, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _select_best_quad_from_contours(cnts_fb, H, W)


def warp_card(img_bgr, quad):
    rect = order_points(quad.astype("float32"))
    (tl, tr, br, bl) = rect

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)

    maxW = int(max(wA, wB))
    maxH = int(max(hA, hB))

    if maxW < 100 or maxH < 100:
        return None

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))

    return warped


# =============================
# INNER FRAME (SMARTER VERSION)
# =============================

def find_inner_general(warped_bgr):
    h, w = warped_bgr.shape[:2]
    overlay = warped_bgr.copy()

    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (3, 3), 0)

    sx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    sy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))

    # LEFT / RIGHT from vertical energy
    yA, yB = int(h * 0.18), int(h * 0.78)
    col_energy = sx[yA:yB, :].mean(axis=0)
    col_energy = _smooth1d(col_energy, k=max(31, w // 40))

    x_lo, x_hi = int(w * 0.02), int(w * 0.15)
    winL = col_energy[x_lo:x_hi]
    winR = col_energy[w - x_hi:w - x_lo]

    if winL.size < 10 or winR.size < 10:
        return None, overlay

    x1 = x_lo + int(np.argmax(winL))
    x2 = (w - x_lo) - int(np.argmax(winR[::-1]))

    if x2 <= x1 + int(w * 0.30):
        return None, overlay

    inner_w = x2 - x1
    inner_h = inner_w * (h / float(w)) * INNER_HEIGHT_SCALE

    y1 = int((h - inner_h) / 2)
    y2 = int(y1 + inner_h)

    pad = int(min(h, w) * INNER_PAD_FRAC)

    x1 += pad
    x2 -= pad
    y1 += pad
    y2 -= pad

    if x2 <= x1 or y2 <= y1:
        return None, overlay

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)

    return (x1, y1, x2, y2), overlay


# =============================
# CENTERING LOGIC
# =============================

def classify_centering(gL, gR, gT, gB):
    lr = max(gL, gR) / (gL + gR)
    tb = max(gT, gB) / (gT + gB)
    worst = max(lr, tb)

    within = lr <= PSA_LIMIT and tb <= PSA_LIMIT

    if worst <= CLEAR_PASS_MAX and within:
        bucket = "CLEAR PASS"
    elif worst <= BORDERLINE_HIGH:
        bucket = "BORDERLINE"
    else:
        bucket = "FAIL"

    return lr, tb, within, bucket


# =============================
# MAIN
# =============================

def analyze(img_pil):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    debug = img.copy()

    quad = find_outer_card_quad(img)
    if quad is None:
        debug_img = Image.fromarray(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB))
        return "Could not detect outer card.", img_pil, debug_img

    quad_int = quad.astype(np.int32)
    cv2.polylines(debug, [quad_int], isClosed=True, color=(0, 0, 255), thickness=3)
    for x, y in quad_int:
        cv2.circle(debug, (int(x), int(y)), radius=6, color=(0, 0, 255), thickness=-1)
    debug_img = Image.fromarray(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB))

    warped = warp_card(img, quad)
    if warped is None:
        return "Warp failed.", img_pil, debug_img

    inner, overlay = find_inner_general(warped)
    if inner is None:
        return "UNCERTAIN: could not detect inner boundary.", Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)), debug_img

    x1, y1, x2, y2 = inner
    h, w = warped.shape[:2]

    gL = x1
    gR = w - x2
    gT = y1
    gB = h - y2

    lr, tb, within, bucket = classify_centering(gL, gR, gT, gB)

    msg = (
        f"L/R: {round(lr,3)}\n"
        f"T/B: {round(tb,3)}\n"
        f"Within 55/45: {within}\n"
        f"Bucket: {bucket}"
    )

    return msg, Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)), debug_img


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="pil"),
    outputs=["text", "image", "image"],
)

if __name__ == "__main__":
    demo.launch()
