import gradio as gr
import cv2
import numpy as np
from PIL import Image

PSA_LIMIT = 0.55
CLEAR_PASS_MAX = 0.54
BORDERLINE_HIGH = 0.56


# -----------------------------
# Geometry helpers
# -----------------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_card(img_bgr, quad, out_w=900):
    rect = order_points(quad.astype("float32"))
    (tl, tr, br, bl) = rect

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)

    maxW = int(max(wA, wB))
    maxH = int(max(hA, hB))

    if maxW < 50 or maxH < 50:
        return None

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))

    scale = out_w / maxW
    warped = cv2.resize(warped, (out_w, int(maxH * scale)))

    return warped


# -----------------------------
# Outer card detection
# -----------------------------
def find_outer_quad(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 130)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.02 * peri, True)

    if len(approx) != 4:
        return None

    return approx.reshape(4, 2)


# -----------------------------
# INNER FRAME (simplified + constrained)
# -----------------------------
def find_inner_frame(warped):
    h, w = warped.shape[:2]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)

    col = edges.sum(axis=0)
    row = edges.sum(axis=1)

    # smooth
    col = np.convolve(col, np.ones(25)/25, mode="same")
    row = np.convolve(row, np.ones(25)/25, mode="same")

    # search rails within 2–14%
    x_lo = int(w * 0.02)
    x_hi = int(w * 0.14)

    left = x_lo + np.argmax(col[x_lo:x_hi])
    right = (w - x_lo) - np.argmax(col[w - x_hi:w - x_lo])

    if right <= left:
        return None, edges

    # expected inner height from width
    outer_aspect = h / float(w)
    inner_w = right - left
    expected_h = inner_w * outer_aspect

    # top within 2–14%
    y_lo = int(h * 0.02)
    y_hi = int(h * 0.14)

    top = y_lo + np.argmax(row[y_lo:y_hi])
    bottom = int(top + expected_h)

    if bottom >= h:
        return None, edges

    # aspect sanity
    inner_h = bottom - top
    aspect_inner = inner_h / float(inner_w)
    aspect_outer = h / float(w)

    if abs(aspect_inner - aspect_outer) > 0.08:
        return None, edges

    return (left, top, right, bottom), edges


# -----------------------------
# Centering classification
# -----------------------------
def classify_centering(gL, gR, gT, gB):
    lr = max(gL, gR) / (gL + gR)
    tb = max(gT, gB) / (gT + gB)
    worst = max(lr, tb)

    within = (lr <= PSA_LIMIT) and (tb <= PSA_LIMIT)

    if worst <= CLEAR_PASS_MAX and within:
        bucket = "CLEAR PASS"
    elif worst <= BORDERLINE_HIGH:
        bucket = "BORDERLINE"
    else:
        bucket = "FAIL"

    return lr, tb, worst, within, bucket


# -----------------------------
# Main
# -----------------------------
def analyze(img_pil):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    quad = find_outer_quad(img)
    if quad is None:
        return "Could not detect outer card.", img_pil

    warped = warp_card(img, quad)
    if warped is None:
        return "Could not warp card.", img_pil

    inner, dbg = find_inner_frame(warped)
    if inner is None:
        dbg_rgb = cv2.cvtColor(dbg, cv2.COLOR_GRAY2RGB)
        return "UNCERTAIN: could not reliably detect INNER frame.", Image.fromarray(dbg_rgb)

    x1, y1, x2, y2 = inner
    h, w = warped.shape[:2]

    gL = x1
    gR = w - x2
    gT = y1
    gB = h - y2

    lr, tb, worst, within, bucket = classify_centering(gL, gR, gT, gB)

    overlay = warped.copy()
    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    msg = (
        f"L/R: {round(gL/(gL+gR)*100,1)} / {round(gR/(gL+gR)*100,1)}\n"
        f"T/B: {round(gT/(gT+gB)*100,1)} / {round(gB/(gT+gB)*100,1)}\n"
        f"Within 55/45: {'YES' if within else 'NO'}\n"
        f"Bucket: {bucket}"
    )

    return msg, Image.fromarray(overlay_rgb)


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(), gr.Image(type="pil")],
    title="Card Pre-Grade (Centering Only)",
)

if __name__ == "__main__":
    demo.launch()