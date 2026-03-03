import gradio as gr
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Config (tuned for your goals)
# -----------------------------
PSA_LIMIT = 0.55          # 55/45 rule (front)
CLEAR_PASS_MAX = 0.54     # "centering eliminated" bucket (comfortable pass)
BORDERLINE_HIGH = 0.56    # above this = clear fail zone for noisy photos


# -----------------------------
# Geometry helpers
# -----------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def warp_card(img_bgr: np.ndarray, quad: np.ndarray, out_w: int = 900):
    """Perspective-correct the detected outer quad to a flat rectangle."""
    rect = order_points(quad.astype("float32"))
    (tl, tr, br, bl) = rect

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)

    maxW = int(max(wA, wB))
    maxH = int(max(hA, hB))

    if maxW < 50 or maxH < 50:
        return None, None

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))

    # normalize width for stable downstream heuristics
    scale = out_w / maxW
    warped = cv2.resize(warped, (out_w, max(1, int(maxH * scale))))

    return warped, rect


# -----------------------------
# Detection: OUTER card quad
# (scanner-like, robust)
# -----------------------------
def find_outer_card_quad(img_bgr: np.ndarray):
    """
    Robust card outline detection:
    - edge map
    - pick largest contour
    - fit rotated rectangle via minAreaRect
    This is more tolerant of messy backgrounds than "exact 4-corner contour".
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 130)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, edges

    c = max(cnts, key=cv2.contourArea)

    # Reject tiny detections (helps when UI/background dominates)
    if cv2.contourArea(c) < 0.03 * (img_bgr.shape[0] * img_bgr.shape[1]):
        return None, edges

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")

    return box, edges


# -----------------------------
# Detection: INNER printed frame
# (improved heuristic)
# -----------------------------
def find_inner_frame_rect(warped_bgr: np.ndarray):
    """
    Detect an inner frame rectangle inside the (already warped) card image.
    Approach:
    - local contrast enhancement (CLAHE)
    - adaptive threshold to isolate edge structure
    - morphology close to connect frame lines
    - choose best interior rectangle contour
    Returns (x1,y1,x2,y2) in warped image coords, or None.
    """
    h, w = warped_bgr.shape[:2]

    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )

    k = max(3, min(h, w) // 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    margin = int(min(h, w) * 0.04)  # must be clearly inside card
    best = None
    best_score = -1.0

    # approximate overall aspect ratio of the warped card
    card_aspect = w / max(1, h)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < (h * w) * 0.06:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        x, y, ww, hh = cv2.boundingRect(approx)
        if x < margin or y < margin or (x + ww) > (w - margin) or (y + hh) > (h - margin):
            continue

        rect_area = ww * hh
        if rect_area <= 0:
            continue

        fill_ratio = area / rect_area
        aspect = ww / max(1, hh)

        # Score favors: large interior rectangles, reasonable "frame-like" fill ratio,
        # and aspect not wildly off from card aspect.
        score = rect_area
        score *= (1.0 - abs(fill_ratio - 0.85))  # frames often ~0.7-0.95 depending on thresholding
        score *= (1.0 - min(0.9, abs(aspect - card_aspect) * 0.2))

        if score > best_score:
            best_score = score
            best = (x, y, x + ww, y + hh)

    return best, th


# -----------------------------
# Centering classification
# -----------------------------
def classify_centering(gL: int, gR: int, gT: int, gB: int):
    lr = max(gL, gR) / (gL + gR)
    tb = max(gT, gB) / (gT + gB)
    worst = max(lr, tb)

    within_55 = (lr <= PSA_LIMIT) and (tb <= PSA_LIMIT)

    if worst <= CLEAR_PASS_MAX and within_55:
        bucket = "CLEAR PASS (centering eliminated)"
    elif worst <= BORDERLINE_HIGH:
        bucket = "BORDERLINE (request better straight-on photo; else UNCERTAIN)"
    else:
        bucket = "FAIL (likely outside 55/45)"

    return lr, tb, worst, within_55, bucket


# -----------------------------
# Corners (photo-based risk flag)
# -----------------------------
def corner_risk(warped_bgr: np.ndarray):
    """
    Corner "risk" heuristic using sharpness in corner ROIs.
    This is NOT microscopic grading—just a photo-based flag.
    """
    h, w = warped_bgr.shape[:2]
    s = int(min(h, w) * 0.10)

    rois = {
        "TL": warped_bgr[0:s, 0:s],
        "TR": warped_bgr[0:s, w - s:w],
        "BL": warped_bgr[h - s:h, 0:s],
        "BR": warped_bgr[h - s:h, w - s:w],
    }

    vals = {}
    for k, roi in rois.items():
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        vals[k] = float(cv2.Laplacian(g, cv2.CV_64F).var())

    med = float(np.median(list(vals.values())))
    out = {}
    for k, v in vals.items():
        if med <= 1e-6:
            out[k] = "UNCERTAIN (photo too soft)"
        elif v < med * 0.65:
            out[k] = "DEFECT/RISK"
        elif v < med * 0.85:
            out[k] = "Slight risk"
        else:
            out[k] = "Looks sharp"
    return out


# -----------------------------
# Main analysis function
# -----------------------------
def analyze(img_pil: Image.Image):
    img_rgb = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    quad, edge_map = find_outer_card_quad(img_bgr)

    # If we fail outer detection, return edge-map for debugging
    if quad is None:
        edge_rgb = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
        return (
            "Insufficient photo quality: could not detect the full OUTER card outline.\n"
            "- Make sure all 4 card corners are visible (not cropped).\n"
            "- Reduce glare; keep the card taking up more of the frame.\n"
            "- Avoid heavy tilt.\n",
            Image.fromarray(edge_rgb),
        )

    warped, _ = warp_card(img_bgr, quad)
    if warped is None:
        edge_rgb = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
        return (
            "Insufficient photo quality: detected outline but could not warp reliably.",
            Image.fromarray(edge_rgb),
        )

    inner, th = find_inner_frame_rect(warped)
    h, w = warped.shape[:2]

    overlay = warped.copy()

    if inner is None:
        th_rgb = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
        return (
            "Insufficient evidence: could not reliably detect INNER printed frame.\n"
            "Debug image shows what the frame detector sees (white = detected structures).\n",
            Image.fromarray(th_rgb),
        )

    # Gaps from OUTER card edge to INNER frame
    gL = int(x1)
    gR = int(w - x2)
    gT = int(y1)
    gB = int(h - y2)

    # sanity checks
    if min(gL, gR, gT, gB) < 5:
        cv2.putText(overlay, "FRAME TOO CLOSE/UNCLEAR", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        return (
            "Insufficient evidence: detected frame but measurement is unreliable (frame too close/unclear).",
            Image.fromarray(overlay_rgb),
        )

    lr, tb, worst, within_55, bucket = classify_centering(gL, gR, gT, gB)

    lr_split = (round(gL / (gL + gR) * 100, 1), round(gR / (gL + gR) * 100, 1))
    tb_split = (round(gT / (gT + gB) * 100, 1), round(gB / (gT + gB) * 100, 1))

    corners = corner_risk(warped)

    # Overlay: outer boundary is the warped image bounds; draw inner frame
    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)          # outer
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)              # inner frame
    cv2.putText(overlay, bucket, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    msg = (
        "Centering (outer edge → inner frame):\n"
        f"- L/R split: {lr_split[0]} / {lr_split[1]}   (max-side={lr:.3f})\n"
        f"- T/B split: {tb_split[0]} / {tb_split[1]}   (max-side={tb:.3f})\n"
        f"- Within 55/45: {'YES' if within_55 else 'NO'}\n"
        f"- Bucket: {bucket}\n\n"
        f"Corners (photo-based risk): {corners}\n\n"
        "Trust check:\n"
        "- Green box = outer card bounds after flatten.\n"
        "- Blue box = detected inner frame.\n"
        "- If blue box is wrong, treat as UNCERTAIN and request a better straight-on photo."
    )

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return msg, Image.fromarray(overlay_rgb)


# -----------------------------
# Gradio App
# -----------------------------
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="pil", label="Upload FRONT screenshot/photo"),
    outputs=[
        gr.Textbox(label="Result"),
        gr.Image(type="pil", label="Overlay / Debug (what was measured)"),
    ],
    title="Card Pre-Grade (Front Only) — Centering 55/45 + Corner Risk",
    description="Uploads a front image, flattens perspective, detects outer card + inner printed frame, measures 55/45, flags corner risk. If detection is uncertain, it will say so.",
)

if __name__ == "__main__":
    demo.launch()