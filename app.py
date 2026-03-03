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
    Inner frame detection via strong straight lines:
    - edge detect
    - Hough lines
    - pick 2 vertical + 2 horizontal lines that form a rectangle
    Returns: (rect, debug_img)
      rect = (x1,y1,x2,y2) or None
      debug_img = RGB image showing detected lines
    """
    h, w = warped_bgr.shape[:2]

    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # edges (less speckle than adaptive threshold)
    edges = cv2.Canny(gray, 60, 160)

    # close small gaps in the frame lines
    k = max(3, min(h, w) // 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Hough line transform
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=120,
        minLineLength=int(min(h, w) * 0.35),
        maxLineGap=int(min(h, w) * 0.03)
    )

    debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if lines is None:
        debug_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        return None, debug_rgb

    # classify lines as vertical-ish or horizontal-ish
    vertical = []
    horizontal = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length < 50:
            continue
        # angle test
        if abs(dx) < abs(dy) * 0.35:
            # vertical-ish: store average x
            vertical.append((int((x1 + x2) / 2), length, (x1, y1, x2, y2)))
        elif abs(dy) < abs(dx) * 0.35:
            # horizontal-ish: store average y
            horizontal.append((int((y1 + y2) / 2), length, (x1, y1, x2, y2)))

    # draw debug lines
    for _, _, (x1, y1, x2, y2) in vertical:
        cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for _, _, (x1, y1, x2, y2) in horizontal:
        cv2.line(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)

    margin = int(min(h, w) * 0.06)

    # keep only interior-ish candidates (ignore outer border)
    vertical = [v for v in vertical if margin < v[0] < (w - margin)]
    horizontal = [u for u in horizontal if margin < u[0] < (h - margin)]

    if len(vertical) < 2 or len(horizontal) < 2:
        debug_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        return None, debug_rgb

    # pick best left/right: far apart, strong lines
    vertical.sort(key=lambda t: t[0])
    left = max(vertical[: max(1, len(vertical)//2)], key=lambda t: t[1])
    right = max(vertical[len(vertical)//2 :], key=lambda t: t[1])

    # pick best top/bottom
    horizontal.sort(key=lambda t: t[0])
    top = max(horizontal[: max(1, len(horizontal)//2)], key=lambda t: t[1])
    bottom = max(horizontal[len(horizontal)//2 :], key=lambda t: t[1])

    xL, xR = left[0], right[0]
    yT, yB = top[0], bottom[0]

    # sanity checks
    if xR - xL < int(w * 0.40) or yB - yT < int(h * 0.40):
        debug_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        return None, debug_rgb

    rect = (xL, yT, xR, yB)

    # draw the chosen rectangle in yellow
    cv2.rectangle(debug, (xL, yT), (xR, yB), (0, 255, 255), 3)
    debug_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
    return rect, debug_rgb


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

    inner, dbg = find_inner_frame_rect(warped)
    h, w = warped.shape[:2]

    overlay = warped.copy()

    if inner is None:
    return (
        "Insufficient evidence: could not reliably detect INNER printed frame.\n"
        "Debug image shows detected lines (green=vertical, blue=horizontal).\n",
        Image.fromarray(dbg),
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