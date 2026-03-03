import gradio as gr
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Config
# -----------------------------
PSA_LIMIT = 0.55
CLEAR_PASS_MAX = 0.54
BORDERLINE_HIGH = 0.56


# -----------------------------
# Geometry helpers
# -----------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def warp_card(img_bgr: np.ndarray, quad: np.ndarray, out_w: int = 900):
    rect = order_points(quad.astype("float32"))
    (tl, tr, br, bl) = rect

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)

    maxW = int(max(wA, wB))
    maxH = int(max(hA, hB))
    if maxW < 80 or maxH < 80:
        return None

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))

    scale = out_w / maxW
    warped = cv2.resize(warped, (out_w, max(1, int(maxH * scale))))
    return warped


# -----------------------------
# OUTER quad detection (robust + avoids screenshot frame)
# -----------------------------
def _score_candidate_quad(pts: np.ndarray, area: float, H: int, W: int) -> float:
    pts_o = order_points(pts)
    (tl, tr, br, bl) = pts_o

    ww = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    hh = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    if ww < 80 or hh < 80:
        return -1.0

    aspect = min(ww, hh) / max(ww, hh)  # ~0.714
    if not (0.62 <= aspect <= 0.80):
        return -1.0

    xs = pts[:, 0]
    ys = pts[:, 1]
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())

    margin = min(minx, miny, (W - 1) - maxx, (H - 1) - maxy)
    margin_norm = margin / float(min(H, W))

    # Reject if it hugs screenshot edges
    if margin_norm < 0.01:
        return -1.0

    bbox_area = (maxx - minx) * (maxy - miny)
    if bbox_area > 0.97 * (H * W):
        return -1.0

    aspect_target = 2.5 / 3.5
    aspect_penalty = abs(aspect - aspect_target)
    margin_bonus = min(1.0, margin_norm / 0.08)

    return float(area * (1.0 - aspect_penalty) * (0.50 + 0.50 * margin_bonus))


def _find_outer_quad_from_edges(edges: np.ndarray) -> np.ndarray | None:
    H, W = edges.shape[:2]
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best_quad = None
    best_score = -1.0

    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:60]:
        area = cv2.contourArea(c)
        if area < 0.03 * (H * W):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) >= 4:
            hull = cv2.convexHull(approx)
            approx2 = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            if len(approx2) == 4:
                pts = approx2.reshape(4, 2).astype("float32")
            else:
                rect = cv2.minAreaRect(c)
                pts = cv2.boxPoints(rect).astype("float32")
        else:
            rect = cv2.minAreaRect(c)
            pts = cv2.boxPoints(rect).astype("float32")

        score = _score_candidate_quad(pts, area, H, W)
        if score > best_score:
            best_score = score
            best_quad = pts

    return best_quad


def find_outer_card_quad(img_bgr: np.ndarray):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 130)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    quad = _find_outer_quad_from_edges(edges.copy())
    if quad is not None:
        return quad, edges

    # fallback: mask bottom (stands sometimes)
    edges2 = edges.copy()
    cut = int(H * 0.78)
    edges2[cut:, :] = 0
    quad2 = _find_outer_quad_from_edges(edges2)
    if quad2 is not None:
        return quad2, edges2

    return None, edges


# -----------------------------
# INNER boundary detection (Sobel energy + geometry)
# -----------------------------
def _smooth1d(a: np.ndarray, k: int) -> np.ndarray:
    k = max(9, k | 1)
    ker = np.ones(k, dtype=np.float32) / k
    return np.convolve(a.astype(np.float32), ker, mode="same")


def find_inner_boundary_rect(warped_bgr: np.ndarray):
    """
    Returns:
      rect: (x1,y1,x2,y2) candidate
      confident: bool
      overlay_rgb: debug overlay on the WARPED image (not edge map)
    """
    h, w = warped_bgr.shape[:2]
    overlay = warped_bgr.copy()

    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    # Contrast normalize a bit (helps on holo glare)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (3, 3), 0)

    # Sobel energy: strong, less speckly than Canny for “where are the rails”
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    sx = np.abs(sx)
    sy = np.abs(sy)

    # Bands
    yA, yB = int(h * 0.18), int(h * 0.78)   # mid-band for L/R
    xC1, xC2 = int(w * 0.25), int(w * 0.75) # central strip for T/B

    # L/R profile from vertical energy
    col_energy = sx[yA:yB, :].mean(axis=0)
    col_energy = _smooth1d(col_energy, k=max(31, w // 45))

    # T profile from horizontal energy
    row_energy = sy[:, xC1:xC2].mean(axis=1)
    row_energy = _smooth1d(row_energy, k=max(31, h // 45))

    # Search windows near edges
    x_lo, x_hi = int(w * 0.02), int(w * 0.14)
    y_lo, y_hi = int(h * 0.02), int(h * 0.14)

    # L/R peaks in the edge bands
    winL = col_energy[x_lo:x_hi]
    winR = col_energy[w - x_hi:w - x_lo]  # not reversed yet
    if winL.size < 10 or winR.size < 10:
        return None, False, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    x1 = x_lo + int(np.argmax(winL))
    x2 = (w - x_lo) - int(np.argmax(winR[::-1]))

    # Top peak
    winT = row_energy[y_lo:y_hi]
    if winT.size < 10:
        return None, False, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    y1 = y_lo + int(np.argmax(winT))

    # Geometry bottom
    if x2 <= x1 + int(w * 0.30):
        return None, False, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    outer_aspect = h / float(w)
    inner_w = float(x2 - x1)
    expected_h = float(np.clip(inner_w * outer_aspect, h * 0.55, h * 0.92))
    y2 = int(y1 + expected_h)

    # Clamp away from extreme bottom zone
    if y2 > int(h * 0.90):
        y2 = int(h * 0.90)

    if y2 <= y1 + int(h * 0.35):
        return None, False, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Validate bottom line exists: check horizontal energy around y2 across inner width
    band = 3
    yb0 = max(0, y2 - band)
    yb1 = min(h, y2 + band + 1)
    xb0 = int(x1 + inner_w * 0.10)
    xb1 = int(x2 - inner_w * 0.10)
    xb0 = max(0, xb0)
    xb1 = min(w, xb1)

    confident = True
    if xb1 - xb0 < 40:
        confident = False
    else:
        bottom_strength = sy[yb0:yb1, xb0:xb1].mean()
        top_strength = sy[max(0, y1 - band):min(h, y1 + band + 1), xb0:xb1].mean()

        # Much looser validation than before (prevents “UNCERTAIN spam”)
        # If bottom is tiny vs top, treat as uncertain.
        if bottom_strength < max(6.0, 0.25 * top_strength):
            confident = False

    # Inset pad (avoid halo)
    pad = int(min(h, w) * 0.006)
    x1p = int(x1 + pad)
    y1p = int(y1 + pad)
    x2p = int(x2 - pad)
    y2p = int(y2 - pad)

    if x2p <= x1p or y2p <= y1p:
        return None, False, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Draw candidate inner box:
    # - Blue if confident
    # - Orange if uncertain (still show what it tried)
    color = (255, 0, 0) if confident else (0, 165, 255)  # BGR
    cv2.rectangle(overlay, (x1p, y1p), (x2p, y2p), color, 2)

    # Also draw rails we found (thin)
    cv2.line(overlay, (x1p, 0), (x1p, h - 1), color, 1)
    cv2.line(overlay, (x2p, 0), (x2p, h - 1), color, 1)
    cv2.line(overlay, (0, y1p), (w - 1, y1p), color, 1)
    cv2.line(overlay, (0, y2p), (w - 1, y2p), color, 1)

    return (x1p, y1p, x2p, y2p), confident, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


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
    try:
        img_rgb = np.array(img_pil.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        quad, edge_map = find_outer_card_quad(img_bgr)
        if quad is None:
            edge_rgb = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
            return (
                "Insufficient photo quality: could not detect OUTER card outline.",
                Image.fromarray(edge_rgb),
            )

        warped = warp_card(img_bgr, quad)
        if warped is None:
            edge_rgb = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
            return (
                "Insufficient photo quality: outline detected but warp failed.",
                Image.fromarray(edge_rgb),
            )

        h, w = warped.shape[:2]
        overlay = warped.copy()
        cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)  # outer green

        inner, confident, inner_overlay_rgb = find_inner_boundary_rect(warped)

        # If inner function returns a full overlay already, use it (it draws candidate box)
        overlay_rgb = inner_overlay_rgb

        if inner is None:
            msg = (
                "UNCERTAIN: could not detect INNER boundary.\n"
                "Try a straighter, less-glare photo if possible."
            )
            return msg, Image.fromarray(overlay_rgb)

        x1, y1, x2, y2 = inner

        if not confident:
            msg = (
                "UNCERTAIN: inner boundary is not reliable in this photo.\n"
                "Orange box = what it *thinks* the inner boundary might be.\n"
                "Do NOT trust centering from this result."
            )
            return msg, Image.fromarray(overlay_rgb)

        gL = int(x1)
        gR = int(w - x2)
        gT = int(y1)
        gB = int(h - y2)

        if min(gL, gR, gT, gB) < 5:
            msg = "UNCERTAIN: detected boundary but gaps are too small/unclear."
            return msg, Image.fromarray(overlay_rgb)

        lr, tb, worst, within_55, bucket = classify_centering(gL, gR, gT, gB)
        lr_split = (round(gL / (gL + gR) * 100, 1), round(gR / (gL + gR) * 100, 1))
        tb_split = (round(gT / (gT + gB) * 100, 1), round(gB / (gT + gB) * 100, 1))

        corners = corner_risk(warped)

        msg = (
            "Centering (outer edge → inner boundary):\n"
            f"- L/R split: {lr_split[0]} / {lr_split[1]}   (max-side={lr:.3f})\n"
            f"- T/B split: {tb_split[0]} / {tb_split[1]}   (max-side={tb:.3f})\n"
            f"- Within 55/45: {'YES' if within_55 else 'NO'}\n"
            f"- Bucket: {bucket}\n\n"
            f"Corners (photo-based risk): {corners}\n\n"
            "Trust check:\n"
            "- Green box = outer card bounds after flatten.\n"
            "- Blue box = confident inner boundary.\n"
            "- Orange box = uncertain inner boundary (do not trust)."
        )

        return msg, Image.fromarray(overlay_rgb)

    except Exception as e:
        return f"Error: {type(e).__name__}: {e}", Image.new("RGB", (10, 10), (0, 0, 0))


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
    description="Uploads a front image, flattens perspective, detects outer card + inner boundary, measures 55/45, flags corner risk. If detection is uncertain, it will say so.",
)

if __name__ == "__main__":
    demo.launch()