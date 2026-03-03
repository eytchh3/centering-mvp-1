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

    if maxW < 80 or maxH < 80:
        return None

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))

    # normalize width for stable downstream heuristics
    scale = out_w / maxW
    warped = cv2.resize(warped, (out_w, max(1, int(maxH * scale))))

    return warped


# -----------------------------
# Detection: OUTER card quad
# -----------------------------
def _score_candidate_quad(pts: np.ndarray, area: float, H: int, W: int) -> float:
    """
    Score:
      - large area good
      - aspect close to card good
      - NOT hugging image borders good (prevents selecting the screenshot frame)
      - prefer convex/quaddy shapes
    """
    pts_o = order_points(pts)
    (tl, tr, br, bl) = pts_o

    ww = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    hh = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    if ww < 80 or hh < 80:
        return -1.0

    # Card aspect ratio: width/height ~ 2.5/3.5 = 0.714
    aspect = min(ww, hh) / max(ww, hh)  # ~0.714 regardless of orientation
    if not (0.62 <= aspect <= 0.80):
        return -1.0

    # Bounding box of candidate in original image
    xs = pts[:, 0]
    ys = pts[:, 1]
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())

    # If candidate hugs the image edges, it's probably the screenshot frame/background
    margin = min(minx, miny, (W - 1) - maxx, (H - 1) - maxy)
    margin_norm = margin / float(min(H, W))

    # Strong penalty if margin is tiny (touching edges)
    if margin_norm < 0.01:
        return -1.0

    # Also penalize if bbox is basically the entire image
    bbox_area = (maxx - minx) * (maxy - miny)
    if bbox_area > 0.97 * (H * W):
        return -1.0

    # Score components
    aspect_target = 2.5 / 3.5
    aspect_penalty = abs(aspect - aspect_target)
    margin_bonus = min(1.0, margin_norm / 0.08)  # reaches 1.0 around 8% margin

    score = area * (1.0 - aspect_penalty) * (0.50 + 0.50 * margin_bonus)
    return float(score)


def _find_outer_quad_from_edges(edges: np.ndarray) -> np.ndarray | None:
    H, W = edges.shape[:2]
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best_quad = None
    best_score = -1.0

    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:50]:
        area = cv2.contourArea(c)
        if area < 0.03 * (H * W):
            continue

        peri = cv2.arcLength(c, True)

        # Prefer true quad approximation when possible
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
    """
    Two-pass outer detection:
      Pass 1: full image
      Pass 2: bottom-masked (helps when stand blocks edges)
    Returns: (quad or None, edge_map_for_debug)
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 130)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    quad = _find_outer_quad_from_edges(edges.copy())
    if quad is not None:
        return quad, edges

    edges2 = edges.copy()
    cut = int(H * 0.78)
    edges2[cut:, :] = 0
    quad2 = _find_outer_quad_from_edges(edges2)
    if quad2 is not None:
        return quad2, edges2

    return None, edges


# -----------------------------
# Detection: INNER boundary (constrained, stable)
# -----------------------------
def find_inner_boundary_rect(warped_bgr: np.ndarray):
    """
    Stable inner boundary:
      - detect L/R rails from mid-band
      - detect top rail near top
      - predict bottom using outer aspect + inner width
      - snap bottom ONLY near predicted location (prevents grabbing player leg/nameplate)
    Returns: (rect or None, debug_rgb)
    """
    h, w = warped_bgr.shape[:2]
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)

    def smooth1d(a, k=31):
        k = max(9, k | 1)
        pad = k // 2
        ap = np.pad(a.astype(np.float32), (pad, pad), mode="edge")
        ker = np.ones(k, dtype=np.float32) / k
        return np.convolve(ap, ker, mode="valid")

    # Profiles
    yA, yB = int(h * 0.18), int(h * 0.78)
    col = edges[yA:yB, :].sum(axis=0) / 255.0
    col_s = smooth1d(col, k=max(31, w // 45))

    xC1, xC2 = int(w * 0.25), int(w * 0.75)
    row = edges[:, xC1:xC2].sum(axis=1) / 255.0
    row_s = smooth1d(row, k=max(31, h // 45))

    x_lo, x_hi = int(w * 0.02), int(w * 0.14)
    y_lo, y_hi = int(h * 0.02), int(h * 0.14)

    # L/R from gradients
    winL = col_s[x_lo:x_hi]
    winR = col_s[w - x_hi:w - x_lo][::-1]
    if winL.size < 10 or winR.size < 10:
        dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return None, cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)

    x1 = x_lo + int(np.argmax(np.abs(np.diff(winL))))
    x2 = (w - x_lo) - int(np.argmax(np.abs(np.diff(winR))))

    if x2 <= x1 + int(w * 0.30):
        dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return None, cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)

    # Top from gradients
    winT = row_s[y_lo:y_hi]
    if winT.size < 10:
        dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return None, cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)

    y1 = y_lo + int(np.argmax(np.abs(np.diff(winT))))

    # Predict bottom and snap locally
    outer_aspect = h / float(w)
    inner_w = float(x2 - x1)
    expected_h = float(np.clip(inner_w * outer_aspect, h * 0.55, h * 0.92))
    y2_pred = int(y1 + expected_h)

    radius = int(h * 0.05)
    loB = max(0, y2_pred - radius)
    hiB = min(h - 2, y2_pred + radius)

    # Avoid absolute bottom (nameplate/stands)
    hiB = min(hiB, int(h * 0.90))

    if hiB - loB < 15:
        dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return None, cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)

    winB = row_s[loB:hiB]
    gradB = np.abs(np.diff(winB))
    if gradB.size < 5:
        dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return None, cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)

    y2 = loB + int(np.argmax(gradB))

    # Aspect consistency
    inner_h = float(y2 - y1)
    if inner_h <= 0:
        dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return None, cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)

    aspect_inner = inner_h / inner_w
    if abs(aspect_inner - outer_aspect) > 0.10:
        dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return None, cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)

    # Small inset to avoid halo edges
    pad = int(min(h, w) * 0.006)
    x1 += pad; y1 += pad; x2 -= pad; y2 -= pad

    if x2 <= x1 or y2 <= y1:
        dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return None, cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)

    dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(dbg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
    return (int(x1), int(y1), int(x2), int(y2)), cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)


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

        inner, dbg_rgb = find_inner_boundary_rect(warped)

        h, w = warped.shape[:2]
        overlay = warped.copy()
        cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)  # outer

        if inner is None:
            return (
                "UNCERTAIN: could not reliably detect INNER boundary.",
                Image.fromarray(dbg_rgb),
            )

        x1, y1, x2, y2 = inner
        gL = int(x1)
        gR = int(w - x2)
        gT = int(y1)
        gB = int(h - y2)

        if min(gL, gR, gT, gB) < 5:
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            return (
                "UNCERTAIN: detected boundary but gaps are too small/unclear.",
                Image.fromarray(overlay_rgb),
            )

        lr, tb, worst, within_55, bucket = classify_centering(gL, gR, gT, gB)
        lr_split = (round(gL / (gL + gR) * 100, 1), round(gR / (gL + gR) * 100, 1))
        tb_split = (round(gT / (gT + gB) * 100, 1), round(gB / (gT + gB) * 100, 1))

        corners = corner_risk(warped)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)  # inner
        cv2.putText(overlay, bucket, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        msg = (
            "Centering (outer edge → inner boundary):\n"
            f"- L/R split: {lr_split[0]} / {lr_split[1]}   (max-side={lr:.3f})\n"
            f"- T/B split: {tb_split[0]} / {tb_split[1]}   (max-side={tb:.3f})\n"
            f"- Within 55/45: {'YES' if within_55 else 'NO'}\n"
            f"- Bucket: {bucket}\n\n"
            f"Corners (photo-based risk): {corners}\n\n"
            "Trust check:\n"
            "- Green box = outer card bounds after flatten.\n"
            "- Blue box = detected inner boundary.\n"
            "- If blue box is wrong, treat as UNCERTAIN and request a better straight-on photo."
        )

        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
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