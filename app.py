import gradio as gr
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Config (your goals)
# -----------------------------
PSA_LIMIT = 0.55          # 55/45 rule (front)
CLEAR_PASS_MAX = 0.54     # "centering eliminated" bucket
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
        return None

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))

    # normalize width for stable downstream heuristics
    scale = out_w / maxW
    warped = cv2.resize(warped, (out_w, max(1, int(maxH * scale))))
    return warped


# -----------------------------
# Detection: OUTER card quad (robust)
# -----------------------------
def _score_quad(pts: np.ndarray, area: float, target_aspect: float, lo: float, hi: float) -> float:
    """
    Score candidate quad:
    - large area is good
    - aspect close to target is good
    """
    rect_o = order_points(pts)
    (tl, tr, br, bl) = rect_o

    ww = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    hh = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    if ww < 80 or hh < 80:
        return -1.0

    aspect = min(ww, hh) / max(ww, hh)  # ~0.714 for card (w/h)
    if not (lo <= aspect <= hi):
        return -1.0

    # closeness score
    aspect_penalty = abs(aspect - target_aspect)
    return area * (1.0 - aspect_penalty)


def _find_outer_quad_from_edges(edges: np.ndarray, H: int, W: int) -> np.ndarray | None:
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    target = 2.5 / 3.5  # ~0.714
    lo, hi = 0.62, 0.80

    best_pts = None
    best_score = -1.0

    # Try top contours, not just biggest
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:40]:
        area = cv2.contourArea(c)
        if area < 0.03 * (H * W):
            continue

        peri = cv2.arcLength(c, True)

        # Prefer true 4-corner approximation if possible
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            quad_bonus = 1.15
        else:
            rect = cv2.minAreaRect(c)
            pts = cv2.boxPoints(rect).astype("float32")
            quad_bonus = 1.0

        score = _score_quad(pts, area, target, lo, hi)
        if score > best_score:
            best_score = score
            best_pts = pts
            best_bonus = quad_bonus

    if best_pts is None:
        return None

    return best_pts


def find_outer_card_quad(img_bgr: np.ndarray):
    """
    Robust outer detection:
    - build edges
    - attempt detection on full image
    - if not found, retry with bottom-masked edges (helps when stands block)
    Returns: (quad or None, edge_map_for_debug)
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 130)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    quad = _find_outer_quad_from_edges(edges.copy(), H, W)
    if quad is not None:
        return quad, edges

    # Retry with bottom masked (stand-safe), but only as fallback
    edges2 = edges.copy()
    cut = int(H * 0.78)
    edges2[cut:, :] = 0
    quad2 = _find_outer_quad_from_edges(edges2, H, W)
    if quad2 is not None:
        return quad2, edges2

    return None, edges


# -----------------------------
# Detection: INNER boundary (constrained)
# -----------------------------
def find_inner_boundary_rect(warped_bgr: np.ndarray):
    """
    Inner boundary strategy (stable):
    1) Detect left/right rails using edge-density in mid-band
    2) Detect top rail in top band
    3) Predict bottom using aspect and inner width
    4) Snap bottom ONLY near prediction (prevents grabbing player leg/nameplate)
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

    # --- Build profiles ---
    # L/R: use mid height only (avoid logos/nameplate)
    yA, yB = int(h * 0.18), int(h * 0.78)
    col = edges[yA:yB, :].sum(axis=0) / 255.0
    col_s = smooth1d(col, k=max(31, w // 45))

    # T/B: use central width to reduce side artifacts
    xC1, xC2 = int(w * 0.25), int(w * 0.75)
    row = edges[:, xC1:xC2].sum(axis=1) / 255.0
    row_s = smooth1d(row, k=max(31, h // 45))

    # Search windows near edges (tight)
    x_lo, x_hi = int(w * 0.02), int(w * 0.14)
    y_lo, y_hi = int(h * 0.02), int(h * 0.14)

    # --- Left boundary: strongest gradient in left band ---
    winL = col_s[x_lo:x_hi]
    if winL.size < 10:
        dbg_rgb = cv2.cvtColor(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        return None, dbg_rgb
    gradL = np.abs(np.diff(winL))
    x1 = x_lo + int(np.argmax(gradL))

    # --- Right boundary: strongest gradient in right band (mirrored) ---
    winR = col_s[w - x_hi:w - x_lo][::-1]
    if winR.size < 10:
        dbg_rgb = cv2.cvtColor(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        return None, dbg_rgb
    gradR = np.abs(np.diff(winR))
    x2 = (w - x_lo) - int(np.argmax(gradR))

    if x2 <= x1 + int(w * 0.30):
        dbg_rgb = cv2.cvtColor(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        return None, dbg_rgb

    # --- Top boundary: strongest gradient in top band ---
    winT = row_s[y_lo:y_hi]
    if winT.size < 10:
        dbg_rgb = cv2.cvtColor(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        return None, dbg_rgb
    gradT = np.abs(np.diff(winT))
    y1 = y_lo + int(np.argmax(gradT))

    # --- Predict bottom from aspect ---
    outer_aspect = h / float(w)  # height/width
    inner_w = float(x2 - x1)
    expected_h = inner_w * outer_aspect

    # keep prediction in sane bounds
    expected_h = float(np.clip(expected_h, h * 0.55, h * 0.92))
    y2_pred = int(y1 + expected_h)

    # --- Bottom boundary: snap near prediction only ---
    # Search within ±5% of height around prediction, but also avoid nameplate zone by not going too low.
    radius = int(h * 0.05)
    loB = max(0, y2_pred - radius)
    hiB = min(h - 2, y2_pred + radius)

    # Hard stop: don't let bottom be in the bottom 12% (usually nameplate/label/stand noise zone)
    hiB = min(hiB, int(h * 0.88))

    if hiB - loB < 15:
        dbg_rgb = cv2.cvtColor(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        return None, dbg_rgb

    winB = row_s[loB:hiB]
    gradB = np.abs(np.diff(winB))
    if gradB.size < 5:
        dbg_rgb = cv2.cvtColor(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        return None, dbg_rgb

    y2 = loB + int(np.argmax(gradB))

    # Aspect consistency (inner should match outer orientation/aspect)
    inner_h = float(y2 - y1)
    if inner_h <= 0:
        dbg_rgb = cv2.cvtColor(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        return None, dbg_rgb

    aspect_inner = inner_h / inner_w
    if abs(aspect_inner - outer_aspect) > 0.10:
        dbg_rgb = cv2.cvtColor(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        return None, dbg_rgb

    # Small inset to avoid glow/halo grabbing
    pad = int(min(h, w) * 0.006)
    x1 += pad
    y1 += pad
    x2 -= pad
    y2 -= pad

    if x2 <= x1 or y2 <= y1:
        dbg_rgb = cv2.cvtColor(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        return None, dbg_rgb

    # Debug overlay on edges
    dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(dbg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)  # yellow inner
    dbg_rgb = cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)

    return (int(x1), int(y1), int(x2), int(y2)), dbg_rgb


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
                "Insufficient photo quality: could not detect OUTER card outline.\n"
                "- Make sure all 4 corners are visible.\n"
                "- Reduce glare.\n"
                "- Avoid heavy tilt.\n",
                Image.fromarray(edge_rgb),
            )

        warped = warp_card(img_bgr, quad)
        if warped is None:
            edge_rgb = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
            return ("Insufficient photo quality: outline detected but warp failed.", Image.fromarray(edge_rgb))

        inner, dbg_rgb = find_inner_boundary_rect(warped)

        h, w = warped.shape[:2]
        overlay = warped.copy()

        # outer box
        cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)

        if inner is None:
            return (
                "UNCERTAIN: could not reliably detect INNER boundary.\n"
                "Overlay shows edge-detected inner attempt (yellow). If wrong/missing, request a better straight-on photo.",
                Image.fromarray(dbg_rgb),
            )

        x1, y1, x2, y2 = inner
        gL = int(x1)
        gR = int(w - x2)
        gT = int(y1)
        gB = int(h - y2)

        # sanity
        if min(gL, gR, gT, gB) < 5:
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            return ("UNCERTAIN: detected boundary but gaps are too small/unclear.", Image.fromarray(overlay_rgb))

        lr, tb, worst, within_55, bucket = classify_centering(gL, gR, gT, gB)

        lr_split = (round(gL / (gL + gR) * 100, 1), round(gR / (gL + gR) * 100, 1))
        tb_split = (round(gT / (gT + gB) * 100, 1), round(gB / (gT + gB) * 100, 1))

        corners = corner_risk(warped)

        # inner box (blue)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(overlay, bucket, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

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
        # Always return 2 outputs
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