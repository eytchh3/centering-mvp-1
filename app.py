import gradio as gr
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Config
# -----------------------------
PSA_LIMIT = 0.55          # PSA 10 centering eligibility (front)
CLEAR_PASS_MAX = 0.54     # comfortable pass bucket
BORDERLINE_HIGH = 0.56    # borderline bucket upper bound


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

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))

    # normalize width for stable downstream heuristics
    scale = out_w / maxW
    warped = cv2.resize(warped, (out_w, max(1, int(maxH * scale))))

    return warped, rect


# -----------------------------
# Detection: OUTER card quad
# -----------------------------
def find_outer_card_quad(img_bgr: np.ndarray):
    """
    Outer outline detection:
    - edge map
    - mask bottom region to avoid stands
    - try top contours, prefer those that approximate to 4 points and match card aspect
    Returns: (quad_points or None, edge_map)
    """
    H, W = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 130)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    # Mask bottom (stand/background clutter) without changing coordinates
    cut = int(H * 0.78)
    edges[cut:, :] = 0

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, edges

    target = 2.5 / 3.5  # ~0.714 (w/h)
    lo, hi = 0.62, 0.80

    best = None
    best_score = -1.0

    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:40]:
        area = cv2.contourArea(c)
        if area < 0.03 * (H * W):
            continue

        peri = cv2.arcLength(c, True)

        # Try true quad first
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            quad_bonus = 1.15
        else:
            rect = cv2.minAreaRect(c)
            pts = cv2.boxPoints(rect).astype("float32")
            quad_bonus = 1.0

        rect_o = order_points(pts)
        (tl, tr, br, bl) = rect_o
        ww = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
        hh = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
        if ww < 50 or hh < 50:
            continue

        aspect = min(ww, hh) / max(ww, hh)
        if not (lo <= aspect <= hi):
            continue

        score = area * (1.0 - abs(aspect - target)) * quad_bonus
        if score > best_score:
            best_score = score
            best = pts

    return best, edges


# -----------------------------
# Detection: INNER "frame boundary" via scan-in
# -----------------------------
def find_inner_frame_rect(warped_bgr: np.ndarray):
    """
    Detect inner boundary by scanning in from each edge using edge-density profiles.
    Returns: (rect or None, debug_rgb)
      rect = (x1,y1,x2,y2) in warped coords
    """
    h, w = warped_bgr.shape[:2]
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 60, 160)

    def smooth1d(a, k=25):
        k = max(5, k | 1)  # odd
        pad = k // 2
        ap = np.pad(a.astype(np.float32), (pad, pad), mode="edge")
        kernel = np.ones(k, dtype=np.float32) / k
        return np.convolve(ap, kernel, mode="valid")

    # ---- Profiles ----
    # L/R: use mid-height band to avoid logos/nameplate
    yA, yB = int(h * 0.18), int(h * 0.78)
    col = edges[yA:yB, :].sum(axis=0) / 255.0

    # T/B: use central vertical strip; also compute full profile to find nameplate band peak
    xC1, xC2 = int(w * 0.25), int(w * 0.75)
    row_full = edges[:, xC1:xC2].sum(axis=1) / 255.0
    row_full_s = smooth1d(row_full, k=max(25, h // 80))

    # Only search for the nameplate band in the bottom ~28% of the card
    search_start = int(h * 0.72)
    band = row_full_s[search_start:]
    peak_idx = int(np.argmax(band)) + search_start
    # Cut above the peak, but keep enough bottom content
    y_cut = peak_idx - int(h * 0.05)             # cut a bit above the nameplate peak
    y_cut = int(np.clip(y_cut, int(h * 0.75), int(h * 0.92)))

    row = edges[:y_cut, xC1:xC2].sum(axis=1) / 255.0

    col_s = smooth1d(col, k=max(25, w // 60))
    row_s = smooth1d(row, k=max(25, h // 60))

    # Search bands near edges (tightened for Optic-style borders)
    x_lo, x_hi = int(w * 0.02), int(w * 0.14)
    y_lo, y_hi = int(h * 0.02), int(h * 0.14)

    def first_or_grad(window, lo_offset, reverse_map=None):
        """
        window: 1D array in search band (already sliced).
        lo_offset: index offset to convert window index -> absolute index in that dimension.
        reverse_map: optional callable that maps window-index to absolute index (for right/bottom scanning).
        """
        if window.size < 10:
            return None

        base = np.median(window[: max(5, window.size // 5)])
        thr = max(base * 1.8, base + 20)

        for i, v in enumerate(window):
            if v >= thr:
                idx = i
                return reverse_map(idx) if reverse_map else (lo_offset + idx)

        grad = np.abs(np.diff(window))
        if grad.size == 0:
            return None
        idx = int(np.argmax(grad))
        return reverse_map(idx) if reverse_map else (lo_offset + idx)

    # Left
    winL = col_s[x_lo:x_hi]
    x1 = first_or_grad(winL, x_lo)

    # Right (mirror the band)
    winR = col_s[w - x_hi : w - x_lo][::-1]
    x2 = first_or_grad(
        winR,
        0,
        reverse_map=lambda idx: (w - x_lo) - idx
    )

    # If L/R rails aren't found, stop
    dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.line(dbg, (0, int(y_cut)), (w - 1, int(y_cut)), (255, 255, 0), 2)
    
    if x1 is None or x2 is None:
        dbg_rgb = cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)
        return None, dbg_rgb
    
    # Estimate expected inner height from measured inner width
    outer_aspect = h / float(w)
    inner_w = float(x2 - x1)
    expected_inner_h = inner_w * outer_aspect
    expected_inner_h = float(np.clip(expected_inner_h, h * 0.55, h * 0.92))
    
    # Top
    winT = row_s[y_lo:y_hi]
    y1 = first_or_grad(winT, y_lo)

    # Bottom: row_s is length y_cut (not full h)
    row_h = len(row_s)
    winB = row_s[row_h - y_hi : row_h - y_lo][::-1]
    y2_local = first_or_grad(
        winB,
        0,
        reverse_map=lambda idx: (row_h - y_lo) - idx
    )
    y2 = y2_local  # in the row_s coordinate space (0..y_cut)

    # Debug image
    dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # draw y_cut line so we can see where bottom was excluded
    cv2.line(dbg, (0, int(y_cut)), (w - 1, int(y_cut)), (255, 255, 0), 2)  # cyan

    if None in (x1, x2, y1, y2):
        dbg_rgb = cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)
        return None, dbg_rgb

    # sanity check on size (use y_cut for bottom since y2 is inside 0..y_cut)
    if (x2 - x1) < int(w * 0.45) or (y2 - y1) < int(h * 0.45):
        dbg_rgb = cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)
        return None, dbg_rgb

    # extra guardrail: bottom boundary should not be absurdly far from edge
    # (if it is, it's likely nameplate/artwork edge)
    bottom_gap = y_cut - y2
    if bottom_gap > int(h * 0.10):
        return None, dbg_rgb

    # shrink inner boundary slightly inward to avoid glow/halo edges
    pad = int(min(h, w) * 0.006)
    x1 += pad; y1 += pad; x2 -= pad; y2 -= pad

    if x2 <= x1 or y2 <= y1:
        dbg_rgb = cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)
        return None, dbg_rgb

    # draw chosen rect (yellow)
    cv2.rectangle(dbg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
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
    """Photo-based corner risk flag (not microscopic grading)."""
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
                "Insufficient photo quality: could not detect the full OUTER card outline.\n"
                "- Make sure all 4 card corners are visible.\n"
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

        overlay = warped.copy()
        h, w = warped.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)

        if inner is None:
            # return the debug image produced by the inner detector
            return (
                "UNCERTAIN: could not reliably detect INNER frame boundary.\n"
                "Overlay shows outer warp; debug image shows edge map + cyan y_cut line + (if found) yellow inner box.",
                Image.fromarray(dbg),
            )

        x1, y1, x2, y2 = inner
        gL = int(x1)
        gR = int(w - x2)
        gT = int(y1)
        gB = int(h - y2)

        # sanity
        if min(gL, gR, gT, gB) < 5:
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            return (
                "UNCERTAIN: detected inner boundary but measurement is unreliable (frame too close/unclear).",
                Image.fromarray(overlay_rgb),
            )

        lr, tb, worst, within_55, bucket = classify_centering(gL, gR, gT, gB)

        lr_split = (round(gL / (gL + gR) * 100, 1), round(gR / (gL + gR) * 100, 1))
        tb_split = (round(gT / (gT + gB) * 100, 1), round(gB / (gT + gB) * 100, 1))

        corners = corner_risk(warped)

        # draw inner on overlay (blue)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
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
        # Always return 2 outputs so Gradio doesn't throw "Error" tiles
        return f"Error: {type(e).__name__}: {e}", Image.new("RGB", (10, 10), (0, 0, 0))


# -----------------------------
# Gradio App
# -----------------------------
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="pil", label="Upload FRONT screenshot/photo"),
    outputs=[
        gr.Textbox(label="Result"),
        gr.Image(type="pil", label="Overlay / Debug"),
    ],
    title="Card Pre-Grade (Front Only) — Centering 55/45 + Corner Risk",
    description="Uploads a front image, flattens perspective, detects outer card + inner boundary, measures 55/45, flags corner risk. If detection is uncertain, it will say so.",
)

if __name__ == "__main__":
    demo.launch()