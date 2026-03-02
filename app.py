import gradio as gr
import cv2
import numpy as np
from PIL import Image

# ---- config ----
CLEAR_PASS_MAX = 0.54      # comfortably inside 55/45
BORDERLINE_LOW = 0.54
BORDERLINE_HIGH = 0.56     # above this is a clear fail zone
PSA_LIMIT = 0.55

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
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
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))
    scale = out_w / maxW
    warped = cv2.resize(warped, (out_w, int(maxH * scale)))
    return warped, rect

def find_outer_card_quad(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, None, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts[:15]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.10 * (img_bgr.shape[0]*img_bgr.shape[1]):
            return approx.reshape(4,2)
    return None

def find_inner_frame_rect(warped_bgr):
    """
    MVP heuristic:
    - find strong rectangular contour inside the card after warping
    - chooses the largest 4-point contour not touching outer edge
    This will be refined per set (Optic/Prizm frames vary).
    """
    h, w = warped_bgr.shape[:2]
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 50, 140)

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0

    margin = int(min(h, w) * 0.03)  # must be inside card by this margin

    for c in cnts:
        area = cv2.contourArea(c)
        if area < (h*w)*0.05:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) != 4:
            continue
        pts = approx.reshape(4,2)
        x, y, ww, hh = cv2.boundingRect(pts)

        # reject things that hug the outer edge (likely the card itself)
        if x < margin or y < margin or (x+ww) > (w-margin) or (y+hh) > (h-margin):
            continue

        if area > best_area:
            best_area = area
            best = pts

    if best is None:
        return None

    # return axis-aligned rect from the 4 points (post-warp mostly axis aligned)
    x, y, ww, hh = cv2.boundingRect(best)
    return (x, y, x+ww, y+hh)

def classify_centering(gL, gR, gT, gB):
    lr = max(gL, gR) / (gL + gR)
    tb = max(gT, gB) / (gT + gB)
    worst = max(lr, tb)

    if worst <= CLEAR_PASS_MAX:
        bucket = "CLEAR PASS"
    elif worst <= BORDERLINE_HIGH:
        bucket = "BORDERLINE (request better photo; if not possible = uncertain)"
    else:
        bucket = "FAIL"

    within_55 = (lr <= PSA_LIMIT) and (tb <= PSA_LIMIT)
    return lr, tb, bucket, within_55

def corner_risk(warped_bgr):
    # simple sharpness heuristic per corner ROI
    h, w = warped_bgr.shape[:2]
    s = int(min(h,w)*0.10)
    corners = {
        "TL": warped_bgr[0:s, 0:s],
        "TR": warped_bgr[0:s, w-s:w],
        "BL": warped_bgr[h-s:h, 0:s],
        "BR": warped_bgr[h-s:h, w-s:w],
    }
    vals = {}
    for k, roi in corners.items():
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3,3), 0)
        vals[k] = float(cv2.Laplacian(g, cv2.CV_64F).var())

    med = np.median(list(vals.values()))
    out = {}
    for k, v in vals.items():
        if v < med*0.65:
            out[k] = "Defect/Risk"
        elif v < med*0.85:
            out[k] = "Slight risk"
        else:
            out[k] = "Looks sharp"
    return out

def analyze(img_pil: Image.Image):
    img_rgb = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    quad = find_outer_card_quad(img_bgr)
    if quad is None:
        return "Insufficient photo quality: could not detect full card edges.", None

    warped, outer_rect_pts = warp_card(img_bgr, quad)

    inner = find_inner_frame_rect(warped)
    if inner is None:
        return "Insufficient evidence: could not reliably detect inner printed frame.", None

    h, w = warped.shape[:2]
    (x1, y1, x2, y2) = inner

    # gaps: outer edge to inner frame
    gL = x1
    gR = (w - x2)
    gT = y1
    gB = (h - y2)

    if min(gL, gR, gT, gB) < 5:
        return "Insufficient evidence: inner frame too close/unclear for reliable measurement.", None

    lr, tb, bucket, within_55 = classify_centering(gL, gR, gT, gB)

    lr_split = (round(gL/(gL+gR)*100, 1), round(gR/(gL+gR)*100, 1))
    tb_split = (round(gT/(gT+gB)*100, 1), round(gB/(gT+gB)*100, 1))

    corners = corner_risk(warped)

    # overlay for trust
    overlay = warped.copy()
    cv2.rectangle(overlay, (0,0), (w-1, h-1), (0,255,0), 2)               # outer
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (255,0,0), 2)                # inner frame
    cv2.putText(overlay, bucket, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    msg = (
        f"Centering:\n"
        f"- L/R split: {lr_split[0]} / {lr_split[1]}  (max-side={lr:.3f})\n"
        f"- T/B split: {tb_split[0]} / {tb_split[1]}  (max-side={tb:.3f})\n"
        f"- 55/45 within tolerance: {'YES' if within_55 else 'NO'}\n"
        f"- Bucket: {bucket}\n\n"
        f"Corners (photo-based): {corners}\n\n"
        f"Note: If overlay boxes don’t match the actual card/frame, treat as UNCERTAIN."
    )

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return msg, Image.fromarray(overlay_rgb)

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="pil", label="Upload FRONT screenshot"),
    outputs=[gr.Textbox(label="Result"), gr.Image(type="pil", label="Overlay (what was measured)")],
    title="Card Pre-Grade (Front Only) — Centering (55/45) + Corners",
    description="Uploads a front screenshot, flattens perspective, detects outer card + inner frame, measures centering, flags corner risk."
)

if __name__ == "__main__":
    demo.launch()