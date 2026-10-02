import cv2
import numpy as np


def get_color(idx):
    """Generate a distinct, bright BGR color based on integer ID using golden ratio HSV palette."""
    idx = int(idx)
    golden_ratio_conjugate = 0.618033988749895
    h = (idx * golden_ratio_conjugate) % 1.0
    s = 0.85
    v = 0.95
    
    # Convert HSV (0..1) to BGR (0..255)
    hsv = np.uint8([[[h * 180, s * 255, v * 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def draw_bounding_box(frame, track_id, bbox, confidence=None, text_prefix="ID"):
    """
    Draw a clean, modern bounding box with a filled header badge for the ID label.
    bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    
    color = get_color(track_id)
    line_thickness = max(2, int(round(min(w, h) / 400.)))
    
    # Draw main rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
    
    # Label text
    if confidence is not None:
        label = f"{text_prefix} #{track_id} ({confidence:.2f})"
    else:
        label = f"{text_prefix} #{track_id}"
        
    font_scale = max(0.4, min(w, h) / 1000.)
    font_thickness = max(1, int(font_scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    # Draw background box for text badge
    badge_y1 = max(0, y1 - text_height - 8)
    badge_y2 = y1
    badge_x2 = min(w, x1 + text_width + 10)
    
    cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
    
    # Text inside badge (white or dark depending on color brightness)
    text_color = (255, 255, 255)
    cv2.putText(
        frame,
        label,
        (x1 + 5, badge_y2 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        font_thickness,
        lineType=cv2.LINE_AA
    )


def create_side_by_side_grid(frames):
    """Stack a list of frames horizontally or in a grid for multi-camera view."""
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
        
    # Resize frames to match height of first frame
    target_h = frames[0].shape[0]
    resized_frames = []
    
    for f in frames:
        h, w = f.shape[:2]
        if h != target_h:
            target_w = int(w * (target_h / float(h)))
            f_res = cv2.resize(f, (target_w, target_h))
        else:
            f_res = f
        resized_frames.append(f_res)
        
    # Combine horizontally
    combined = np.hstack(resized_frames)
    return combined
