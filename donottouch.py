#no changing here stuff claude
import os
import cv2
import urllib.request
from ultralytics import YOLO

# --- Download YOLO11s if needed ---
model_url = 'https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11s.pt'
model_path = 'yolo11s.pt'
if not os.path.exists(model_path):
    print("⬇️ Downloading YOLO11s...")
    urllib.request.urlretrieve(model_url, model_path)

# Load model
model = YOLO(model_path)

# Input/output paths
video_path = 'samplevideo.mp4'
output_path = 'output_final_directional.mp4'

# Setup video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# --- Line: horizontal across middle of screen ---
line_y = h // 2

# Tracking
last_y = {}
counted_inbound = set()
counted_outbound = set()
inbound = 0   # right side, going up (↑)
outbound = 0  # left side, going down (↓)

CLASS_NAMES = {2: "Car", 7: "Truck"}

# --- Tracking with persist=True ---
for result in model.track(source=video_path, stream=True, persist=True, classes=[2, 7]):
    img = result.orig_img.copy()

    # Draw horizontal midline
    cv2.line(img, (0, line_y), (w, line_y), (255, 0, 0), 2)

    if result.boxes and result.boxes.id is not None:
        for box, cls, track_id in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.id):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            tid = int(track_id)
            cls_id = int(cls)
            label = CLASS_NAMES.get(cls_id, "Vehicle")

            # Use horizontal position (cx) to determine left vs right
            if tid in last_y:
                prev_y = last_y[tid]

                # 🚗 Left side = outbound (down)
                if cx < w // 2 and prev_y < line_y and cy >= line_y and tid not in counted_outbound:
                    outbound += 1
                    counted_outbound.add(tid)

                # 🚗 Right side = inbound (up)
                elif cx >= w // 2 and prev_y > line_y and cy <= line_y and tid not in counted_inbound:
                    inbound += 1
                    counted_inbound.add(tid)

            last_y[tid] = cy

            # Draw
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} #{tid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

    # Show counts
    cv2.putText(img, f"INBOUND: {inbound}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(img, f"OUTBOUND: {outbound}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    out.write(img)

out.release()
print(f"\n✅ Done! Saved final output to: {output_path}")
