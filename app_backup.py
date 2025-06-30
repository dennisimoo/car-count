from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import logging
import threading
import queue
import time
import requests
from collections import defaultdict
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
frame_buffer = queue.Queue(maxsize=1)  # Smaller buffer for less lag
processing = False
class_counts = defaultdict(int)
crossed_ids = set()
source_url = "https://wzmedia.dot.ca.gov/D12/NB55DELMAR.stream/playlist.m3u8"
source_thread = None
line_y = 360
inbound_count = 0
outbound_count = 0
last_y = {}
counted_inbound = set()
counted_outbound = set()
frame_times = []
processing_start_time = time.time()

def load_model():
    """Load YOLO model"""
    global model
    try:
        logger.info("Loading YOLO model...")
        model = YOLO('yolov8n.pt')
        logger.info("YOLO model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return False

def get_caltrans_cameras():
    """Return available Caltrans camera feeds"""
    return {
        "D6 Kamm Ave": "https://wzmedia.dot.ca.gov/D12/NB55DELMAR.stream/playlist.m3u8"
    }

def get_hls_frames(url):
    """Generate frames from HLS stream"""
    logger.info(f"Connecting to HLS stream: {url}")
    
    try:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            logger.info("Successfully opened HLS stream")
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame, reconnecting...")
                    cap.release()
                    cap = cv2.VideoCapture(url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not cap.isOpened():
                        break
                    continue
                
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
                
                yield frame
            cap.release()
        else:
            raise Exception("Failed to open HLS stream")
            
    except Exception as e:
        logger.error(f"HLS stream error: {e}")
        # Generate error frames
        while True:
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Stream Connection Failed", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(error_frame, "Retrying connection...", (50, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            yield error_frame
            time.sleep(5)

def process_frame(frame):
    """Process frame with YOLO detection and counting - matching original style"""
    global model, line_y, last_y, counted_inbound, counted_outbound, inbound_count, outbound_count, frame_times
    
    frame_start_time = time.time()
    
    if model is None:
        cv2.putText(frame, "YOLO model not loaded", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    line_y = h // 2
    
    # Draw horizontal midline (blue like original)
    cv2.line(frame, (0, line_y), (w, line_y), (255, 0, 0), 2)
    
    try:
        # Run YOLO detection with tracking (cars and trucks only like original)
        results = model.track(frame, persist=True, classes=[2, 7], conf=0.3, verbose=False)
        
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            
            for box, class_idx, track_id in zip(boxes, class_indices, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tid = int(track_id)
                cls_id = int(class_idx)
                
                # Use same class names as original
                CLASS_NAMES = {2: "Car", 7: "Truck"}
                label = CLASS_NAMES.get(cls_id, "Vehicle")
                
                # Tracking logic - exactly like original
                if tid in last_y:
                    prev_y = last_y[tid]
                    
                    # Left side = outbound (down)
                    if cx < w // 2 and prev_y < line_y and cy >= line_y and tid not in counted_outbound:
                        outbound_count += 1
                        counted_outbound.add(tid)
                        logger.info(f"OUTBOUND: {label} #{tid}")
                    
                    # Right side = inbound (up)
                    elif cx >= w // 2 and prev_y > line_y and cy <= line_y and tid not in counted_inbound:
                        inbound_count += 1
                        counted_inbound.add(tid)
                        logger.info(f"INBOUND: {label} #{tid}")
                
                last_y[tid] = cy
                
                # Draw exactly like original
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} #{tid}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                
    except Exception as e:
        logger.error(f"Detection error: {e}")
        cv2.putText(frame, "Detection Error", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show counts exactly like original
    cv2.putText(frame, f"INBOUND: {inbound_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"OUTBOUND: {outbound_count}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Track frame processing time for lag calculation
    frame_end_time = time.time()
    processing_time = (frame_end_time - frame_start_time) * 1000  # ms
    frame_times.append(processing_time)
    if len(frame_times) > 30:
        frame_times.pop(0)  # Keep only last 30 frames
    
    return frame

def video_processor():
    """Main video processing loop"""
    global processing, frame_buffer, source_url, model
    
    processing = True
    
    if model is None:
        logger.info("Loading YOLO model...")
        if not load_model():
            logger.error("Failed to load YOLO model")
    
    try:
        for frame in get_hls_frames(source_url):
            if not processing:
                break
                
            # Process frame with YOLO
            processed_frame = process_frame(frame)
            
            # Add to buffer
            if frame_buffer.full():
                try:
                    frame_buffer.get_nowait()
                except queue.Empty:
                    pass
            frame_buffer.put(processed_frame)
            
    except Exception as e:
        logger.error(f"Video processing error: {e}")
    finally:
        processing = False

def generate_frames():
    """Generate frames for video stream"""
    last_frame = None
    
    while True:
        try:
            frame = frame_buffer.get(timeout=0.1)
            last_frame = frame
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                   
        except queue.Empty:
            if last_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', last_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                # Create waiting frame
                waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(waiting_frame, "Connecting to camera...", (120, 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', waiting_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
        time.sleep(0.033)  # Maintain 30 FPS regardless of processing speed

# Routes
@app.route('/')
def index():
    camera_options = get_caltrans_cameras()
    return render_template('simple.html', camera_options=camera_options)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_camera', methods=['POST'])
def set_camera():
    global processing, source_thread, source_url, class_counts, crossed_ids, inbound_count, outbound_count
    
    data = request.get_json()
    camera_url = data.get('camera_url')
    if not camera_url:
        return jsonify({"status": "error", "message": "No camera URL provided"}), 400
    
    logger.info(f"Setting camera to: {camera_url}")
    
    # Stop current processing
    if processing:
        processing = False
        if source_thread and source_thread.is_alive():
            source_thread.join(timeout=1.0)
    
    # Reset counts
    source_url = camera_url
    class_counts.clear()
    crossed_ids.clear()
    inbound_count = 0
    outbound_count = 0
    
    # Start new processing thread
    source_thread = threading.Thread(target=video_processor)
    source_thread.daemon = True
    source_thread.start()
    
    return jsonify({"status": "success", "message": "Camera set successfully"})

@app.route('/reset_counts', methods=['POST'])
def reset_counts():
    global class_counts, crossed_ids, inbound_count, outbound_count, last_y, counted_inbound, counted_outbound
    class_counts.clear()
    crossed_ids.clear()
    last_y.clear()
    counted_inbound.clear()
    counted_outbound.clear()
    inbound_count = 0
    outbound_count = 0
    return jsonify({"status": "counts reset"})

@app.route('/stats')
def stats():
    # Calculate lag percentage
    lag_percent = 0
    if len(frame_times) > 5:
        avg_processing_time = sum(frame_times) / len(frame_times)
        expected_frame_time = 33.33  # 30 FPS = 33.33ms per frame
        lag_percent = max(0, ((avg_processing_time - expected_frame_time) / expected_frame_time) * 100)
    
    return jsonify({
        "inbound": inbound_count,
        "outbound": outbound_count,
        "total": inbound_count + outbound_count,
        "class_counts": dict(class_counts),
        "lag_percent": lag_percent,
        "avg_processing_time": sum(frame_times) / len(frame_times) if frame_times else 0
    })

if __name__ == '__main__':
    load_model()
    
    # Start default video processing
    source_thread = threading.Thread(target=video_processor)
    source_thread.daemon = True
    source_thread.start()
    
    app.run(debug=True, host='0.0.0.0', port=5050)