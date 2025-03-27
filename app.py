import cv2
import torch
import numpy as np
from ultralytics import YOLO
import argparse
import time

# Default video paths
default_video_paths = [
    r"C:\Users\RAMESWAR BISOYI\Downloads\cctv1.mp4",
    r"C:\Users\RAMESWAR BISOYI\Downloads\cctv2.mp4"
]

# Parse arguments
parser = argparse.ArgumentParser(description='Head detection in videos')
parser.add_argument('--model', type=str, default='head_detection/scut_head/weights/best.pt', 
                    help='Path to model weights')
parser.add_argument('--videos', nargs='+', default=default_video_paths, 
                    help='Paths to video files')
parser.add_argument('--conf', type=float, default=0.3, 
                    help='Detection confidence threshold')
parser.add_argument('--show-fps', action='store_true', 
                    help='Display FPS counter')
args = parser.parse_args()

# Load the custom head detection model
print(f"Loading model from {args.model}")
head_model = YOLO(args.model)

# Open video captures
caps = [cv2.VideoCapture(video) for video in args.videos if cv2.VideoCapture(video).isOpened()]
if not caps:
    raise ValueError("No valid video sources found!")

print(f"Opened {len(caps)} video sources")

# Function to detect heads
def detect_heads(results, conf_threshold=0.3):
    heads = []
    for r in results:
        if hasattr(r, 'boxes'):
            for box in r.boxes:
                if box.conf[0] > conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0])
                    
                    # Calculate head center and radius
                    head_x = (x1 + x2) // 2
                    head_y = (y1 + y2) // 2
                    radius = max(5, min(x2 - x1, y2 - y1) // 3)
                    
                    heads.append((head_x, head_y, radius, confidence))
    return heads

# Initialize FPS tracking
prev_time = time.time()
fps_list = []

while True:
    active_cameras = 0
    
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            continue
            
        active_cameras += 1

        # Update FPS calculation
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time
        
        if elapsed_time > 0:
            fps = 1.0 / elapsed_time
            fps_list.append(fps)
            if len(fps_list) > 10:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)
        else:
            avg_fps = 0

        # Run inference with the head detection model
        results = head_model.predict(frame, conf=args.conf)  # Use `predict()`
        
        # Get head locations
        heads = detect_heads(results, conf_threshold=args.conf)
        head_count = len(heads)

        # Create a display frame
        display_frame = frame.copy()

        # Draw green circles on detected heads
        for head_x, head_y, head_r, conf in heads:
            cv2.circle(display_frame, (head_x, head_y), head_r, (0, 255, 0), 2)

        # Create a semi-transparent overlay for text
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (30, 30), (300, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        # Display head count
        head_count_text = f"Head Count: {head_count}"
        cv2.putText(display_frame, head_count_text, (50, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show FPS if requested
        if args.show_fps:
            fps_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(display_frame, fps_text, (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(f"Camera {i+1} - Head Detection", display_frame)
    
    # If no active cameras, break
    if active_cameras == 0:
        print("No active video sources remaining")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video captures
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
