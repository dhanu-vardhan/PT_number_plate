import cv2
import numpy as np
import sys
import os
from ultralytics import YOLO

# Add SORT path
sys.path.append('/Users/anthapuvivekanandareddy/Desktop/Vehicle_helmet_detection_cursor/sort')
from sort import Sort

# Import custom utility functions
import util
from util import get_car, read_license_plate, write_csv

# Paths
license_plate_model_path = '/Users/anthapuvivekanandareddy/Desktop/Vehicle_helmet_detection_cursor/numberplate_PT/license_plate_detector.pt'
video_path = '/Users/anthapuvivekanandareddy/Desktop/Vehicle_helmet_detection_cursor/Data_Input/traffic_1.mp4'

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO(license_plate_model_path)

# Setup tracker
mot_tracker = Sort()

# Vehicle classes from COCO
vehicle_classes = [2, 3, 5, 7]  # Car, motorcycle, bus, truck

# Result dictionary
results = {}

# Open video
cap = cv2.VideoCapture(video_path)
frame_nmr = -1
ret = True

while ret:
    ret, frame = cap.read()
    if not ret:
        break

    frame_nmr += 1
    results[frame_nmr] = {}

    # --- Vehicle detection ---
    vehicle_detections = coco_model(frame)[0]
    detections_ = []
    for det in vehicle_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) in vehicle_classes:
            detections_.append([x1, y1, x2, y2, score])

    # --- Tracking ---
    track_ids = mot_tracker.update(np.asarray(detections_))

    # --- License plate detection ---
    license_plates = license_plate_detector(frame)[0]
    for lp in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = lp

        # Get vehicle associated with this license plate
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)

        if car_id != -1:
            # Crop license plate region
            lp_crop = frame[int(y1):int(y2), int(x1):int(x2)]

            # Preprocess for OCR
            lp_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
            _, lp_thresh = cv2.threshold(lp_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # Read plate text
            lp_text, lp_text_score = read_license_plate(lp_thresh)

            if lp_text is not None:
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': lp_text,
                        'bbox_score': score,
                        'text_score': lp_text_score
                    }
                }

        # Draw license plate bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(frame, 'Plate', (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw vehicle bounding boxes
    for track in track_ids:
        x1, y1, x2, y2, track_id = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID:{int(track_id)}', (int(x1), int(y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Vehicle & License Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Save results
write_csv(results, 'output_results.csv')