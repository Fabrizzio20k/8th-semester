import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import torch

# Configuración
VIDEO_PATH = 'a.mp4'
MODEL_NAME = 'yolo11x.pt'
CONF = 0.40
IOU = 0.50
MIN_HITS = 6
MAX_AGE = 30
MATCHING_THRESHOLD = 90
KALMAN_Q = 0.01
KALMAN_R = 5
KALMAN_P0 = 10
USE_CONFIDENCE = True
CONF_THRESHOLD = 0.35
VEHICLE_CLASSES = [2, 5, 7, 3]


class KalmanTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_Q
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_R
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * KALMAN_P0

    def predict(self):
        return self.kf.predict()[:2].flatten()

    def update(self, measurement):
        return self.kf.correct(measurement)[:2].flatten()


class Track:
    def __init__(self, detection, track_id, bbox, confidence):
        self.id = track_id
        self.tracker = KalmanTracker()
        self.tracker.kf.statePost = np.array(
            [[detection[0]], [detection[1]], [0], [0]], np.float32)
        self.age = 0
        self.total_visible_count = 1
        self.consecutive_invisible_count = 0
        self.last_pos = np.array(detection, dtype=np.float32)
        self.bbox = bbox
        self.confidence_history = [confidence]
        self.avg_confidence = confidence
        self.position_history = [detection]


def calculate_iou(box1, box2):
    if box1 is None or box2 is None:
        return 0.0
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


# Inicializar
model = YOLO(MODEL_NAME)
model.to('cuda' if torch.cuda.is_available() else 'cpu')
cap = cv2.VideoCapture(VIDEO_PATH)
tracks = []
next_id = 0
counted_vehicles = set()

print("🎬 Presiona 'q' para salir\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (frame.shape[1], frame.shape[0]))

    # Detección
    results = model(frame, classes=VEHICLE_CLASSES, verbose=False,
                    conf=CONF, iou=IOU, agnostic_nms=True)
    detections = []
    for box in results[0].boxes:
        x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
        y_center = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
        bbox = (int(box.xyxy[0][0]), int(box.xyxy[0][1]),
                int(box.xyxy[0][2]), int(box.xyxy[0][3]))
        confidence = float(box.conf[0])
        detections.append(([x_center, y_center], bbox, confidence))

    # Predicción
    for track in tracks:
        track.last_pos = track.tracker.predict()

    # Asociación
    if len(tracks) > 0 and len(detections) > 0:
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, (det_pos, det_bbox, det_conf) in enumerate(detections):
                distance = np.linalg.norm(track.last_pos - np.array(det_pos))
                iou_val = calculate_iou(track.bbox, det_bbox)
                cost_matrix[i, j] = distance - (iou_val * 50)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_tracks = set()
        matched_detections = set()
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < MATCHING_THRESHOLD:
                det_pos, det_bbox, det_conf = detections[j]
                measurement = np.array(
                    [[det_pos[0]], [det_pos[1]]], np.float32)
                tracks[i].last_pos = tracks[i].tracker.update(measurement)
                tracks[i].age += 1
                tracks[i].total_visible_count += 1
                tracks[i].consecutive_invisible_count = 0
                tracks[i].bbox = det_bbox
                tracks[i].confidence_history.append(det_conf)
                tracks[i].avg_confidence = np.mean(
                    tracks[i].confidence_history[-10:])
                tracks[i].position_history.append(det_pos)
                if len(tracks[i].position_history) > 15:
                    tracks[i].position_history.pop(0)
                matched_tracks.add(i)
                matched_detections.add(j)
        for i, track in enumerate(tracks):
            if i not in matched_tracks:
                track.consecutive_invisible_count += 1
                track.age += 1
        for j, (det_pos, det_bbox, det_conf) in enumerate(detections):
            if j not in matched_detections:
                tracks.append(Track(det_pos, next_id, det_bbox, det_conf))
                next_id += 1
    elif len(detections) > 0:
        for det_pos, det_bbox, det_conf in detections:
            tracks.append(Track(det_pos, next_id, det_bbox, det_conf))
            next_id += 1
    else:
        for track in tracks:
            track.consecutive_invisible_count += 1
            track.age += 1

    tracks = [t for t in tracks if t.consecutive_invisible_count < MAX_AGE]

    # Conteo
    for track in tracks:
        if track.total_visible_count >= MIN_HITS:
            if USE_CONFIDENCE and track.avg_confidence >= CONF_THRESHOLD:
                counted_vehicles.add(track.id)
            elif not USE_CONFIDENCE:
                counted_vehicles.add(track.id)

    # Visualización
    for track in tracks:
        color = (0, 255, 0) if track.id in counted_vehicles else (0, 255, 255)
        if track.bbox:
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        x, y = int(track.last_pos[0]), int(track.last_pos[1])
        cv2.putText(frame, f"ID:{track.id}", (x-20, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (x, y), 4, color, -1)
        if len(track.position_history) > 1:
            points = np.array(track.position_history, dtype=np.int32)
            cv2.polylines(frame, [points], False, color, 2)

    # Stats
    cv2.putText(frame, f"Contados: {len(counted_vehicles)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Vehicle Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Total: {len(counted_vehicles)} vehículos")
