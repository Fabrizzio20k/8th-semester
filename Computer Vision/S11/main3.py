import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import torch
import json
import os
from pathlib import Path
from datetime import datetime
import time

# ================== CONFIGURACIÓN ==================
INPUT_FOLDER = 'kfilter'
OUTPUT_FOLDER = 'resultados_tracking'
MODEL_NAME = 'yolo11l.pt'
VEHICLE_CLASSES = [2, 5, 7, 3]
MIN_HITS = 4
MAX_AGE = 20
MATCHING_THRESHOLD = 90
RESIZE_FACTOR = 0.5

# Evaluación
EVAL_RESIZE = 0.4
CONF_VALUES = [0.40, 0.50, 0.60]
IOU_VALUES = [0.40, 0.50]
MAX_SECONDS_EVAL = 20  # Reducido para ir más rápido


class KalmanTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10

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


def evaluate_config(video_path, model, conf, iou, max_frames):
    """Evalúa una configuración específica"""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = int(width * EVAL_RESIZE)
    new_height = int(height * EVAL_RESIZE)

    tracks = []
    next_id = 0
    counted_vehicles = set()
    frame_count = 0
    total_detections = 0
    tracks_created = 0
    confidence_sum = 0

    SKIP_FRAMES = 3

    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue

        frame = cv2.resize(frame, (new_width, new_height),
                           interpolation=cv2.INTER_NEAREST)
        results = model(frame, classes=VEHICLE_CLASSES,
                        verbose=False, conf=conf, iou=iou, agnostic_nms=True)
        detections = []

        for box in results[0].boxes:
            x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            y_center = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
            bbox = (int(box.xyxy[0][0]), int(box.xyxy[0][1]),
                    int(box.xyxy[0][2]), int(box.xyxy[0][3]))
            confidence = float(box.conf[0])
            detections.append(([x_center, y_center], bbox, confidence))
            confidence_sum += confidence

        total_detections += len(detections)

        for track in tracks:
            track.last_pos = track.tracker.predict()

        if len(tracks) > 0 and len(detections) > 0:
            cost_matrix = np.zeros((len(tracks), len(detections)))
            for i, track in enumerate(tracks):
                for j, (det_pos, det_bbox, det_conf) in enumerate(detections):
                    distance = np.linalg.norm(
                        track.last_pos - np.array(det_pos))
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
                    tracks_created += 1
        elif len(detections) > 0:
            for det_pos, det_bbox, det_conf in detections:
                tracks.append(Track(det_pos, next_id, det_bbox, det_conf))
                next_id += 1
                tracks_created += 1
        else:
            for track in tracks:
                track.consecutive_invisible_count += 1
                track.age += 1

        tracks = [t for t in tracks if t.consecutive_invisible_count < MAX_AGE]

        for track in tracks:
            if track.total_visible_count >= MIN_HITS and track.avg_confidence >= 0.35:
                counted_vehicles.add(track.id)

    cap.release()
    avg_confidence = confidence_sum / total_detections if total_detections > 0 else 0
    efficiency = tracks_created / \
        len(counted_vehicles) if len(counted_vehicles) > 0 else 999

    return {
        'conf': conf,
        'iou': iou,
        'vehicles': len(counted_vehicles),
        'tracks': tracks_created,
        'efficiency': round(efficiency, 2),
        'avg_conf': round(avg_confidence, 3)
    }


def find_best_config(video_path, model):
    """Encuentra la mejor configuración para un video específico"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    max_frames = fps * MAX_SECONDS_EVAL

    results = []
    for conf in CONF_VALUES:
        for iou in IOU_VALUES:
            result = evaluate_config(video_path, model, conf, iou, max_frames)
            results.append(result)

    # Mejor configuración: menor eficiencia (menos tracks por vehículo)
    sorted_results = sorted(results, key=lambda x: (
        x['efficiency'], -x['vehicles']))
    return sorted_results[0], sorted_results


def process_video_with_config(video_path, model, conf, iou, output_folder, video_num, total_videos):
    """Procesa un video completo con configuración específica"""
    video_name = Path(video_path).stem
    output_video_path = os.path.join(
        output_folder, f"{video_name}_tracked.mp4")
    output_json_path = os.path.join(
        output_folder, f"{video_name}_metrics.json")

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    new_width = int(width * RESIZE_FACTOR)
    new_height = int(height * RESIZE_FACTOR)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc,
                          fps, (new_width, new_height))

    tracks = []
    next_id = 0
    counted_vehicles = set()
    frame_count = 0
    total_detections = 0
    confidence_sum = 0

    start_time = time.time()
    last_update = start_time

    print(f"\n   {'─'*60}")
    print(f"   Procesando frames: 0/{total_frames} (0.0%)", end='', flush=True)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        frame = cv2.resize(frame, (new_width, new_height))

        # Actualizar progreso cada 0.5 segundos
        current_time = time.time()
        if current_time - last_update > 0.5:
            progress = (frame_count / total_frames) * 100
            elapsed = current_time - start_time
            fps_processing = frame_count / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_count) / \
                fps_processing if fps_processing > 0 else 0

            print(f"\r   Procesando frames: {frame_count}/{total_frames} ({progress:.1f}%) | "
                  f"FPS: {fps_processing:.1f} | ETA: {eta:.0f}s", end='', flush=True)
            last_update = current_time

        # Detección
        results = model(frame, classes=VEHICLE_CLASSES,
                        verbose=False, conf=conf, iou=iou, agnostic_nms=True)
        detections = []

        for box in results[0].boxes:
            x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            y_center = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
            bbox = (int(box.xyxy[0][0]), int(box.xyxy[0][1]),
                    int(box.xyxy[0][2]), int(box.xyxy[0][3]))
            confidence = float(box.conf[0])
            detections.append(([x_center, y_center], bbox, confidence))
            confidence_sum += confidence

        total_detections += len(detections)

        # Predicción
        for track in tracks:
            track.last_pos = track.tracker.predict()

        # Asociación
        if len(tracks) > 0 and len(detections) > 0:
            cost_matrix = np.zeros((len(tracks), len(detections)))
            for i, track in enumerate(tracks):
                for j, (det_pos, det_bbox, det_conf) in enumerate(detections):
                    distance = np.linalg.norm(
                        track.last_pos - np.array(det_pos))
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
            if track.total_visible_count >= MIN_HITS and track.avg_confidence >= 0.35:
                counted_vehicles.add(track.id)

        # Visualización
        for track in tracks:
            if track.id in counted_vehicles:
                color = (0, 255, 0)
                if track.bbox:
                    x1, y1, x2, y2 = track.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

                x, y = int(track.last_pos[0]), int(track.last_pos[1])
                cv2.putText(frame, f"{track.id}", (x-10, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Info
        cv2.putText(frame, f"Contados: {len(counted_vehicles)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

    print(
        f"\r   Procesando frames: {total_frames}/{total_frames} (100.0%) ✓                    ")

    cap.release()
    out.release()

    processing_time = time.time() - start_time
    avg_conf = confidence_sum / total_detections if total_detections > 0 else 0

    # JSON completo
    metrics = {
        'video_info': {
            'filename': video_name,
            'original_resolution': f"{width}x{height}",
            'processed_resolution': f"{new_width}x{new_height}",
            'fps': fps,
            'total_frames': total_frames,
            'duration_seconds': round(total_frames / fps, 2)
        },
        'processing_info': {
            'processing_time_seconds': round(processing_time, 2),
            'processing_fps': round(total_frames / processing_time, 2),
            'date_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'optimal_configuration': {
            'model': MODEL_NAME,
            'conf_threshold': conf,
            'iou_threshold': iou,
            'min_hits': MIN_HITS,
            'max_age': MAX_AGE,
            'matching_threshold': MATCHING_THRESHOLD
        },
        'tracking_results': {
            'vehicles_counted': len(counted_vehicles),
            'unique_vehicle_ids': list(sorted(counted_vehicles)),
            'total_detections': total_detections,
            'avg_confidence': round(avg_conf, 3)
        }
    }

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("\n" + "="*70)
    print("🚗 SISTEMA DE TRACKING DE VEHÍCULOS")
    print("="*70)

    # Cargar modelo
    print("\n🔄 Cargando modelo YOLO...")
    model = YOLO(MODEL_NAME)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'GPU' if torch.cuda.is_available() else 'CPU'
    print(f"✅ Modelo cargado en: {device}")

    # Obtener videos
    video_files = sorted(
        [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.mp4')])
    if not video_files:
        print(f"❌ No se encontraron videos en '{INPUT_FOLDER}'")
        return

    total_videos = len(video_files)

    print(f"\n📁 Videos encontrados: {total_videos}")
    print("="*70)

    all_metrics = []
    total_start_time = time.time()

    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(INPUT_FOLDER, video_file)
        video_name = Path(video_file).stem

        print(f"\n{'█'*70}")
        print(f"📹 VIDEO [{i}/{total_videos}]: {video_name}")
        print(f"{'█'*70}")

        # FASE 1: Evaluar parámetros óptimos
        print(f"\n🔬 FASE 1: Evaluando configuraciones óptimas...")
        print(
            f"   Analizando primeros {MAX_SECONDS_EVAL} segundos del video...")

        eval_start = time.time()
        best_config, all_configs = find_best_config(video_path, model)
        eval_time = time.time() - eval_start

        print(f"\n   📊 Resultados de evaluación ({eval_time:.1f}s):")
        for j, cfg in enumerate(all_configs[:3], 1):
            emoji = "🥇" if j == 1 else "🥈" if j == 2 else "🥉"
            print(f"   {emoji} CONF={cfg['conf']:.2f} IOU={cfg['iou']:.2f} → "
                  f"V:{cfg['vehicles']} Eff:{cfg['efficiency']:.2f} Conf:{cfg['avg_conf']:.3f}")

        print(f"\n   ✅ Configuración óptima seleccionada:")
        print(f"      • CONF = {best_config['conf']:.2f}")
        print(f"      • IOU = {best_config['iou']:.2f}")
        print(f"      • Eficiencia = {best_config['efficiency']:.2f}")

        # FASE 2: Procesar video completo
        print(f"\n🎬 FASE 2: Procesando video completo...")

        metrics = process_video_with_config(
            video_path, model,
            best_config['conf'], best_config['iou'],
            OUTPUT_FOLDER, i, total_videos
        )

        all_metrics.append(metrics)

        # Resumen del video
        print(f"\n   ✅ Video completado:")
        print(
            f"      • Vehículos contados: {metrics['tracking_results']['vehicles_counted']}")
        print(
            f"      • Tiempo de procesamiento: {metrics['processing_info']['processing_time_seconds']:.1f}s")
        print(
            f"      • Confianza promedio: {metrics['tracking_results']['avg_confidence']:.3f}")
        print(f"      • Video guardado: {video_name}_tracked.mp4")
        print(f"      • JSON guardado: {video_name}_metrics.json")

        # Progreso general
        elapsed_total = time.time() - total_start_time
        avg_time_per_video = elapsed_total / i
        videos_remaining = total_videos - i
        eta_total = avg_time_per_video * videos_remaining

        print(f"\n   📊 PROGRESO GENERAL:")
        print(
            f"      • Completados: {i}/{total_videos} ({(i/total_videos)*100:.1f}%)")
        print(f"      • Tiempo transcurrido: {elapsed_total/60:.1f} min")
        print(f"      • Tiempo promedio/video: {avg_time_per_video:.1f}s")
        print(f"      • Videos restantes: {videos_remaining}")
        print(f"      • Tiempo estimado restante: {eta_total/60:.1f} min")
        print(
            f"      • Finalización estimada: {datetime.fromtimestamp(time.time() + eta_total).strftime('%H:%M:%S')}")

    # Resumen final
    total_time = time.time() - total_start_time
    total_vehicles = sum(m['tracking_results']['vehicles_counted']
                         for m in all_metrics)

    summary = {
        'processing_summary': {
            'total_videos_processed': total_videos,
            'total_vehicles_counted': total_vehicles,
            'total_processing_time_seconds': round(total_time, 2),
            'total_processing_time_minutes': round(total_time / 60, 2),
            'average_time_per_video_seconds': round(total_time / total_videos, 2),
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'videos_summary': [
            {
                'filename': m['video_info']['filename'],
                'vehicles_counted': m['tracking_results']['vehicles_counted'],
                'optimal_conf': m['optimal_configuration']['conf_threshold'],
                'optimal_iou': m['optimal_configuration']['iou_threshold'],
                'processing_time_seconds': m['processing_info']['processing_time_seconds'],
                'avg_confidence': m['tracking_results']['avg_confidence']
            }
            for m in all_metrics
        ]
    }

    summary_path = os.path.join(OUTPUT_FOLDER, '_RESUMEN_COMPLETO.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'█'*70}")
    print("🎉 PROCESAMIENTO COMPLETADO")
    print(f"{'█'*70}")
    print(f"\n📊 ESTADÍSTICAS FINALES:")
    print(f"   • Total de videos procesados: {total_videos}")
    print(f"   • Total de vehículos contados: {total_vehicles}")
    print(f"   • Tiempo total: {total_time/60:.1f} minutos")
    print(f"   • Promedio por video: {total_time/total_videos:.1f} segundos")
    print(f"\n📁 ARCHIVOS GENERADOS:")
    print(f"   • Carpeta de salida: {OUTPUT_FOLDER}/")
    print(f"   • Resumen general: {summary_path}")
    print(f"   • Videos con tracking: {total_videos} archivos *_tracked.mp4")
    print(
        f"   • Métricas individuales: {total_videos} archivos *_metrics.json")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
