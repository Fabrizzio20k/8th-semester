from typing import List, Tuple, Dict
import os
import cv2
import json
import numpy as np
from ultralytics import YOLO

folder = "kfilter"
output_folder = "output_videos"
results_json = "conteo_resultados.json"

os.makedirs(output_folder, exist_ok=True)

rest_of_videos: List[str] = [os.path.join(folder, video) for video in [
    "1-1.mp4", "2-1.mp4", "3-1.mp4", "4-1.mp4", "5-1.mp4",
    "6-1.mp4", "7-1.mp4", "8-1.mp4", "9-1.mp4", "10-1.mp4",
    "11-1.mp4", "12-1.mp4", "13-1.mp4", "14-1.mp4", "15-1.mp4",
    "16-1.mp4", "17-1.mp4", "18-1.mp4", "19-1.mp4", "20-1.mp4",
    "21-1.mp4", "22-1.mp4", "23-1.mp4", "24-1.mp4", "25-1.mp4",
    "28-1.mp4", "29-1.mp4", "30-1.mp4", "31-1.mp4", "32-1.mp4",
    "33-1.mp4", "34-1.mp4", "35-1.mp4", "36-1.mp4", "37-1.mp4",
    "38-1.mp4", "39-1.mp4", "41-1.mp4", "42-1.mp4", "43-1.mp4"
]]

available_videos = [v for v in rest_of_videos if os.path.exists(v)]
current_video_index = 0
all_results = {}

lines_points = []
display_frame = None
original_frame = None
window_name = "Configurar Lineas"


def mouse_callback(event, x, y, flags, param):
    global lines_points, display_frame, original_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        lines_points.append((x, y))
        print(f"Punto {len(lines_points)} marcado en ({x}, {y})")

        display_frame = original_frame.copy()

        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255),
                  (255, 255, 0), (255, 128, 0)]

        for i, point in enumerate(lines_points):
            color = colors[(i // 2) % len(colors)]
            cv2.circle(display_frame, point, 8, color, -1)
            cv2.putText(display_frame, f"P{i+1}", (point[0]+12, point[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for i in range(0, len(lines_points)-1, 2):
            color = colors[(i // 2) % len(colors)]
            cv2.line(display_frame,
                     lines_points[i], lines_points[i+1], color, 3)
            mid_x = (lines_points[i][0] + lines_points[i+1][0]) // 2
            mid_y = (lines_points[i][1] + lines_points[i+1][1]) // 2
            cv2.putText(display_frame, f"Linea {(i//2)+1}", (mid_x-40, mid_y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if len(lines_points) % 2 == 1:
            cv2.putText(display_frame, "Haz clic en el segundo punto", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(window_name, display_frame)


def configure_lines_for_video(video_path: str) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    global lines_points, display_frame, original_frame, window_name

    lines_points = []

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error al leer video: {video_path}")
        return None

    original_frame = frame.copy()
    display_frame = frame.copy()

    window_name = f"Configurar - {os.path.basename(video_path)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    height, width = frame.shape[:2]
    cv2.resizeWindow(window_name, min(width, 1400), min(height, 900))

    print(f"\n{'='*60}")
    print(f"VIDEO: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    print("INSTRUCCIONES:")
    print("  - Haz clic en 2 puntos para crear cada línea")
    print("  - Las líneas se dibujan automáticamente al marcar el 2do punto")
    print("  - ENTER: Confirmar y procesar")
    print("  - R: Borrar todas las líneas")
    print("  - U: Deshacer último punto")
    print("  - Q: Saltar este video")
    print(f"{'='*60}\n")

    status_text = f"Puntos: 0 | Lineas: 0"
    cv2.putText(display_frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow(window_name, display_frame)

    while True:
        temp_frame = display_frame.copy()

        status_text = f"Puntos: {len(lines_points)} | Lineas: {len(lines_points)//2}"
        cv2.rectangle(temp_frame, (0, 0), (400, 45), (0, 0, 0), -1)
        cv2.putText(temp_frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(window_name, temp_frame)

        key = cv2.waitKey(50) & 0xFF

        if key == 13:
            if len(lines_points) >= 2 and len(lines_points) % 2 == 0:
                break
            else:
                print("⚠️  Necesitas al menos 2 puntos para formar 1 línea completa")

        elif key == ord('r') or key == ord('R'):
            lines_points = []
            display_frame = original_frame.copy()
            cv2.imshow(window_name, display_frame)
            print("🔄 Líneas borradas")

        elif key == ord('u') or key == ord('U'):
            if lines_points:
                removed = lines_points.pop()
                display_frame = original_frame.copy()

                colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255),
                          (255, 255, 0), (255, 128, 0)]

                for i, point in enumerate(lines_points):
                    color = colors[(i // 2) % len(colors)]
                    cv2.circle(display_frame, point, 8, color, -1)
                    cv2.putText(display_frame, f"P{i+1}", (point[0]+12, point[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                for i in range(0, len(lines_points)-1, 2):
                    color = colors[(i // 2) % len(colors)]
                    cv2.line(display_frame,
                             lines_points[i], lines_points[i+1], color, 3)
                    mid_x = (lines_points[i][0] + lines_points[i+1][0]) // 2
                    mid_y = (lines_points[i][1] + lines_points[i+1][1]) // 2
                    cv2.putText(display_frame, f"Linea {(i//2)+1}", (mid_x-40, mid_y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.imshow(window_name, display_frame)
                print(f"⬅️  Punto eliminado: {removed}")

        elif key == ord('q') or key == ord('Q'):
            cv2.destroyAllWindows()
            print("⏭️  Video saltado\n")
            return None

    cv2.destroyAllWindows()

    configured_lines = []
    for i in range(0, len(lines_points), 2):
        configured_lines.append((lines_points[i], lines_points[i+1]))

    print(f"✅ {len(configured_lines)} línea(s) configurada(s)\n")
    return configured_lines


def point_side(point, line_start, line_end):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def process_video_with_lines(video_path: str, lines: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Dict:
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = os.path.join(output_folder, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🎬 Procesando video...")
    print(f"   Resolución: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")

    model = YOLO('yolov8n.pt')
    vehicle_classes = [2, 3, 5, 7]

    line_counts = [0] * len(lines)
    tracked_crossings = {}

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 100 == 0:
            print(
                f"   Progreso: {frame_count}/{total_frames} frames ({frame_count*100//total_frames}%)")

        results = model.track(frame, persist=True,
                              classes=vehicle_classes, verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if track_id not in tracked_crossings:
                    tracked_crossings[track_id] = {
                        'prev_centroid': (cx, cy),
                        'crossed_lines': set()
                    }

                prev_cx, prev_cy = tracked_crossings[track_id]['prev_centroid']

                for line_idx, (line_start, line_end) in enumerate(lines):
                    if line_idx not in tracked_crossings[track_id]['crossed_lines']:
                        prev_side = point_side(
                            (prev_cx, prev_cy), line_start, line_end)
                        curr_side = point_side((cx, cy), line_start, line_end)

                        if (prev_side > 0 and curr_side < 0) or (prev_side < 0 and curr_side > 0):
                            line_counts[line_idx] += 1
                            tracked_crossings[track_id]['crossed_lines'].add(
                                line_idx)

                tracked_crossings[track_id]['prev_centroid'] = (cx, cy)

                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.rectangle(frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), (255, 0, 0), 2)

        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255),
                  (255, 255, 0), (255, 128, 0)]
        for line_idx, (line_start, line_end) in enumerate(lines):
            color = colors[line_idx % len(colors)]
            cv2.line(frame, line_start, line_end, color, 3)
            mid_x = (line_start[0] + line_end[0]) // 2
            mid_y = (line_start[1] + line_end[1]) // 2
            cv2.putText(frame, f"L{line_idx+1}: {line_counts[line_idx]}",
                        (mid_x, mid_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"Total: {sum(line_counts)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        out.write(frame)

    cap.release()
    out.release()

    print(f"✅ Procesamiento completado")
    print(f"   Conteos por línea: {line_counts}")
    print(f"   Total: {sum(line_counts)} vehículos")
    print(f"   Guardado en: {output_path}\n")

    return {
        "conteos_por_linea": line_counts,
        "total": sum(line_counts)
    }


print("\n" + "="*70)
print("🚗 CONTADOR DE VEHÍCULOS CON YOLO")
print("="*70)
print(f"Videos a procesar: {len(available_videos)}")
print("="*70 + "\n")

for video_path in available_videos:
    lines = configure_lines_for_video(video_path)

    if lines is None:
        continue

    counts_data = process_video_with_lines(video_path, lines)

    video_name = os.path.basename(video_path)
    all_results[video_name] = {
        "lineas": [{"inicio": line[0], "fin": line[1]} for line in lines],
        "conteos_por_linea": counts_data["conteos_por_linea"],
        "total": counts_data["total"]
    }

with open(results_json, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("\n" + "="*70)
print("🎉 PROCESAMIENTO COMPLETADO")
print("="*70)
print(f"Resultados guardados en: {results_json}")
print(f"Videos procesados en: {output_folder}")
print("\nRESUMEN:")
for video, data in all_results.items():
    print(f"  📹 {video}: {data['total']} vehículos")
print("="*70 + "\n")
