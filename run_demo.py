# run_demo.py
"""
Live / offline demo: detect obstacles, track them and raise alerts.

Source can be a video file, an image file, or a camera index (0, 1, 2 ...).
Model can be 'both' (thermal + RGB, picks the more confident one per frame),
'thermal', 'rgb', or a path to any YOLO .pt file (e.g. the stock yolov8n.pt).
"""
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from alert_logic import DEFAULT_MIN_CONFIDENCE, ObjectTracker, format_alert, passes_confidence

THERMAL_MODEL_PATH = 'runs/detect/thermal_full/weights/best.pt'
RGB_MODEL_PATH = 'runs/detect/rgb_full/weights/best.pt'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def get_detections(model, frame, show_conf=None):
    """
    show_conf=None: normal mode, per-class thresholds from alert_logic.
    show_conf=0.25 (--show-conf): debug mode, draw everything above that
    confidence so you can see what the model *almost* detected. Alerts still
    use the normal thresholds (the tracker filters again).
    """
    results = model.predict(frame, verbose=False, conf=show_conf or DEFAULT_MIN_CONFIDENCE)[0]
    # drop weak-class guesses below their own threshold, so they are neither
    # drawn on screen nor alerted on
    keep = [i for i, box in enumerate(results.boxes)
            if show_conf or passes_confidence(results.names[int(box.cls)], float(box.conf))]
    if len(keep) != len(results.boxes):
        try:
            results = results[keep]
        except Exception:  # older/newer ultralytics versions
            results.update(boxes=results.boxes.data[keep])
    dets = [
        (results.names[int(box.cls)], float(box.conf), box.xywhn[0].tolist())
        for box in results.boxes
    ]
    avg_conf = sum(d[1] for d in dets) / len(dets) if dets else 0.0
    return dets, avg_conf, results


def load_models(model='both'):
    """Returns a list of (label, YOLO model)."""
    if model == 'both':
        return [('THERMAL', YOLO(THERMAL_MODEL_PATH)), ('RGB', YOLO(RGB_MODEL_PATH))]
    if model == 'thermal':
        return [('THERMAL', YOLO(THERMAL_MODEL_PATH))]
    if model == 'rgb':
        return [('RGB', YOLO(RGB_MODEL_PATH))]
    return [(Path(model).stem.upper(), YOLO(model))]


def open_source(source):
    """Camera index (int or digit string) or a file path."""
    if isinstance(source, int) or str(source).isdigit():
        index = int(source)
        # CAP_DSHOW opens USB / phone-as-webcam devices much faster on Windows
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, 'CAP_DSHOW') else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        return cap, False
    is_image = Path(str(source)).suffix.lower() in IMAGE_EXTS
    return cv2.VideoCapture(str(source)), is_image


def save_snapshot(raw, annotated):
    """'s' key: save the camera frame and the annotated result."""
    Path('snapshots').mkdir(exist_ok=True)
    stamp = time.strftime('%H%M%S')
    for name, img in (('raw', raw), ('annotated', annotated)):
        cv2.imwrite(f'snapshots/{stamp}_{name}.jpg', img)
    print(f"Saved snapshots/{stamp}_*.jpg")


def run_on_video(source, model='both', danger_distance_m=None, show_conf=None, save_path=None):
    models = load_models(model)
    tracker = ObjectTracker(danger_distance_m) if danger_distance_m else ObjectTracker()

    cap, is_image = open_source(source)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        return

    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and 1 < fps < 121 else 30.0  # live streams often report 0

    print("Running... click the video window, press 's' to save a snapshot, 'q' to quit")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # run every model, keep the one that is most confident on this frame
        best = None
        for label, m in models:
            dets, conf, results = get_detections(m, frame, show_conf)
            if best is None or conf > best[1]:
                best = (dets, conf, results, label)
        chosen_dets, _, chosen_results, source_label = best

        active_tracks = tracker.update(chosen_dets, frame.shape[0])
        alerts = tracker.get_alerts(active_tracks)

        for alert in alerts:
            print(f"ALERT: {format_alert(alert)}")

        annotated_frame = chosen_results.plot()
        cv2.putText(annotated_frame, f"Source: {source_label}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for i, alert in enumerate(alerts):  # one line per alert, no overlap
            cv2.putText(annotated_frame, f"ALERT: {format_alert(alert)}",
                        (20, 65 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if save_path and not is_image:
            if writer is None:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                h, w = annotated_frame.shape[:2]
                writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            writer.write(annotated_frame)
        elif save_path and is_image:
            cv2.imwrite(save_path, annotated_frame)

        cv2.imshow('Night-Guide Demo', annotated_frame)
        # images stay open until a key is pressed; video/camera keep streaming
        key = cv2.waitKey(0 if is_image else 1) & 0xFF
        if key == ord('s'):
            save_snapshot(frame, annotated_frame)
        if key == ord('q') or is_image:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if save_path:
        print(f"Saved annotated output to {save_path}")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_on_video('data/test_videos/walking.mp4')  # quick standalone test
