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

from alert_logic import (DANGER_DISTANCE_M, DEFAULT_MIN_CONFIDENCE, ObjectTracker,
                         format_alert, passes_confidence)
from speech import Speaker, alerts_to_speech, make_narrated_video

THERMAL_MODEL_PATH = 'runs/detect/thermal_full/weights/best.pt'
RGB_MODEL_PATH = 'runs/detect/rgb_full/weights/best.pt'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MAX_PREVIEW_HEIGHT = 900   # px; keeps tall phone videos on screen
MAX_PREVIEW_WIDTH = 1600


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


CLASS_COLORS = {  # BGR
    'person': (255, 80, 0), 'car': (0, 200, 255), 'motorcycle': (180, 0, 255),
    'bicycle': (0, 255, 0), 'bus': (0, 140, 255), 'truck': (0, 90, 255),
}
ALERT_HOLD_S = 2.0  # keep each alert on screen this long so it can be read


def draw_tracks(frame, tracks):
    """Draw confirmed objects with class, confidence and estimated distance."""
    H, W = frame.shape[:2]
    thick = max(2, round(H / 500))
    font = max(0.5, H / 1600)
    for t in tracks:
        cx, cy, bw, bh = t['box']
        x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
        x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
        color = CLASS_COLORS.get(t['class_name'], (200, 200, 200))
        label = f"{t['class_name']} {t['confidence']:.2f}"
        if t['distance_m'] is not None:
            label += f" {t['distance_m']:.1f}m"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        ty = max(y1, th + 6)
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 4, ty), color, -1)
        cv2.putText(frame, label, (x1 + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thick)
    return frame


def run_on_video(source, model='both', danger_distance_m=None, show_conf=None, save_path=None,
                 min_persist_s=0.0, speak=False):
    models = load_models(model)
    cap, is_image = open_source(source)
    is_file = not (isinstance(source, int) or str(source).isdigit() or str(source).startswith('http'))
    # min_persist_s > 0 (--min-persist): only show/alert objects seen that long.
    # A single photo can't show persistence, so it is always off there.
    min_persist_s = 0.0 if is_image else (min_persist_s or 0.0)
    tracker = ObjectTracker(danger_distance_m or DANGER_DISTANCE_M, min_persist_s)
    on_screen = []  # (alert text, expiry time)
    speaker = Speaker() if speak else None
    spoken_events = []  # (video time, sentence) for the narrated video
    frames_written = 0

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

        # video files use their own timeline: processing is slower than real
        # time, and "seen for 1 second" must mean 1 second of video
        now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if is_file else time.time()

        active_tracks = tracker.update(chosen_dets, frame.shape[0], now=now,
                                       frame_width_px=frame.shape[1])
        alerts = tracker.get_alerts(active_tracks, now=now)

        for alert in alerts:
            stamp = f"[{now:6.1f}s] " if is_file else ""
            print(f"{stamp}ALERT: {format_alert(alert)}")
            on_screen.append((f"ALERT: {format_alert(alert)}", now + ALERT_HOLD_S))
        if alerts and speak:
            sentence = alerts_to_speech(alerts)
            speaker.say(sentence)
            if is_file:
                spoken_events.append((now, sentence))
        on_screen = [(text, until) for text, until in on_screen if until > now][-4:]

        if show_conf or not min_persist_s:  # every detection the model made (default)
            annotated_frame = chosen_results.plot()
        else:          # --min-persist: only objects confirmed by the tracker
            confirmed = [t for t in active_tracks if tracker.is_confirmed(t, now)]
            annotated_frame = draw_tracks(frame.copy(), confirmed)

        H = annotated_frame.shape[0]
        font, thick, line = max(0.8, H / 1200), max(2, round(H / 500)), int(max(35, H / 30))
        cv2.putText(annotated_frame, f"Source: {source_label}", (20, line),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 255, 0), thick)
        for i, (text, _) in enumerate(on_screen):  # one line per alert, no overlap
            cv2.putText(annotated_frame, text, (20, line * (i + 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 255), thick)

        if save_path and not is_image:
            if writer is None:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                h, w = annotated_frame.shape[:2]
                writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            writer.write(annotated_frame)
            frames_written += 1
        elif save_path and is_image:
            cv2.imwrite(save_path, annotated_frame)

        # shrink the preview to fit on screen (saved output keeps full resolution)
        h, w = annotated_frame.shape[:2]
        scale = min(1.0, MAX_PREVIEW_HEIGHT / h, MAX_PREVIEW_WIDTH / w)
        preview = cv2.resize(annotated_frame, None, fx=scale, fy=scale) if scale < 1 else annotated_frame
        cv2.imshow('Night-Guide Demo', preview)
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
    if speaker:
        speaker.close()
    # with --speak and --save on a video file, also write a copy with the voice in it
    if speak and save_path and is_file and not is_image and frames_written:
        out = str(Path(save_path).with_name(Path(save_path).stem + '_voice.mp4'))
        print("Making narrated video (spoken alerts baked in)...")
        if make_narrated_video(save_path, spoken_events, out, frames_written / fps):
            print(f"Saved narrated video to {out}")


if __name__ == '__main__':
    run_on_video('data/test_videos/walking.mp4')  # quick standalone test
