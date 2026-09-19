# run_demo.py
import cv2
from ultralytics import YOLO
from alert_logic import should_alert

THERMAL_MODEL_PATH = 'runs/detect/thermal_full/weights/best.pt'
RGB_MODEL_PATH = 'runs/detect/rgb_full/weights/best.pt'

def get_detections(model, frame):
    results = model.predict(frame, verbose=False)[0]
    dets = [
        (results.names[int(box.cls)], float(box.conf), box.xywhn[0].tolist())
        for box in results.boxes
    ]
    avg_conf = sum(d[1] for d in dets) / len(dets) if dets else 0.0
    return dets, avg_conf, results

def run_on_video(video_path):
    thermal_model = YOLO(THERMAL_MODEL_PATH)
    rgb_model = YOLO(RGB_MODEL_PATH)

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        thermal_dets, thermal_conf, thermal_results = get_detections(thermal_model, frame)
        rgb_dets, rgb_conf, rgb_results = get_detections(rgb_model, frame)

        if thermal_conf >= rgb_conf:
            chosen_dets, chosen_results, source = thermal_dets, thermal_results, 'THERMAL'
        else:
            chosen_dets, chosen_results, source = rgb_dets, rgb_results, 'RGB'

        alerts = should_alert(chosen_dets)
        annotated_frame = chosen_results.plot()

        cv2.putText(annotated_frame, f"Source: {source}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for alert in alerts:
            cv2.putText(annotated_frame, f"ALERT: {alert['class']} {alert['direction']}",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('NightCompass Demo', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_on_video('data/left_overs/video_thermal_test/sample.mp4')  # quick standalone test