# Night-Guide
Low-Light and Night-Time Obstacle Detection for Blind Pedestrians: an obstacle detection system that works in low-light/night conditions using RGB + thermal/infrared imaging and gives real-time alerts.

## How it works
1. **Detect** – two YOLOv8n models trained on the FLIR dataset, one on thermal images and one on RGB. On each frame both run and the more confident one is used.
2. **Track** – `ObjectTracker` gives every obstacle a persistent ID across frames, so the same car isn't announced over and over.
3. **Measure** – distance in metres from the pinhole camera formula (`real height × focal length ÷ pixel height`, with the focal length scaled to the video's resolution), direction (left / center / right) from the box position.
4. **Alert** – only for real pedestrian obstacles closer than 2 m (configurable), only once per object unless it gets ≥0.3 m closer, most dangerous first.

## Project layout
| File | Purpose |
|---|---|
| `main.py` | Entry point – run the demo on a video, image or live camera, or re-run evaluation |
| `run_demo.py` | Detection + tracking + alert loop with on-screen overlay |
| `alert_logic.py` | Class filtering, distance/direction, `ObjectTracker`, alert ranking |
| `merge_detections.py` | IoU-based merge of thermal + RGB detections (for when two separate camera feeds are available) |
| `find_camera.py` | Lists working camera indexes (laptop webcam, phone-as-webcam, USB) |
| `test_alert.py` | Offline tests for the alert pipeline – no camera or model needed |
| `train.py` / `evaluate.py` | Train the models / plot per-class mAP50 on the test split |
| `convert_coco_to_yolo.py` | Converts FLIR COCO annotations to YOLO labels (9 classes) |
| `configs/` | Dataset configs for the thermal and RGB models |
| `runs/detect/*/weights/best.pt` | Trained thermal and RGB weights |
| `results/` | Test-set evaluation charts |

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate on macOS/Linux)
pip install ultralytics opencv-python matplotlib
```
The `data/` folder (FLIR dataset + test videos) is not in git – place it in the project root.

## Usage
```bash
python main.py                                          # default test video
python main.py --source data/test_videos/human_night.mp4
python main.py --source data/test_videos/photo.jpeg     # image: press any key to close
python find_camera.py                                   # which camera index is which?
python main.py --source 0                               # live webcam
python main.py --source 2 --model rgb                   # phone as webcam, RGB model only
python main.py --source 0 --model yolov8n.pt            # stock COCO model, good for indoor tests
python main.py --danger-distance 3                      # alert from 3 m instead of 2 m
python main.py --source road.mp4 --model rgb --save output/road_demo.mp4   # save annotated video
python main.py --evaluate                               # regenerate the charts in results/
python test_alert.py                                    # check the alert logic
```
Press `q` in the video window to quit.

### Using a phone as the camera (DroidCam)
DroidCam's Windows virtual webcam does not work with OpenCV (green/empty frames), so read the phone's stream directly over Wi-Fi instead:
1. Install the **DroidCam** app on the phone; laptop and phone on the same Wi-Fi.
2. Open the app and note the **Wi-Fi IP** it shows (it can change after reconnecting).
3. Make sure nothing else is using the phone: close the DroidCam Windows client and any browser tab showing the stream – the phone serves **one viewer at a time** and otherwise replies "DroidCam is Busy".
4. Run:
   ```bash
   python main.py --source http://<phone-ip>:4747/video --model rgb
   ```
Use `--model rgb` for a phone: it is a normal colour camera, the thermal model expects heat images.

### Confidence thresholds
A detection must be ≥ 50% confident to count. Classes the models are weak on (bicycle, truck, sign, fire hydrant, traffic light) need ≥ 70% to avoid false alarms. Tune `MIN_CONFIDENCE` in `alert_logic.py`.

## Results (held-out test set, mAP50)
| Model | Overall | person | car | bike (motorcycle) |
|---|---|---|---|---|
| Thermal | 0.28 | 0.74 | 0.75 | 0.28 |
| RGB | 0.25 | 0.44 | 0.55 | 0.47 |

Bicycle, truck, traffic light, hydrant and sign are still weak (< 0.3); the test split has no buses.

## Known limitations / next steps
- Distances assume a typical phone lens (`FOCAL_RATIO = 0.72` in `alert_logic.py`) and average object heights (a person = 1.7 m), so they are estimates. Calibrate `FOCAL_RATIO` for a specific camera (method in the code comment); the FLIR thermal camera has a different lens.
- Both models currently see the **same** frame. True RGB + thermal fusion needs two camera feeds; `merge_detections.py` is ready for that.
- Improve weak classes (more data, class weighting, or a larger model such as YOLOv8s).

## Contributors
Soumik, Shrutika, Vaibhavi
