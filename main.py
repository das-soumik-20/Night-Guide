"""
Night-Guide entry point.

Defaults: RGB model, spoken alerts on.

Examples:
    python main.py --source 0                        # laptop webcam
    python main.py --source http://192.168.137.45:4747/video   # phone (DroidCam over Wi-Fi)
    python main.py --source data/test_videos/Vnit.mp4 --save output/vnit.mp4  # + output/vnit_voice.mp4
    python main.py --source data/test_videos/photo.jpeg
    python main.py --source 0 --model yolov8n.pt     # stock COCO model, handy for indoor testing
    python main.py --source 0 --model both           # thermal + RGB, keep the more confident
    python main.py --source 0 --no-speak             # silent
    python main.py --source 0 --show-conf 0.2        # debug: also draw low-confidence detections
    python main.py --source 0 --min-persist 0.5      # hide objects seen < 0.5 s
    python main.py --source 0 --danger-distance 3    # alert from 3 m instead of 2 m
    python main.py --evaluate                        # re-run test-set evaluation for both models
"""
import argparse

from run_demo import run_on_video


def evaluate_both():
    from evaluate import evaluate_model
    print("Evaluating thermal model...")
    evaluate_model('runs/detect/thermal_full/weights/best.pt', 'configs/thermal.yaml', 'test')
    print("\nEvaluating rgb model...")
    evaluate_model('runs/detect/rgb_full/weights/best.pt', 'configs/rgb.yaml', 'test')


def main():
    parser = argparse.ArgumentParser(description='Night-Guide obstacle detection demo')
    parser.add_argument('--source', default='data/test_videos/walking.mp4',
                        help='video file, image file, or camera index (0, 1, 2 ...)')
    parser.add_argument('--model', default='rgb',
                        help="'rgb' (default), 'thermal', 'both', or a path to a YOLO .pt file")
    parser.add_argument('--danger-distance', type=float, default=None,
                        help='alert when an object is closer than this many metres (default 2.0)')
    parser.add_argument('--show-conf', type=float, default=None,
                        help='debug: draw every detection above this confidence (e.g. 0.2); alerts unchanged')
    parser.add_argument('--save', default=None,
                        help='save the annotated result (boxes + alerts), e.g. output/road_demo.mp4 or output/photo.jpg')
    parser.add_argument('--min-persist', type=float, default=0.0,
                        help='only show/alert objects tracked at least this many seconds, e.g. 1.0 '
                             '(filters flickering false positives, but can miss short appearances). Default: off')
    parser.add_argument('--speak', action=argparse.BooleanOptionalAction, default=True,
                        help='speak alerts out loud (default: on; --no-speak to turn off). With --save on a '
                             'video file, also writes <name>_voice.mp4 with the spoken alerts baked in')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate both trained models on the test split instead of running the demo')
    args = parser.parse_args()

    if args.evaluate:
        evaluate_both()
        return

    print("\nRunning demo...")
    run_on_video(args.source, model=args.model, danger_distance_m=args.danger_distance,
                 show_conf=args.show_conf, save_path=args.save,
                 min_persist_s=args.min_persist, speak=args.speak)


if __name__ == "__main__":
    main()
