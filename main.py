"""
Night-Guide entry point.

Examples:
    python main.py                                   # demo on the default test video
    python main.py --source data/test_videos/human_night.mp4
    python main.py --source data/test_videos/photo.jpeg
    python main.py --source 0                        # laptop webcam
    python main.py --source 2 --model rgb            # phone used as a webcam (find the index with find_camera.py)
    python main.py --source 0 --model yolov8n.pt     # stock COCO model, handy for indoor testing
    python main.py --source 0 --show-conf 0.2        # debug: also draw low-confidence detections
    python main.py --source road.mp4 --model rgb --save output/road_demo.mp4   # save result for slides
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
    parser.add_argument('--model', default='both',
                        help="'both', 'thermal', 'rgb', or a path to a YOLO .pt file")
    parser.add_argument('--danger-distance', type=float, default=None,
                        help='alert when an object is closer than this many metres (default 2.0)')
    parser.add_argument('--show-conf', type=float, default=None,
                        help='debug: draw every detection above this confidence (e.g. 0.2); alerts unchanged')
    parser.add_argument('--save', default=None,
                        help='save the annotated result (boxes + alerts), e.g. output/road_demo.mp4 or output/photo.jpg')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate both trained models on the test split instead of running the demo')
    args = parser.parse_args()

    if args.evaluate:
        evaluate_both()
        return

    print("\nRunning demo...")
    run_on_video(args.source, model=args.model, danger_distance_m=args.danger_distance,
                 show_conf=args.show_conf, save_path=args.save)


if __name__ == "__main__":
    main()
