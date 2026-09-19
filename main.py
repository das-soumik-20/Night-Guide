from evaluate import evaluate_model
from run_demo import run_on_video

def main(path):
    # print("Evaluating thermal model...")
    # evaluate_model('runs/detect/thermal_full/weights/best.pt', 'configs/thermal.yaml', 'test')
    # print("Evaluating thermal model...")
    #
    # print("\nEvaluating rgb model...")
    # evaluate_model('runs/detect/rgb_full/weights/best.pt', 'configs/rgb.yaml', 'test')

    print("\nRunning demo...")
    run_on_video(path)



if __name__ == "__main__":
    main('data/test_videos/human_night.mp4')