from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    # model.train(
    #     data='configs/thermal.yaml',
    #     epochs=100,
    #     patience=20,
    #     device=0,
    #     imgsz=640,
    #     batch=16,
    #     name='thermal_full'
    # )

    model.train(
        data='configs/rgb.yaml',
        epochs=100,
        patience=20,
        device=0,
        imgsz=640,
        batch=16,
        name='rgb_full'
    )