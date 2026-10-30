"""
Lists which camera indexes work on this machine (laptop webcam, USB camera,
phone connected as a webcam, ...) and briefly previews each one.

    python find_camera.py            # check indexes 0-4
    python find_camera.py --max 8    # check more
    python find_camera.py --no-preview

Then run the demo on the one you want:  python main.py --source <index>
"""
import argparse

import cv2


def open_camera(index):
    # CAP_DSHOW is faster and more reliable for phone/USB cameras on Windows
    if hasattr(cv2, 'CAP_DSHOW'):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=5, help='number of indexes to check')
    parser.add_argument('--no-preview', action='store_true')
    args = parser.parse_args()

    for i in range(args.max):
        cap = open_camera(i)
        if not cap.isOpened():
            print(f"Camera index {i}: not available")
            continue
        ret, frame = cap.read()
        if ret:
            print(f"Camera index {i}: WORKING, resolution {frame.shape[1]}x{frame.shape[0]}")
            if not args.no_preview:
                cv2.imshow(f"Camera {i}", frame)
                cv2.waitKey(1500)  # show for 1.5 seconds
                cv2.destroyWindow(f"Camera {i}")
        else:
            print(f"Camera index {i}: opened but no frame")
        cap.release()


if __name__ == '__main__':
    main()
