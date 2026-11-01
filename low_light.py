# low_light.py
"""
Low-light image enhancement applied to each frame before detection.

Two steps, both cheap enough for real time on a laptop CPU:
1. Adaptive gamma - lifts dark frames towards a normal brightness. The darker
   the frame, the stronger the lift; bright frames are left alone.
2. CLAHE (Contrast Limited Adaptive Histogram Equalization) on the lightness
   channel only - restores local contrast (edges, faces, outlines) that the
   model needs, without shifting colours.
Very dark frames also get a light median blur first, so sensor grain isn't amplified.

Modes (python main.py --enhance ...):
    auto  - only enhance frames darker than DARK_THRESHOLD (default)
    on    - always enhance
    off   - never enhance (original behaviour)
"""
import cv2
import numpy as np

DARK_THRESHOLD = 90      # mean brightness (0-255) below which 'auto' kicks in
TARGET_BRIGHTNESS = 0.45  # gamma aims the mean brightness at ~45% grey
MAX_LIFT = 0.3            # smallest gamma exponent allowed (strongest brightening)
CLAHE_CLIP = 2.0          # higher = more contrast, but also more amplified noise
CLAHE_GRID = (8, 8)
VERY_DARK = 35           # below this, smooth sensor noise before brightening it

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)


def brightness(frame):
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def adaptive_gamma(frame, mean):
    m = max(mean, 1.0) / 255.0
    exponent = np.log(TARGET_BRIGHTNESS) / np.log(m)       # m ** exponent == target
    exponent = float(np.clip(exponent, MAX_LIFT, 1.0))     # only brighten, never darken
    if exponent >= 0.99:
        return frame
    lut = (np.linspace(0, 1, 256) ** exponent * 255).astype(np.uint8)
    return cv2.LUT(frame, lut)


def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((_clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def enhance_low_light(frame):
    mean = brightness(frame)
    if mean < VERY_DARK:
        frame = cv2.medianBlur(frame, 3)  # cheap; stops grain turning into false edges
    return apply_clahe(adaptive_gamma(frame, mean))


def maybe_enhance(frame, mode='auto'):
    """Returns (frame_for_the_model, was_enhanced, mean_brightness)."""
    mean = brightness(frame)
    if mode == 'on' or (mode == 'auto' and mean < DARK_THRESHOLD):
        return enhance_low_light(frame), True, mean
    return frame, False, mean
