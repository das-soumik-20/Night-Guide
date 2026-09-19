# alert_logic.py

def estimate_proximity(box):
    _, _, _, h = box
    if h > 0.5:
        return 'close'
    elif h > 0.25:
        return 'medium'
    return 'far'

def estimate_direction(box):
    x_center, _, _, _ = box
    if x_center < 0.35:
        return 'left'
    elif x_center > 0.65:
        return 'right'
    return 'center'

def should_alert(detections):
    alerts = []
    for class_name, confidence, box in detections:
        proximity = estimate_proximity(box)
        direction = estimate_direction(box)
        if proximity == 'close':
            alerts.append({'class': class_name, 'direction': direction, 'proximity': proximity})
    return alerts

'''
---------------------------------------------------------------------------
METRIC DISTANCE ESTIMATION (disabled until real hardware is available)

Uses the pinhole camera formula to convert a detected object's pixel height
into an approximate real-world distance in meters:

    distance_m = (real_world_height_m * focal_length_px) / pixel_height

FOCAL_LENGTH_PX must come from the actual thermal camera's datasheet or a
calibration procedure (e.g. photographing an object of known size at a
known distance and solving for focal length). It is NOT the same as the
imgsz=640 training parameter, that's just image resolution, not focal length.

AVERAGE_REAL_HEIGHTS are rough real-world sizes (in meters) used as the
assumed size for each class. These are approximations, accuracy will vary
more for classes with high size variance (e.g. "sign") and less for
classes with consistent size (e.g. "person").
---------------------------------------------------------------------------

FOCAL_LENGTH_PX = 700  # placeholder, replace with real camera's calibrated value

AVERAGE_REAL_HEIGHTS = {
    'person': 1.7,
    'bike': 1.1,
    'car': 1.5,
    'bus': 3.0,
    'truck': 2.5,
    'light': 3.0,     # traffic light pole height, rough
    'hydrant': 0.75,
    'sign': 2.0,       # highly variable, least reliable estimate
}

def estimate_distance_meters(class_name, box, frame_height_px):
    """
    box: (x_center, y_center, w, h) normalized 0-1, as YOLO outputs
    frame_height_px: actual pixel height of the video frame (e.g. 512 for thermal)
    """
    if class_name not in AVERAGE_REAL_HEIGHTS:
        return None  # no reliable size assumption for this class

    _, _, _, h_normalized = box
    pixel_height = h_normalized * frame_height_px

    if pixel_height <= 0:
        return None

    real_height = AVERAGE_REAL_HEIGHTS[class_name]
    distance_m = (real_height * FOCAL_LENGTH_PX) / pixel_height
    return round(distance_m, 2)
'''