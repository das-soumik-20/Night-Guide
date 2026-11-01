# alert_logic.py
"""
Turns raw detections into spoken/printed alerts for the user.

Pipeline per frame:
    detections -> normalize class names -> drop irrelevant classes
               -> ObjectTracker.update()   (persistent IDs + distance + direction)
               -> ObjectTracker.get_alerts() (only close objects, de-duplicated, ranked)

A detection is always a tuple: (class_name, confidence, (x_center, y_center, w, h))
with the box normalized to 0-1, exactly as YOLO's `box.xywhn` gives it.
"""
import time

# ---------------------------------------------------------------------
# Camera calibration
# Wrong focal length = wrong metres, even if everything else works.
# Calibrate: photograph an object of known height H (m) at a known distance D (m),
# measure its pixel height P, then FOCAL_LENGTH_PX = P * D / H.
# ---------------------------------------------------------------------
FOCAL_LENGTH_PX = 376  # placeholder - replace with the calibrated value for your camera

# ---------------------------------------------------------------------
# Class names
# Our trained FLIR models (configs/*.yaml) and the stock COCO yolov8n.pt use
# different names for the same things. Everything is mapped to one canonical
# name so the rest of the logic only has to know one vocabulary.
# ---------------------------------------------------------------------
CLASS_ALIASES = {
    # trained FLIR model names -> canonical
    'cycle': 'bicycle',
    'bike': 'motorcycle',
    'light': 'traffic light',
    'hydrant': 'fire hydrant',
    'sign': 'sign',
    # COCO names -> canonical (only where they differ)
    'stop sign': 'sign',
}


def normalize_class(class_name):
    return CLASS_ALIASES.get(class_name, class_name)


# Only these (canonical) classes are real obstacles for a pedestrian.
# Everything else (laptop, bed, remote, cell phone, ...) is ignored.
RELEVANT_CLASSES = {
    'person', 'bicycle', 'motorcycle', 'car', 'bus', 'truck',
    'traffic light', 'sign', 'fire hydrant', 'bench', 'dog',
    'pothole',  # future custom class
}

# Rough real-world heights in metres, used for distance estimation.
AVERAGE_REAL_HEIGHTS = {
    'person': 1.7,
    'bicycle': 1.1,
    'motorcycle': 1.1,
    'car': 1.5,
    'bus': 3.0,
    'truck': 2.5,
    'traffic light': 3.0,   # pole height, rough
    'fire hydrant': 0.75,
    'sign': 2.0,            # highly variable, least reliable estimate
}

# Minimum confidence before a detection counts at all.
# Classes our trained models are weak on (test mAP50 < 0.3, see results/) need
# more confidence, otherwise they produce false alarms - e.g. a seated person
# detected as 'cycle 0.54'. Raise/lower these as the models improve.
DEFAULT_MIN_CONFIDENCE = 0.5
MIN_CONFIDENCE = {
    'bicycle': 0.7,
    'truck': 0.7,
    'sign': 0.7,
    'fire hydrant': 0.7,
    'traffic light': 0.7,
}

# How dangerous a class is when it is close. Higher = announced first.
RISK_LEVELS = {
    'pothole': 3,
    'person': 2,
    'bicycle': 2,
    'motorcycle': 2,
    'car': 2,
    'bus': 2,
    'truck': 2,
}
DEFAULT_RISK = 1

# ---------------------------------------------------------------------
# Alert tuning
# ---------------------------------------------------------------------
DANGER_DISTANCE_M = 2.0          # only alert inside this range
RE_ALERT_CLOSER_MARGIN_M = 0.3   # an object must get this much closer to re-alert
MIN_TIME_BETWEEN_ALERTS_S = 1.0  # debounce, avoids double-firing on noisy frames
MAX_MATCH_DIST = 0.15            # normalized centroid distance to count as "same object"
TRACK_TIMEOUT_S = 2.0            # forget an object if unseen this long


def estimate_direction(box):
    x_center, _, _, _ = box
    if x_center < 0.35:
        return 'left'
    elif x_center > 0.65:
        return 'right'
    return 'center'


def estimate_proximity(box):
    """Fallback bucket for classes without a known real-world height."""
    _, _, _, h = box
    if h > 0.5:
        return 'close'
    elif h > 0.25:
        return 'medium'
    return 'far'


def estimate_distance_meters(class_name, box, frame_height_px):
    """Pinhole camera model: distance = real_height * focal_length / pixel_height."""
    class_name = normalize_class(class_name)
    if class_name not in AVERAGE_REAL_HEIGHTS:
        return None
    _, _, _, h_normalized = box
    pixel_height = h_normalized * frame_height_px
    if pixel_height <= 0:
        return None
    return round((AVERAGE_REAL_HEIGHTS[class_name] * FOCAL_LENGTH_PX) / pixel_height, 2)


def passes_confidence(class_name, confidence):
    """Per-class confidence check (see MIN_CONFIDENCE)."""
    name = normalize_class(class_name)
    return confidence >= MIN_CONFIDENCE.get(name, DEFAULT_MIN_CONFIDENCE)


def filter_relevant(detections):
    """Normalize class names, drop non-obstacles and low-confidence guesses."""
    out = []
    for class_name, confidence, box in detections:
        name = normalize_class(class_name)
        if name in RELEVANT_CLASSES and passes_confidence(name, confidence):
            out.append((name, confidence, tuple(box)))
    return out


def alert_score(alert):
    """Higher = more urgent. Risk of the class plus a bonus for being closer."""
    risk = RISK_LEVELS.get(alert['class'], DEFAULT_RISK)
    distance_bonus = 0
    if alert.get('distance_m') is not None:
        distance_bonus = max(0, 5 - alert['distance_m'])
    elif alert.get('proximity') == 'close':
        distance_bonus = 3
    return risk + distance_bonus


def rank_alerts(alerts):
    return sorted(alerts, key=alert_score, reverse=True)


def format_alert(alert):
    where = f"{alert['distance_m']}m away" if alert['distance_m'] is not None else 'close'
    return f"{alert['class']} to your {alert['direction']}, {where}"


class ObjectTracker:
    """
    Gives each detected object a persistent ID across frames by matching
    same-class detections to the nearest previous position. Without this,
    every frame is anonymous and there's no way to tell 'the same car'
    from a brand new one two frames later.
    """

    def __init__(self, danger_distance_m=DANGER_DISTANCE_M):
        self.tracks = {}
        self._next_id = 0
        self.danger_distance_m = danger_distance_m

    def _new_id(self):
        self._next_id += 1
        return self._next_id

    def _measure(self, class_name, box, frame_height_px):
        return {
            'cx': box[0], 'cy': box[1],
            'distance_m': estimate_distance_meters(class_name, box, frame_height_px),
            'proximity': estimate_proximity(box),
            'direction': estimate_direction(box),
        }

    def update(self, detections, frame_height_px, now=None):
        """Feed one frame of raw detections. Returns the tracks seen this frame."""
        now = time.time() if now is None else now
        detections = filter_relevant(detections)
        unmatched = list(range(len(detections)))
        touched_ids = []

        # match each existing track to the nearest detection of the same class
        for track_id, track in list(self.tracks.items()):
            best_i, best_dist = None, None
            for i in unmatched:
                class_name, _, box = detections[i]
                if class_name != track['class_name']:
                    continue
                dist = ((box[0] - track['cx']) ** 2 + (box[1] - track['cy']) ** 2) ** 0.5
                if dist < MAX_MATCH_DIST and (best_dist is None or dist < best_dist):
                    best_i, best_dist = i, dist

            if best_i is not None:
                class_name, confidence, box = detections[best_i]
                track.update(self._measure(class_name, box, frame_height_px))
                track.update({'confidence': confidence, 'last_seen': now})
                unmatched.remove(best_i)
                touched_ids.append(track_id)

        # anything left over is a newly seen object
        for i in unmatched:
            class_name, confidence, box = detections[i]
            track_id = self._new_id()
            self.tracks[track_id] = {
                'id': track_id,
                'class_name': class_name,
                'confidence': confidence,
                'last_seen': now,
                'last_alert_time': None,
                'last_alert_distance': None,
                **self._measure(class_name, box, frame_height_px),
            }
            touched_ids.append(track_id)

        # drop objects we haven't seen in a while
        for track_id in list(self.tracks):
            if now - self.tracks[track_id]['last_seen'] > TRACK_TIMEOUT_S:
                del self.tracks[track_id]

        return [self.tracks[tid] for tid in touched_ids if tid in self.tracks]

    def _is_close(self, track):
        if track['distance_m'] is not None:
            return track['distance_m'] <= self.danger_distance_m
        return track['proximity'] == 'close'

    def get_alerts(self, active_tracks, now=None):
        """
        Alerts for objects that are close AND either seen for the first time
        or getting meaningfully closer. Most urgent first.
        """
        now = time.time() if now is None else now
        alerts = []

        for track in active_tracks:
            if not self._is_close(track):
                continue

            distance_m = track['distance_m']
            last_time = track['last_alert_time']
            last_dist = track['last_alert_distance']

            if last_time is None:
                fire = True
            elif now - last_time < MIN_TIME_BETWEEN_ALERTS_S:
                fire = False
            elif (distance_m is not None and last_dist is not None
                  and (last_dist - distance_m) >= RE_ALERT_CLOSER_MARGIN_M):
                fire = True
            else:
                fire = False

            if fire:
                alerts.append({
                    'id': track['id'],
                    'class': track['class_name'],
                    'direction': track['direction'],
                    'distance_m': distance_m,
                    'proximity': track['proximity'],
                })
                track['last_alert_time'] = now
                track['last_alert_distance'] = distance_m

        return rank_alerts(alerts)
