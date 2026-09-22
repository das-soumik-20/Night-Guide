# alert_logic.py
import time

# ---------------------------------------------------------------------
# Calibrate this for your real camera (see original comment for method).
# Wrong focal length = wrong meters, even if everything else works.
# ---------------------------------------------------------------------
FOCAL_LENGTH_PX = 376  # placeholder — replace with calibrated value

AVERAGE_REAL_HEIGHTS = {
    'person': 1.7,
    'bike': 1.1,
    'car': 1.5,
    'bus': 3.0,
    'truck': 2.5,
    'light': 3.0,
    'hydrant': 0.75,
    'sign': 2.0,
}

DANGER_DISTANCE_M = 2.0          # only alert inside this range
RE_ALERT_CLOSER_MARGIN_M = 0.3   # must close by this much to re-alert
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


def estimate_distance_meters(class_name, box, frame_height_px):
    if class_name not in AVERAGE_REAL_HEIGHTS:
        return None  # no reliable size assumption for this class
    _, _, _, h_normalized = box
    pixel_height = h_normalized * frame_height_px
    if pixel_height <= 0:
        return None
    real_height = AVERAGE_REAL_HEIGHTS[class_name]
    return round((real_height * FOCAL_LENGTH_PX) / pixel_height, 2)


class ObjectTracker:
    """
    Gives each detected object a persistent ID across frames by matching
    same-class detections to the nearest previous position. Without this,
    every frame is anonymous and there's no way to tell 'the same car'
    from a brand new one two frames later.
    """
    def __init__(self):
        self.tracks = {}
        self._next_id = 0

    def _new_id(self):
        self._next_id += 1
        return self._next_id

    def update(self, detections, frame_height_px):
        now = time.time()
        unmatched = list(range(len(detections)))
        touched_ids = []

        # try to match each existing track to a detection of the same class
        for track_id, track in list(self.tracks.items()):
            best_i, best_dist = None, None
            for i in unmatched:
                class_name, confidence, box = detections[i]
                if class_name != track['class_name']:
                    continue
                dist = ((box[0] - track['cx']) ** 2 + (box[1] - track['cy']) ** 2) ** 0.5
                if dist < MAX_MATCH_DIST and (best_dist is None or dist < best_dist):
                    best_i, best_dist = i, dist

            if best_i is not None:
                class_name, confidence, box = detections[best_i]
                track.update({
                    'cx': box[0], 'cy': box[1],
                    'confidence': confidence,
                    'distance_m': estimate_distance_meters(class_name, box, frame_height_px),
                    'direction': estimate_direction(box),
                    'last_seen': now,
                })
                unmatched.remove(best_i)
                touched_ids.append(track_id)

        # anything left over is a newly seen object
        for i in unmatched:
            class_name, confidence, box = detections[i]
            track_id = self._new_id()
            self.tracks[track_id] = {
                'id': track_id,
                'class_name': class_name,
                'cx': box[0], 'cy': box[1],
                'confidence': confidence,
                'distance_m': estimate_distance_meters(class_name, box, frame_height_px),
                'direction': estimate_direction(box),
                'last_seen': now,
                'last_alert_time': None,
                'last_alert_distance': None,
            }
            touched_ids.append(track_id)

        # drop objects we haven't seen in a while
        for track_id in list(self.tracks.keys()):
            if now - self.tracks[track_id]['last_seen'] > TRACK_TIMEOUT_S:
                del self.tracks[track_id]

        return [self.tracks[tid] for tid in touched_ids if tid in self.tracks]

    def get_alerts(self, active_tracks):
        """Only returns alerts for objects that are dangerously close AND
        either being seen for the first time or closing distance meaningfully."""
        alerts = []
        now = time.time()

        for track in active_tracks:
            distance_m = track['distance_m']
            if distance_m is None or distance_m > DANGER_DISTANCE_M:
                continue

            last_time = track['last_alert_time']
            last_dist = track['last_alert_distance']

            if last_time is None:
                fire = True
            elif now - last_time < MIN_TIME_BETWEEN_ALERTS_S:
                fire = False
            elif last_dist is not None and (last_dist - distance_m) >= RE_ALERT_CLOSER_MARGIN_M:
                fire = True
            else:
                fire = False

            if fire:
                alerts.append({
                    'id': track['id'],
                    'class': track['class_name'],
                    'direction': track['direction'],
                    'distance_m': distance_m,
                })
                track['last_alert_time'] = now
                track['last_alert_distance'] = distance_m

        return alerts