"""
Offline checks for the alert pipeline - no camera, no model, runs in a second.

    python test_alert.py

Uses fake thermal + RGB detections, merges them, and feeds them through the
tracker with simulated timestamps to confirm the alert rules behave.
Format: (class_name, confidence, (x_center, y_center, w, h)), normalized 0-1.
"""
from alert_logic import ObjectTracker, filter_relevant, format_alert
from merge_detections import merge_detections

FRAME_H = 480


def frame(tracker, dets, t):
    return tracker.get_alerts(tracker.update(dets, FRAME_H, now=t), now=t)


def test_merge_and_first_alert():
    thermal = [('person', 0.85, (0.2, 0.6, 0.15, 0.80))]   # close, left
    rgb = [('person', 0.40, (0.21, 0.6, 0.15, 0.80)),      # same person, weaker
           ('car', 0.55, (0.5, 0.5, 0.3, 0.05))]           # far away
    merged = merge_detections(thermal, rgb)
    assert len(merged) == 2, merged                        # duplicate person removed

    alerts = frame(ObjectTracker(), merged, t=0.0)
    assert [a['class'] for a in alerts] == ['person'], alerts
    assert alerts[0]['direction'] == 'left'
    print("first alert:", format_alert(alerts[0]))


def test_no_repeat_unless_closer():
    tr = ObjectTracker()
    person = lambda h: [('person', 0.9, (0.5, 0.5, 0.2, h))]
    assert frame(tr, person(0.80), t=0.0)                  # ~1.66 m -> alert
    assert not frame(tr, person(0.80), t=0.5)              # same distance -> silent
    assert not frame(tr, person(0.82), t=2.0)              # barely closer -> silent
    assert frame(tr, person(1.00), t=3.0)                  # ~1.33 m, 0.3 m closer -> alert


def test_class_names_and_filtering():
    dets = [('cycle', 0.9, (0.5, 0.5, 0.1, 0.1)),          # trained-model name
            ('stop sign', 0.9, (0.5, 0.5, 0.1, 0.1)),      # COCO name
            ('laptop', 0.9, (0.5, 0.5, 0.9, 0.9))]         # irrelevant indoors object
    names = [d[0] for d in filter_relevant(dets)]
    assert names == ['bicycle', 'sign'], names


def test_weak_classes_need_more_confidence():
    dets = [('cycle', 0.54, (0.4, 0.7, 0.3, 0.5)),         # the false 'cycle 0.54' from the phone test
            ('cycle', 0.75, (0.4, 0.7, 0.3, 0.5)),         # confident bicycle -> kept
            ('person', 0.55, (0.5, 0.6, 0.3, 0.6))]        # strong class, normal 0.5 threshold
    kept = [(n, c) for n, c, _ in filter_relevant(dets)]
    assert kept == [('bicycle', 0.75), ('person', 0.55)], kept


def test_fallback_without_height_and_ranking():
    tr = ObjectTracker()
    dets = [('dog', 0.9, (0.8, 0.7, 0.2, 0.6)),            # no known height -> 'close' bucket
            ('car', 0.9, (0.3, 0.5, 0.4, 0.9))]            # ~1.3 m
    alerts = frame(tr, dets, t=0.0)
    assert [a['class'] for a in alerts] == ['car', 'dog'], alerts
    assert alerts[1]['distance_m'] is None


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f"PASS {name}")
