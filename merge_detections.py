# merge_detections.py

def iou(box1, box2):
    """Basic IoU between two normalized (x_center, y_center, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    box1_x1, box1_y1 = x1 - w1/2, y1 - h1/2
    box1_x2, box1_y2 = x1 + w1/2, y1 + h1/2
    box2_x1, box2_y1 = x2 - w2/2, y2 - h2/2
    box2_x2, box2_y2 = x2 + w2/2, y2 + h2/2

    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def merge_detections(thermal_dets, rgb_dets, thermal_weight=1.0, rgb_weight=0.6, iou_threshold=0.5):
    """
    thermal_dets / rgb_dets: list of (class_name, confidence, box)
    Thermal is weighted higher since it's your primary sensor in low light.
    """
    weighted = [(c, conf * thermal_weight, box) for c, conf, box in thermal_dets]
    weighted += [(c, conf * rgb_weight, box) for c, conf, box in rgb_dets]

    weighted.sort(key=lambda d: d[1], reverse=True)  # highest confidence first

    final = []
    while weighted:
        best = weighted.pop(0)
        final.append(best)
        weighted = [
            d for d in weighted
            if not (d[0] == best[0] and iou(d[2], best[2]) > iou_threshold)
        ]
    return final