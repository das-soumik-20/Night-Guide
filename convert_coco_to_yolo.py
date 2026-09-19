import json
from pathlib import Path

KEEP_CLASSES = {
    1: (0, 'person'),
    2: (1, 'cycle'),
    3: (2, 'car'),
    4: (3, 'bike'),
    6: (4, 'bus'),
    8: (5, 'truck'),
    10: (6, 'light'),
    11: (7, 'hydrant'),
    12: (8, 'sign'),
}

def convert(image_dir, labels_out_dir):
    image_dir = Path(image_dir)
    labels_out = Path(labels_out_dir)
    labels_out.mkdir(parents=True, exist_ok=True)

    with open(image_dir / 'coco.json') as f:
        data = json.load(f)

    images_by_id = {img['id']: img for img in data['images']}

    anns_by_image = {}
    for ann in data['annotations']:
        if ann['category_id'] not in KEEP_CLASSES:
            continue
        anns_by_image.setdefault(ann['image_id'], []).append(ann)

    written = 0
    for image_id, anns in anns_by_image.items():
        img_info = images_by_id[image_id]
        img_w, img_h = img_info['width'], img_info['height']
        file_name = Path(img_info['file_name']).name

        label_lines = []
        for ann in anns:
            class_idx,_  = KEEP_CLASSES[ann['category_id']]
            x, y, w, h = ann['bbox']
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            label_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        label_path = labels_out / (Path(file_name).stem + '.txt')
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))
        written += 1

    print(f"{labels_out}: {written} label files written")


if __name__ == '__main__':
    # convert('data/left_overs/images_thermal_train', 'data/thermal/labels/train')
    # convert('data/left_overs/images_thermal_val', 'data/thermal/labels/val')
    # convert('data/left_overs/images_rgb_train', 'data/rgb/labels/train')
    # convert('data/left_overs/images_rgb_val', 'data/rgb/labels/val')
    convert('data/left_overs/video_thermal_test', 'data/thermal/labels/test')