from ultralytics import YOLO
import matplotlib.pyplot as plt

def evaluate_model(model, metrics, split = 'val'):
    yolo_model = YOLO(model)
    results = yolo_model.val(data=metrics, split=split)
    model_name = model.split('/')[2]
    all_class_names = list(results.names.values())  # all 9, in order
    present_class_ids = results.ap_class_index        # which classes actually had ground truth
    per_class_map50 = results.box.ap50                # matches present_class_ids, same length

    # Build a full-length array, 0 for classes with no test instances
    map50_by_class = {cid: ap for cid, ap in zip(present_class_ids, per_class_map50)}
    full_map50 = [map50_by_class.get(i, 0.0) for i in range(len(all_class_names))]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(all_class_names, full_map50, color='steelblue')
    plt.ylabel('mAP50')
    plt.title(f'Per-Class mAP50 on Held-Out Test Set (Overall mAP50: {results.box.map50:.3f})')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    for i, (bar, val) in enumerate(zip(bars, full_map50)):
        label = f'{val:.2f}' if i in present_class_ids else 'no data'
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, label, ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"results/{model_name}_{split}_results_chart.png", dpi=300)
    plt.close()

    return results

if __name__ == "__main__":
    evaluate_model('runs/detect/thermal_full/weights/best.pt', 'configs/thermal.yaml' )