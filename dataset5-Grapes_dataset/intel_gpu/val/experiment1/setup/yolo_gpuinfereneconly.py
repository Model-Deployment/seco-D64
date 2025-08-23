# Import the libraries
import os
import cv2
import time
import numpy as np
import csv
from tqdm import tqdm
from openvino.runtime import Core
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

# Paths
model_path = "global_model_openvino_model/global_model.xml"
image_dir = "data/images"
output_dir = "outputs"
gt_dir = "data/labels"  # Ground truth in YOLO format

# Constants
# Unify the size
input_height, input_width = 640, 640
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
LABELS = [f"Grape {i}" for i in range(80)]  # Update if needed

# Initialization
os.makedirs(output_dir, exist_ok=True)
core = Core()
compiled_model = core.compile_model(model_path, "GPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Metrics accumulators
all_y_true, all_y_pred = [], []
inference_times = []

# Helper functions
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (input_width, input_height))
    normalized = resized / 255.0
    transposed = normalized.transpose(2, 0, 1)
    input_tensor = np.expand_dims(transposed, axis=0).astype(np.float32)
    return image, input_tensor

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

def postprocess(output, original_shape):
    detections = output[0].T
    orig_h, orig_w = original_shape[:2]
    scale_x = orig_w / input_width
    scale_y = orig_h / input_height

    boxes, scores, class_ids = [], [], []

    for det in detections:
        if len(det) < 5 or det[4] < CONFIDENCE_THRESHOLD:
            continue
        x, y, w, h, conf = det[:5]
        class_id = 0  # Use 0 unless the model predicts class IDs

        x1 = int((x - w / 2) * scale_x)
        y1 = int((y - h / 2) * scale_y)
        x2 = int((x + w / 2) * scale_x)
        y2 = int((y + h / 2) * scale_y)
        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

    final = []
    if isinstance(indices, (np.ndarray, list)) and len(indices) > 0:
        for i in indices:
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            final.append((boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], scores[idx], class_ids[idx]))
    return final

def load_ground_truth(image_file, image_shape):
    label_file = os.path.join(gt_dir, os.path.splitext(image_file)[0] + ".txt")
    if not os.path.exists(label_file):
        return []

    h, w = image_shape[:2]
    boxes = []
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x, y, bw, bh = map(float, parts)
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            boxes.append([x1, y1, x2, y2])
    return boxes

# Inference loop
for image_file in tqdm(sorted(os.listdir(image_dir))):
    if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(image_dir, image_file)
    image, tensor = preprocess_image(path)

    # Inference
    start = time.time()
    result = compiled_model([tensor])[output_layer]
    inference_times.append(time.time() - start)

    # Postprocess
    preds = postprocess(result, image.shape)

    # Ground truth
    gts = load_ground_truth(image_file, image.shape)

    # Matching predictions to GT
    y_true, y_pred = [], []
    matched_preds = set()

    for gt_box in gts:
        matched = False
        for i, (x1, y1, x2, y2, _, _) in enumerate(preds):
            if i in matched_preds:
                continue
            iou = compute_iou(gt_box, [x1, y1, x2, y2])
            if iou >= IOU_THRESHOLD:
                matched_preds.add(i)
                matched = True
                break
        y_true.append(1)
        y_pred.append(1 if matched else 0)

    # False positives
    for i in range(len(preds)):
        if i not in matched_preds:
            y_true.append(0)
            y_pred.append(1)

    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)

    # Draw predictions with label and confidence
    for (x1, y1, x2, y2, conf, class_id) in preds:
        label = f"{LABELS[class_id]}: {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out_path = os.path.join(output_dir, f"out_{image_file}")
    cv2.imwrite(out_path, image)

# === Final Metrics ===
print("\n--- Performance Summary ---")
print(f"Images Processed: {len(inference_times)}") # Output the number  of images handled
print(f"Avg Inference Time: {np.mean(inference_times)*1000:.2f} ms")

if all_y_true:
    precision = precision_score(all_y_true, all_y_pred)
    recall = recall_score(all_y_true, all_y_pred)
    f1 = f1_score(all_y_true, all_y_pred)
    avg_prec = average_precision_score(all_y_true, all_y_pred)

    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1 Score:            {f1:.4f}")
    print(f"Average Precision:   {avg_prec:.4f}")
else:
    print("No ground truth found — cannot compute metrics.")
    precision = recall = f1 = avg_prec = "N/A"

# === Save CSV Results ===
csv_path = os.path.join(os.getcwd(), "results.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Images Processed", len(inference_times)])
    writer.writerow(["Average Inference Time (ms)", f"{np.mean(inference_times)*1000:.2f}"])
    writer.writerow(["Precision", precision])
    writer.writerow(["Recall", recall])
    writer.writerow(["F1 Score", f1])
    writer.writerow(["Average Precision", avg_prec])
