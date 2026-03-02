import cv2
import os
import sys
from dataset import COCO_NAMES

COLORS = [
    (255, 0,   0), (0, 255,   0), (0,   0, 255), (255, 255,   0),
    (255, 0, 255), (0, 255, 255), (128,  0,   0), (0, 128,   0),
    (0,   0, 128), (128, 128, 0), (128,  0, 128), (0, 128, 128),
]


def visualise(base_dir):
    IMAGES_FOLDER = os.path.join(base_dir, "frames_clean")
    LABELS_FOLDER = os.path.join(base_dir, "dataset", "labels", "train")
    OUTPUT_FOLDER = os.path.join(base_dir, "visualized")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    all_images = sorted([f for f in os.listdir(IMAGES_FOLDER) if f.endswith(".jpg")])
    print("Drawing boxes on", len(all_images), "frames...")

    for img_filename in all_images:

        img        = cv2.imread(os.path.join(IMAGES_FOLDER, img_filename))
        label_path = os.path.join(LABELS_FOLDER, img_filename.replace(".jpg", ".txt"))

        if img is None:
            continue

        img_h = img.shape[0]
        img_w = img.shape[1]

        if not os.path.exists(label_path):
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, img_filename), img)
            continue

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width    = float(parts[3])
                height   = float(parts[4])

                cx = int(x_center * img_w)
                cy = int(y_center * img_h)
                bw = int(width    * img_w)
                bh = int(height   * img_h)

                x1 = cx - bw // 2
                y1 = cy - bh // 2
                x2 = cx + bw // 2
                y2 = cy + bh // 2

                color      = COLORS[class_id % len(COLORS)]
                class_name = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else "class_" + str(class_id)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                text_w = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
                cv2.putText(img, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(os.path.join(OUTPUT_FOLDER, img_filename), img)

    print("Visualisation done. Saved to:", OUTPUT_FOLDER)


# Still works as a script too
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualise_labels.py <video_id>")
        sys.exit(1)
    video_id = sys.argv[1]
    visualise(os.path.join("data", "runs", video_id))