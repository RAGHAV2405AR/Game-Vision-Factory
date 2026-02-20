from ultralytics import YOLO
import os
import cv2
import shutil

def auto_label(*, images_dir, dataset_dir, conf_thresh=0.35):
    images_out = os.path.join(dataset_dir, "images", "train")
    labels_out = os.path.join(dataset_dir, "labels", "train")

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    model = YOLO("yolov8n.pt")

    images = sorted(os.listdir(images_dir))
    if not images:
        raise RuntimeError("No frames found for labeling")

    print(f"[INFO] Auto-labeling {len(images)} frames")

    for idx, img_name in enumerate(images):
        src_img = os.path.join(images_dir, img_name)
        dst_img = os.path.join(images_out, img_name)

        img = cv2.imread(src_img)
        if img is None:
            continue

        shutil.copy(src_img, dst_img)

        results = model(img, conf=conf_thresh, verbose=False)[0]
        h, w, _ = img.shape

        lines = []
        for box in results.boxes:
            cls = int(box.cls)
            x, y, bw, bh = box.xywh[0]
            lines.append(
                f"{cls} {x/w:.6f} {y/h:.6f} {bw/w:.6f} {bh/h:.6f}"
            )

        if lines:
            label_path = os.path.join(
                labels_out,
                os.path.splitext(img_name)[0] + ".txt"
            )
            with open(label_path, "w") as f:
                f.write("\n".join(lines))

        if idx % 20 == 0:
            print(f"[YOLO] {idx}/{len(images)} labeled")

    print("[INFO] Auto-labeling complete")
