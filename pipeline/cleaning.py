import os
import cv2
import numpy as np


def clean_frames(input_dir, output_dir, dup_thresh=0.98):
    if not os.path.exists(input_dir):
        raise RuntimeError(f"Input frames directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    prev_img = None
    kept = 0

    for img_name in sorted(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        if prev_img is not None:
            diff = cv2.absdiff(prev_img, img)
            similarity = 1 - (diff.mean() / 255)

            if similarity > dup_thresh:
                continue

        cv2.imwrite(os.path.join(output_dir, img_name), img)
        prev_img = img
        kept += 1

    print(f"Cleaned frames saved: {kept}")


