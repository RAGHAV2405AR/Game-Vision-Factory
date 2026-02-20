import os
import csv


def yolo_dataset_to_csv(dataset_dir: str, output_csv: str):
    """
    Converts a YOLO dataset into a single CSV file.
    """

    rows = []
    labels_base = os.path.join(dataset_dir, "labels")

    for split in ["train", "val"]:
        split_dir = os.path.join(labels_base, split)

        if not os.path.exists(split_dir):
            continue

        for label_file in os.listdir(split_dir):
            if not label_file.endswith(".txt"):
                continue

            image_name = label_file.replace(".txt", ".jpg")
            label_path = os.path.join(split_dir, label_file)

            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id, x, y, w, h = parts

                    rows.append([
                        image_name,
                        split,
                        int(class_id),
                        float(x),
                        float(y),
                        float(w),
                        float(h),
                    ])

    # Write CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "image_name",
            "split",
            "class_id",
            "x_center",
            "y_center",
            "width",
            "height"
        ])
        writer.writerows(rows)

    print(f"CSV saved to: {output_csv}")
