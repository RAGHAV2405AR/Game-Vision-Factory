import os
import yaml
import csv

def create_data_yaml(dataset_dir: str):
    data = {
        "path": dataset_dir,
        "train": "images/train",
        "val": "images/train",
        "names": []
    }

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f)

    return yaml_path


def yolo_to_csv(dataset_dir: str, csv_path: str):
    labels_dir = os.path.join(dataset_dir, "labels", "train")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name",
            "class_id",
            "x_center",
            "y_center",
            "width",
            "height"
        ])

        for label_file in os.listdir(labels_dir):
            with open(os.path.join(labels_dir, label_file)) as lf:
                for line in lf:
                    cls, x, y, w, h = line.split()
                    writer.writerow([
                        label_file.replace(".txt", ".jpg"),
                        cls, x, y, w, h
                    ])
