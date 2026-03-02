import os
import yaml
import csv
import shutil
import random


# These are the 80 object names that yolov8n.pt was trained to detect.
# Position in the list = class ID number.
# So ID 0 = "person", ID 1 = "bicycle", ID 2 = "car", and so on.
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def make_val_split(dataset_dir):
    """
    Takes 20% of training images and moves them into a separate val folder.
    YOLO needs two separate folders - one to train on, one to test on.
    Without this, YOLO was validating on the same images it trained on (cheating).
    """

    # Define the 4 folders we are working with
    train_images_folder = os.path.join(dataset_dir, "images", "train")
    train_labels_folder = os.path.join(dataset_dir, "labels", "train")
    val_images_folder   = os.path.join(dataset_dir, "images", "val")
    val_labels_folder   = os.path.join(dataset_dir, "labels", "val")

    # Create the val folders if they don't exist yet
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)

    # If val folder already has images, don't do anything
    # This stops files being moved twice if you re-run the pipeline
    if len(os.listdir(val_images_folder)) > 0:
        print("Val folder already has images. Skipping split.")
        return

    # Get a list of all jpg images in the train folder
    all_images = []
    for filename in os.listdir(train_images_folder):
        if filename.endswith(".jpg"):
            all_images.append(filename)

    # If there are no images, nothing to do
    if len(all_images) == 0:
        print("No training images found.")
        return

    # Shuffle the list randomly (seed 42 means it shuffles the same way every time)
    random.seed(42)
    random.shuffle(all_images)

    # Take 20% for val - at least 1 file minimum
    number_of_val_images = max(1, int(len(all_images) * 0.2))
    val_images = all_images[:number_of_val_images]

    for image_filename in val_images:

        # Get the name without extension e.g. "000042.jpg" -> "000042"
        name_without_ext = image_filename.replace(".jpg", "")
        label_filename   = name_without_ext + ".txt"

        # Move the image file
        old_image_path = os.path.join(train_images_folder, image_filename)
        new_image_path = os.path.join(val_images_folder,   image_filename)

        if os.path.exists(old_image_path):
            shutil.move(old_image_path, new_image_path)

        # Move the label file (might not exist if frame had no detections)
        old_label_path = os.path.join(train_labels_folder, label_filename)
        new_label_path = os.path.join(val_labels_folder,   label_filename)

        if os.path.exists(old_label_path):
            shutil.move(old_label_path, new_label_path)

    train_count = len(all_images) - number_of_val_images
    print("Split done. Train:", train_count, "| Val:", number_of_val_images)


def create_data_yaml(dataset_dir):
    """
    Creates the data.yaml file that YOLO reads before training.
    This file tells YOLO:
      - Where the dataset folder is
      - Where the train images are
      - Where the val images are
      - How many classes there are
      - What the class names are
    """

    #  make the val split first
    make_val_split(dataset_dir)


    
    full_dataset_path = os.path.abspath(dataset_dir)

    #  build the data dictionary that will be saved as YAML
    data = {
        "path":  full_dataset_path,   
        "train": "images/train",      
        "val":   "images/val",       
        "nc":    len(COCO_NAMES),     
        "names": COCO_NAMES           
    }
    
    yaml_file_path = os.path.join(full_dataset_path, "data.yaml")

    with open(yaml_file_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    print("data.yaml created at:", yaml_file_path)
    return yaml_file_path


def yolo_to_csv(dataset_dir, csv_path):
    """
    Reads all the YOLO label .txt files and saves them into one CSV file.
    This makes it easy to open your annotations in Excel or pandas.

    Each row in the CSV = one bounding box detection.
    """

    all_rows = []

    # Go through both train and val label folders
    for split in ["train", "val"]:

        labels_folder = os.path.join(dataset_dir, "labels", split)

        # Skip if this folder doesn't exist
        if not os.path.exists(labels_folder):
            print("Folder not found, skipping:", labels_folder)
            continue

        # Read every .txt label file in the folder
        for label_filename in os.listdir(labels_folder):

            if not label_filename.endswith(".txt"):
                continue

            # The image has the same name but .jpg extension
            image_filename = label_filename.replace(".txt", ".jpg")

            label_file_path = os.path.join(labels_folder, label_filename)

            with open(label_file_path, "r") as f:
                for line in f:

                    # Each line looks like: "0 0.52 0.34 0.12 0.08"
                    # meaning: class_id x_center y_center width height
                    parts = line.strip().split()

                    # Skip empty or broken lines
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width    = float(parts[3])
                    height   = float(parts[4])

                    # Look up the class name from our list
                    # If the ID is somehow out of range, use a fallback name
                    if class_id < len(COCO_NAMES):
                        class_name = COCO_NAMES[class_id]
                    else:
                        class_name = "unknown"

                    # Add this detection as one row
                    all_rows.append([
                        image_filename,
                        split,
                        class_id,
                        class_name,
                        x_center,
                        y_center,
                        width,
                        height
                    ])

    # Write all rows to the CSV file
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Write the header row first
        writer.writerow([
            "image_name",
            "split",
            "class_id",
            "class_name",
            "x_center",
            "y_center",
            "width",
            "height"
        ])

        # Write all the data rows
        for row in all_rows:
            writer.writerow(row)

    print("CSV saved at:", csv_path, "| Total detections:", len(all_rows))