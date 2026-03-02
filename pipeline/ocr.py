import cv2
import csv
import os
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def read_text_from_frame(img, min_confidence=60):
    image_height = img.shape[0]
    image_width  = img.shape[1]

    # Tesseract reads text much better on grayscale images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Scale the image up 2x before OCR
    # Game UI text is often very small (like 12-16px).
    # Tesseract struggles with tiny text. Making it bigger helps a lot.
    scale = 2
    bigger = cv2.resize(gray, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC)
    #INTER_Cubic = best for upsacling (4*4)

    bigger = cv2.GaussianBlur(bigger, (3, 3), 0)
    
    # config='--psm 11' tells Tesseract how to scan the image:
    # psm 11 = "sparse text" mode - looks for text anywhere in the frame
    raw_data = pytesseract.image_to_data(
        bigger,
        output_type=pytesseract.Output.DICT,
        config="--psm 11"
    )

    
    found_text = []

    number_of_results = len(raw_data["text"])

    for i in range(number_of_results):

        word = raw_data["text"][i].strip()
        confidence = int(raw_data["conf"][i])

        # Skip blank results
        if word == "":
            continue

        # Skip low confidence results
        if confidence < min_confidence:
            continue

        # Get the bounding box of this word in the SCALED up image
        x_in_scaled = raw_data["left"][i]
        y_in_scaled = raw_data["top"][i]
        w_in_scaled = raw_data["width"][i]
        h_in_scaled = raw_data["height"][i]

        # Scale the coordinates back down to the original image size
        x = x_in_scaled // scale
        y = y_in_scaled // scale
        w = w_in_scaled // scale
        h = h_in_scaled // scale

        # Normalize to 0.0-1.0 range 
        x_center_normalized = (x + w / 2) / image_width
        y_center_normalized = (y + h / 2) / image_height
        width_normalized    = w / image_width
        height_normalized   = h / image_height

        found_text.append({
            "text":       word,
            "confidence": confidence,
            "x_center":   round(x_center_normalized, 6),
            "y_center":   round(y_center_normalized, 6),
            "width":      round(width_normalized, 6),
            "height":     round(height_normalized, 6)
        })

    return found_text


def run_ocr_on_dataset(images_dir, output_csv_path, min_confidence=60):
    """
    Runs OCR on every image in a folder and saves all the text it finds to a CSV.

    This gives you a second CSV alongside your YOLO annotations CSV.
    You can open it in Excel to see what text appears in your game frames
    and where on screen it appears.

    Parameters:
        images_dir       - folder containing your .jpg frames
        output_csv_path  - where to save the resulting CSV file
        min_confidence   - only save text Tesseract is this % sure about
    """

    all_rows = []

    # Get all jpg files in the folder, sorted by name
    all_filenames = []
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg"):
            all_filenames.append(filename)

    all_filenames.sort()

    total = len(all_filenames)
    print("Running OCR on", total, "images...")

    for count, filename in enumerate(all_filenames):

        image_path = os.path.join(images_dir, filename)
        img = cv2.imread(image_path)

        if img is None:
            print("Could not read image, skipping:", filename)
            continue

        # Get all text found in this frame
        detections = read_text_from_frame(img, min_confidence)

        # Add each detection as a row
        for detection in detections:
            all_rows.append([
                filename,
                detection["text"],
                detection["confidence"],
                detection["x_center"],
                detection["y_center"],
                detection["width"],
                detection["height"]
            ])

        # Print progress every 20 images so you know it's working
        if count % 20 == 0:
            print("Progress:", count, "/", total)

    # Write everything to CSV
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header row
        writer.writerow([
            "image_name",
            "text",
            "ocr_confidence",
            "x_center",
            "y_center",
            "width",
            "height"
        ])

        # Data rows
        for row in all_rows:
            writer.writerow(row)

    print("OCR done. CSV saved at:", output_csv_path)
    print("Total text detections found:", len(all_rows))