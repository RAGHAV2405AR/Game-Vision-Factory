import streamlit as st
import os
import re
import zipfile

from pipeline.video import download_video
from pipeline.frames import extract_frames
from pipeline.cleaning import clean_frames
from pipeline.labeling import auto_label
from pipeline.dataset import create_data_yaml, yolo_to_csv
from pipeline.train import train_model
from pipeline.ocr import run_ocr_on_dataset
from pipeline.visualise_labels import visualise
if "busy" not in st.session_state:
    st.session_state.busy = False

if "dataset_ready" not in st.session_state:
    st.session_state.dataset_ready = False

if "model_ready" not in st.session_state:
    st.session_state.model_ready = False

if "base_dir" not in st.session_state:
    st.session_state.base_dir = None

if "active_video" not in st.session_state:
    st.session_state.active_video = None
    

def get_video_id(url: str):
    m = re.search(r"v=([^&]+)", url)
    return m.group(1) if m else "unknown"

def zip_dir(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname)

#UI
st.title("Game Vision Factory")
st.caption("Gameplay video → YOLO dataset → CSV → Model")

youtube_url = st.text_input(
    "YouTube Gameplay URL",
    disabled=st.session_state.busy
)

fps = st.slider(
    "Frame extraction FPS",
    1, 10, 2,
    disabled=st.session_state.busy
)

epochs = st.slider(
    "Training epochs",
    1, 50, 10,
    disabled=st.session_state.busy
)

gen_btn = st.button("Generate Dataset", disabled=st.session_state.busy)
train_btn = st.button("Train Model", disabled=not st.session_state.dataset_ready)


if gen_btn:
    if not youtube_url:
        st.error("Please enter a YouTube URL.")
    else:
        st.session_state.busy = True

        video_id = get_video_id(youtube_url)
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        BASE_RUN_DIR = os.path.join(PROJECT_ROOT, "data", "runs")
        BASE_DIR = os.path.join(BASE_RUN_DIR, video_id)       
        VIDEO_PATH = os.path.join(BASE_DIR, "video.mp4")
        FRAMES_RAW = os.path.join(BASE_DIR, "frames_raw")
        FRAMES_CLEAN = os.path.join(BASE_DIR, "frames_clean")
        LABELS_DIR = os.path.join(BASE_DIR, "labels")
        DATASET_DIR = os.path.join(BASE_DIR, "dataset")
        CSV_PATH = os.path.join(BASE_DIR, "annotations.csv")

        try:
            os.makedirs(FRAMES_RAW, exist_ok=True)
            os.makedirs(FRAMES_CLEAN, exist_ok=True)
            os.makedirs(LABELS_DIR, exist_ok=True)
            os.makedirs(DATASET_DIR, exist_ok=True)

           
            if not os.path.exists(VIDEO_PATH):
                download_video(youtube_url, VIDEO_PATH)
            else:
                st.info("Video already exists. Skipping download.")

            
            if not os.listdir(FRAMES_RAW):
                extract_frames(VIDEO_PATH, FRAMES_RAW, fps)
            else:
                st.info("Raw frames already exist. Skipping extraction.")

          
            if not os.listdir(FRAMES_CLEAN):
                clean_frames(FRAMES_RAW, FRAMES_CLEAN)
            else:
                st.info("Clean frames already exist. Skipping cleaning.")

           
            labels_train = os.path.join(DATASET_DIR, "labels", "train")
            if not os.path.exists(labels_train) or not os.listdir(labels_train):
                auto_label(
                    images_dir=FRAMES_CLEAN,
                    dataset_dir=DATASET_DIR,
                    conf_thresh=0.35
                )
            else:
                st.info("Labels already exist. Skipping auto-labeling.")

              
            create_data_yaml(DATASET_DIR)

          
            if not os.path.exists(CSV_PATH):
                yolo_to_csv(DATASET_DIR, CSV_PATH)
            else:
             st.info("CSV already exists. Skipping CSV generation.")


            OCR_CSV_PATH = os.path.join(BASE_DIR, "ocr_text.csv")
            if not os.path.exists(OCR_CSV_PATH):
                run_ocr_on_dataset(FRAMES_CLEAN, OCR_CSV_PATH)
            else:
             st.info("OCR CSV already exists. Skipping OCR.")
                
            st.session_state.dataset_ready = True
            st.session_state.base_dir = BASE_DIR
            st.session_state.active_video = video_id

            st.success("Dataset + CSV generated successfully!")

        except Exception as e:
            st.session_state.dataset_ready = False
            st.error(f"Pipeline failed: {e}")
            st.session_state.busy = False


            
visualise_dir = os.path.join(BASE_DIR, "visualized")
if not os.path.exists(visualise_dir) or not os.listdir(visualise_dir):
    visualise(BASE_DIR)
else:
    st.info("Visualized frames already exist. Skipping.")
            

if train_btn:
    st.session_state.busy = True

    BASE_DIR = st.session_state.base_dir
    DATA_YAML = os.path.join(BASE_DIR, "dataset", "data.yaml")

    try:
        train_model(DATA_YAML, epochs)
        st.session_state.model_ready = True
        st.success("Model training completed!")

    except Exception as e:
        st.error(str(e))

    st.session_state.busy = False


if st.session_state.dataset_ready and st.session_state.base_dir:
    BASE_DIR = st.session_state.base_dir

   
    csv_path = os.path.join(BASE_DIR, "annotations.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            st.download_button(
                "Download Annotations CSV",
                f,
                file_name="annotations.csv",
                mime="text/csv")    
    ocr_csv_path = os.path.join(BASE_DIR, "ocr_text.csv")
    if os.path.exists(ocr_csv_path):
         with open(ocr_csv_path, "rb") as f:
              st.download_button("Download OCR Text CSV",
                f,
                file_name="ocr_text.csv",
                mime="text/csv")
            

    # YOLO ZIP

yolo_zip = os.path.join(BASE_DIR, "yolo_dataset.zip")
if not os.path.exists(yolo_zip):
    zip_dir(BASE_DIR, yolo_zip)

with open(yolo_zip, "rb") as f:
    st.download_button("Download YOLO Dataset", f,
                       file_name="yolo_dataset.zip", mime="application/zip")

# TO:
yolo_zip = os.path.join(BASE_DIR, "yolo_dataset.zip")

# Delete old zip if it exists so we always create a fresh one
if os.path.exists(yolo_zip):
    os.remove(yolo_zip)

# Create the zip first, fully, before opening it
zip_dir(BASE_DIR, yolo_zip)

# Only open for download after zip is completely written
if os.path.exists(yolo_zip) and os.path.getsize(yolo_zip) > 0:
    with open(yolo_zip, "rb") as f:
        st.download_button("Download YOLO Dataset", f,
                           file_name="yolo_dataset.zip", mime="application/zip")
