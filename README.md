# 🎮 Game Vision Factory

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00A6ED?style=for-the-badge)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**Automated Gameplay Video → YOLO Dataset → CSV → Trainable Model**

Game Vision Factory is an end-to-end computer vision data pipeline that transforms raw gameplay videos into YOLO-formatted datasets, CSV annotations, and trainable object detection models — **without manual labeling.**

---

## 📌 Why This Project Exists
In most computer vision projects, the biggest challenge is not the model — it is **data availability and preparation.**

> [!CAUTION]
> **The Real Problem:**
> * No domain-specific datasets exist for most games.
> * Manual annotation is slow, expensive, and impractical.
> * Dataset engineering usually takes 80% of the project time.

Game Vision Factory addresses this gap by automating the entire pipeline, enabling faster iteration and feasibility testing.

---

## 🧠 How It Works (The Pipeline)

Using **Weak Supervision**, the system leverages pretrained models to "teach" new models, bypassing the manual labeling bottleneck.

### 🧠 Pipeline Overview

```mermaid
graph TD
    A([fa:fa-youtube YouTube URL]) --> B[Video Download]
    B --> C[Frame Extraction]
    C --> D[Duplicate Removal]
    D --> E{Auto-Labeling}
    E --> F[YOLO Dataset]
    F --> G[CSV Export]
    G --> H[Model Training]
    H --> I((Trained .pt Model))

    style A fill:#f00,color:#fff
    style E fill:#636,color:#fff
    style I fill:#2c5,color:#fff
```
## 🌟 Core Capabilities

    🎥 Smart Ingestion: Accepts YouTube URLs with local caching and resume-safe downloads.

    🧹 Data Distillation: Uses pixel-difference analysis to remove near-duplicate frames, ensuring high visual diversity.

    🧠 Auto-Annotation: Uses YOLOv8 to generate bounding boxes. No manual clicking required for early-stage prototypes.

    📦 Industry Standard Outputs: Generates full YOLO directory structures (data.yaml) and framework-agnostic CSVs.

    🌐 No-Code UI: Powered by Streamlit, allowing users to manage state and downloads through a browser.

## 📁 Project Structure

```text

├── app.py              # Streamlit UI & pipeline orchestration
├── requirements.txt    # Project dependencies
├── yolov8n.pt          # Pretrained model weights
├── pipeline/           # Core logic modules
│   ├── video.py        # YouTube download logic
│   ├── frames.py       # Frame extraction (FFmpeg)
│   ├── cleaning.py     # Redundancy removal
│   ├── labeling.py     # YOLO auto-labeling
│   ├── dataset.py      # YAML & CSV generation
│   └── train.py        # Training wrapper
├── data/               # Input/Output data storage
└── tests/              # Unit tests for pipeline
    ├── test_video.py
    ├── test_frames.py
    └── test_cleaning.py
```
## 🛠️ Technology Stack

| Category | Tools & Frameworks |
| :--- | :--- |
| **Computer Vision** | ![YOLOv8](https://img.shields.io/badge/YOLOv8-00A6ED?style=for-the-badge&logo=ultralytics&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white) |
| **Machine Learning** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) |
| **Media Processing** | ![FFmpeg](https://img.shields.io/badge/FFmpeg-007800?style=for-the-badge&logo=ffmpeg&logoColor=white) |
| **Interface** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) |
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| **Data Format** | ![JSON](https://img.shields.io/badge/JSON-000000?style=for-the-badge&logo=json&logoColor=white) ![CSV](https://img.shields.io/badge/CSV-4169E1?style=for-the-badge&logo=data:image/png;base64,...) |

---

## 🚀 Getting Started

Follow these steps to set up the Game Vision Factory on your local machine.

**1. Clone & Environment:**
```text
git clone <repository-url>
cd <repository-directory>

Create a virtual environment**

python -m venv venv

Activate it**

On Windows:
venv\Scripts\activate

On Mac/Linux:
source venv/bin/activate
```
**2. Install FFmpeg (System Dependency)**
Since your pipeline uses FFmpeg for high-speed frame extraction, you must install it on your OS:

| OS | Installation Command / Instructions |
| :--- | :--- |
| **Windows** | Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), extract, and add the `bin` folder to your **System PATH**. |
| **macOS** | `brew install ffmpeg` |
| **Linux (Ubuntu)** | `sudo apt update && sudo apt install ffmpeg` |



**3. Install Python Packages**
```text
pip install --upgrade pip
pip install ultralytics opencv-python-headless streamlit yt-dlp pandas numpy torch torchvision
```
**[!TIP]**
*Use opencv-python-headless if you are running this on a server or Docker without a GUI. Use opencv-python if you need to pop up local windows for debugging!

**4. Launch the App**
```text
streamlit run app.py
```
**📄 Your requirements.txt**

-- ultralytics>=8.0.0
-- opencv-python
-- streamlit
-- yt-dlp
-- pandas
-- numpy





---
### 📦 Key Dependencies
* `ultralytics` : Core engine for running the **YOLOv8** object detection model.
* `opencv-python` : Handles all image manipulation and video frame buffering.
* `numpy` : High-performance numerical operations for frame cleaning logic.
* `torch` : The deep learning backend required for model training.
