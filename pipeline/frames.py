import os
import subprocess
import shutil


FFMPEG_PATH = "ffmpeg"
FFMPEG_PATH = r"C:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

def extract_frames(video_path, output_dir, fps, max_frames=300):
    """
    Extract frames from a video using ffmpeg.
    """

    # Check ffmpeg availability
    if shutil.which(FFMPEG_PATH) is None:
        raise RuntimeError("ffmpeg not found in PATH")

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-frames:v", str(max_frames),
        os.path.join(output_dir, "%06d.jpg")
    ]

    print("[INFO] Running ffmpeg frame extraction")

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print("[INFO] Frame extraction completed")
