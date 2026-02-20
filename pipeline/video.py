import os
import sys
import subprocess

def download_video(youtube_url: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # HARD FAIL if yt-dlp is missing
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bv*[ext=mp4]/best",
        "--no-cache-dir",
        "--force-overwrites",
        "-o", output_path,
        youtube_url
    ]

    subprocess.run(cmd, check=True)
