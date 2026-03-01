# app.py

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["video", "webcam"], required=True)
args = parser.parse_args()

if args.mode == "video":
    subprocess.run([sys.executable, "scripts/run_video.py"])
elif args.mode == "webcam":
    subprocess.run([sys.executable, "scripts/run_webcam.py"])