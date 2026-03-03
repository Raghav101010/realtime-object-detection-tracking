# realtime-object-detection-tracking

A production-ready computer vision application that performs:

- Real-time person detection using YOLOv8
- Object tracking with unique IDs
- Centroid-based line crossing detection
- Bi-directional people counting (UP / DOWN)
- Performance monitoring (Inference FPS & Display FPS)
- Video post-processing using FFmpeg
- Streamlit web interface

Features
- YOLOv8-based person detection
- Multi-object tracking with ID assignment
- Line-crossing based directional counting
- Frame skipping optimization for performance
- Frame resizing for efficient inference
- Stable overlay rendering (no flickering)
- Browser-compatible video encoding (H264 via FFmpeg)
- Web interface using Streamlit

Tech Stack
- Python
- OpenCV
- Ultralytics YOLOv8
- DeepSORT / Custom Tracker
- Streamlit
- FFmpeg

How it works
1. Video uploaded via Streamlit
2. Frames resized & optionally skipped
3. YOLO detects persons
4. Tracker assigns unique IDs
5. Centroid positions monitored
6. Line crossing triggers UP/DOWN counter
7. Annotated video saved
8. FFmpeg re-encodes for browser compatibility
9. Output displayed in UI

To run locally
- pip install -r requirements.txt
- streamlit run app.py

Performance Optimization
- Frame skipping
- Frame scaling
- Decoupled inference and rendering
- Cached overlay drawing
- H264 re-encoding for web playback

Future Improvements
- Live webcam streaming support
- GPU acceleration
- Cloud storage integration
- REST API backend
- Multi-line counting zones
- Analytics dashboard

Author
Raghwendra Mahato