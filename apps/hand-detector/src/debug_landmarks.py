"""Debug tool for visualizing MediaPipe hand landmarks and their coordinates.

This script opens the webcam, detects a hand using MediaPipe's HandLandmarker
(Tasks API), and renders every landmark with its index, name, and real-time
normalized coordinates (x, y, z) both on the video feed and in the terminal.

MediaPipe Hand Landmark Model
-----------------------------
The model outputs 21 3D landmarks per detected hand.  Each landmark carries
three normalized values:

  - x : horizontal position (0.0 = left edge, 1.0 = right edge of the frame)
  - y : vertical position   (0.0 = top edge,  1.0 = bottom edge of the frame)
  - z : depth relative to the wrist, roughly in the same scale as x.
        Negative values mean the point is closer to the camera than the wrist.

Landmark index map (see also: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
---------------------
    0  - WRIST
    1  - THUMB_CMC          (carpometacarpal joint)
    2  - THUMB_MCP          (metacarpophalangeal joint)
    3  - THUMB_IP           (interphalangeal joint)
    4  - THUMB_TIP
    5  - INDEX_FINGER_MCP
    6  - INDEX_FINGER_PIP   (proximal interphalangeal joint)
    7  - INDEX_FINGER_DIP   (distal interphalangeal joint)
    8  - INDEX_FINGER_TIP
    9  - MIDDLE_FINGER_MCP
   10  - MIDDLE_FINGER_PIP
   11  - MIDDLE_FINGER_DIP
   12  - MIDDLE_FINGER_TIP
   13  - RING_FINGER_MCP
   14  - RING_FINGER_PIP
   15  - RING_FINGER_DIP
   16  - RING_FINGER_TIP
   17  - PINKY_MCP
   18  - PINKY_PIP
   19  - PINKY_DIP
   20  - PINKY_TIP

Controls
--------
  q      - Quit the application.
  p      - Pause / resume the coordinate printout in the terminal.
  s      - Take a single snapshot of all coordinates to the terminal
           (useful when the live printout is paused).

Requirements
------------
  - Python 3.9+
  - opencv-python   (cv2)
  - mediapipe       (>= 0.10)
  - The model file ``hand_landmarker.task`` must be in the same directory
    as this script (or adjust MODEL_PATH below).

Usage
-----
    python debug_landmarks.py
"""

import os
import time
import cv2
import mediapipe as mp
from lib import utils

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Camera index.  0 is usually the built-in FaceTime HD camera on macOS.
# If an iPhone is acting as a Continuity Camera, try index 1.
CAMERA_INDEX = 0

# Path to the HandLandmarker model (same folder as this script).
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task",
)

# Download the model if it's missing.
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print(f"Downloading HandLandmarker model to {MODEL_PATH}...")
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(model_url, MODEL_PATH)
    print("Download complete.")

# Human-readable landmark names, indexed 0-20.
LANDMARK_NAMES = utils.LANDMARK_NAMES

# Skeleton connections (pairs of landmark indices that should be joined by a
# line).  These follow the standard MediaPipe hand topology.
HAND_CONNECTIONS = utils.HAND_CONNECTIONS

# ---------------------------------------------------------------------------
# MediaPipe Tasks API setup
# ---------------------------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    """Entry point: open camera, detect hand, display landmarks and coords."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Could not open camera. Check CAMERA_INDEX.")
        return

    print("Starting camera...")
    print("Controls:  q = quit  |  p = pause/resume terminal output  |  s = snapshot coords")

    printing_enabled = True  # Toggle with 'p'.

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: Could not read frame. Is the camera in use?")
                    break

                # Mirror the image so it feels natural.
                frame = cv2.flip(frame, 1)

                # Convert BGR -> RGB for MediaPipe.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                timestamp_ms = int(time.monotonic() * 1000)
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                if results.hand_landmarks:
                    landmarks = results.hand_landmarks[0]

                    # Draw skeleton and per-landmark index numbers.
                    utils.draw_skeleton(frame, landmarks, HAND_CONNECTIONS)

                    # Draw the coordinate overlay panel.
                    utils.draw_coordinate_panel(frame, landmarks)

                    # Terminal output (continuous, ~every frame).
                    if printing_enabled:
                        utils.print_coordinates_table(landmarks)
                else:
                    # No hand detected - show a hint.
                    cv2.putText(frame, "No hand detected", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255),
                                2, cv2.LINE_AA)

                cv2.imshow("Hand Landmarks Debug", frame)

                key = cv2.waitKey(5) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    printing_enabled = not printing_enabled
                    state = "ON" if printing_enabled else "OFF"
                    print(f"[Terminal output {state}]")
                elif key == ord("s"):
                    if results.hand_landmarks:
                        utils.print_coordinates_table(results.hand_landmarks[0])
                    else:
                        print("[No hand detected - nothing to snapshot]")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")


if __name__ == "__main__":
    main()
