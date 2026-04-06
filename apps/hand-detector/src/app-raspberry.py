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

Requirements
------------
  - Python 3.9+
  - opencv-python   (cv2)
  - mediapipe       (>= 0.10)
  - The model file ``hand_landmarker.task`` must be in the same directory
    as this script (or adjust MODEL_PATH below).

Usage
-----
    python src/app.py
"""

import os
import time
import glob
import cv2
import mediapipe as mp
from lib import helpers
from lib import utils
from lib import handlers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Camera source can be:
#   - integer index ("0", "1", ...)
#   - explicit device path ("/dev/video0")
#   - stream URL or file path
# Prefer setting this via env var on each target device.
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")

# Camera backend preference:
#   auto      -> try V4L2/OpenCV first, then Picamera2 on Linux
#   v4l2      -> force OpenCV VideoCapture
#   picamera2 -> force Raspberry Pi Camera Module backend
CAMERA_BACKEND = os.getenv("CAMERA_BACKEND", "auto").strip().lower()

# Capture resolution.
FRAME_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))

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
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


def parse_camera_source(value):
    """Convert numeric source strings to int, keep all other values as-is."""
    value = value.strip()
    if value.isdigit():
        return int(value)
    return value


def list_video_devices():
    """List available V4L2 video nodes on Linux-like systems."""
    return sorted(glob.glob("/dev/video*"))


def open_opencv_camera(source, width, height):
    """Open camera using OpenCV VideoCapture and apply frame size hints."""
    if os.uname().sysname == "Darwin":
        backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    elif os.uname().sysname == "Linux":
        backend = cv2.CAP_V4L2
    else:
        backend = cv2.CAP_ANY

    cap = cv2.VideoCapture(source, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def open_picamera2_camera(width, height):
    """Open Raspberry Pi Camera Module via Picamera2 if available."""
    try:
        from picamera2 import Picamera2
    except ImportError:
        return None, "Picamera2 is not installed in this environment."

    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.2)
        return picam2, None
    except Exception as exc:  # noqa: BLE001
        return None, f"Picamera2 initialization failed: {exc}"


def build_camera_open_error(source, backend, picam2_reason=None):
    """Return a detailed multi-line camera setup error message."""
    devices = list_video_devices()
    lines = [
        "ERROR: Could not open camera.",
        f"  CAMERA_SOURCE={source!r}",
        f"  CAMERA_BACKEND={backend!r}",
    ]

    if devices:
        lines.append(f"  Detected video devices: {', '.join(devices)}")
    else:
        lines.append("  Detected video devices: none")

    if picam2_reason:
        lines.append(f"  Picamera2 status: {picam2_reason}")

    lines.extend(
        [
            "",
            "Troubleshooting:",
            "  1. In Docker/Dev Container, pass the camera device with --device /dev/video0:/dev/video0.",
            "  2. On Raspberry Pi Camera Module (CSI), install and use Picamera2:",
            "     sudo apt update && sudo apt install -y python3-picamera2 libcamera0",
            "     then run with CAMERA_BACKEND=picamera2.",
            "  3. If your camera is not /dev/video0, set CAMERA_SOURCE to the right index/path.",
        ]
    )

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    """Entry point: open camera, detect hand, display landmarks and coords."""
    source = parse_camera_source(CAMERA_SOURCE)
    cap = None
    picam2 = None
    active_backend = None
    picam2_reason = None

    if CAMERA_BACKEND in ("auto", "v4l2"):
        cap = open_opencv_camera(source, FRAME_WIDTH, FRAME_HEIGHT)
        if cap is not None:
            active_backend = "v4l2"

    if active_backend is None and CAMERA_BACKEND in ("auto", "picamera2"):
        picam2, picam2_reason = open_picamera2_camera(FRAME_WIDTH, FRAME_HEIGHT)
        if picam2 is not None:
            active_backend = "picamera2"

    if active_backend is None:
        print(build_camera_open_error(source, CAMERA_BACKEND, picam2_reason))
        return

    print("Starting camera...")
    print(f"Camera backend: {active_backend}")
    print("Controls:  q = quit  |  p = pause/resume terminal output")

    printing_enabled = True  # Toggle with 'p'.
    last_sign = "NONE"

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                if active_backend == "v4l2":
                    ret, frame = cap.read()
                else:
                    try:
                        rgb_frame = picam2.capture_array()
                        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                        ret = True
                    except Exception:  # noqa: BLE001
                        ret = False
                        frame = None

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
                    current_sign = "NONE"
                    label = "PARTIAL"
                    color = (200, 200, 200)

                    # Check for two-hand gestures first (higher priority)
                    two_hand_gesture = helpers.detect_two_hand_gesture(
                        results.hand_landmarks)

                    if two_hand_gesture:
                        current_sign, label, color = two_hand_gesture
                    else:
                        # Fall back to single-hand gestures (use first detected hand)
                        landmarks = results.hand_landmarks[0]

                        if helpers.is_hand_open(landmarks):
                            current_sign = "ALL_ON"
                            label = "Turn All ON (Open Hand)"
                            color = (0, 255, 255)
                        elif helpers.is_hand_closed(landmarks):
                            current_sign = "ALL_OFF"
                            label = "Turn All OFF (Fist)"
                            color = (0, 0, 255)
                        elif helpers.is_thumb_open(landmarks):
                            current_sign = "LIGHTS_TOGGLE"
                            label = "Power On/Off lights"
                            color = (0, 255, 0)
                        elif helpers.is_middle_and_index_open(landmarks):
                            current_sign = "FAN_TOGGLE"
                            label = "Power On/Off Fan"
                            color = (255, 0, 0)
                        elif helpers.is_pinky_open(landmarks):
                            current_sign = "DEVICE_3"
                            label = "Toggle Device 3 (Pinky)"
                            color = (255, 0, 255)

                    if current_sign != last_sign and current_sign != "NONE":
                        handlers.send_to_arduino(current_sign)
                    last_sign = current_sign

                    cv2.putText(frame, label, (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

                    # Draw skeleton for all detected hands
                    for hand_landmarks in results.hand_landmarks:
                        utils.draw_skeleton(
                            frame, hand_landmarks, utils.HAND_CONNECTIONS)
                else:
                    # No hand detected - show a hint.
                    cv2.putText(frame, "No hand detected", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (100,
                                                              100, 255), 2,
                                cv2.LINE_AA)

                cv2.imshow("Hand Landmarks Debug", frame)

                key = cv2.waitKey(5) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    printing_enabled = not printing_enabled
                    state = "ON" if printing_enabled else "OFF"
                    print(f"[Terminal output {state}]")

    finally:
        if cap is not None:
            cap.release()
        if picam2 is not None:
            picam2.stop()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")


if __name__ == "__main__":
    main()
