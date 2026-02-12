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
import math
import cv2
import mediapipe as mp

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
LANDMARK_NAMES: list[str] = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]

# Skeleton connections (pairs of landmark indices that should be joined by a
# line).  These follow the standard MediaPipe hand topology.
HAND_CONNECTIONS: list[tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# Colors (BGR format for OpenCV).
COLOR_BONE = (0, 255, 0)        # Green lines between joints.
COLOR_JOINT = (0, 0, 255)       # Red circles at each joint.
COLOR_LABEL = (255, 255, 255)   # White text for landmark labels.
COLOR_COORD = (0, 255, 255)     # Yellow text for the coordinate panel.

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
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_skeleton(frame, landmarks, connections):
    """Draw the hand skeleton (bones + joints) onto *frame*.

    Parameters
    ----------
    frame : numpy.ndarray
        The BGR image to draw on (modified in place).
    landmarks : list
        List of 21 ``NormalizedLandmark`` objects from MediaPipe.
    connections : list[tuple[int, int]]
        Pairs of landmark indices to connect with lines.
    """
    h, w, _ = frame.shape
    pixel_coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Draw bones (lines).
    for start_idx, end_idx in connections:
        cv2.line(frame, pixel_coords[start_idx], pixel_coords[end_idx],
                 COLOR_BONE, 2)

    # Draw joints (circles) and index numbers.
    for idx, (px, py) in enumerate(pixel_coords):
        cv2.circle(frame, (px, py), 6, COLOR_JOINT, -1)
        # Place the landmark index slightly above-right of the dot.
        cv2.putText(frame, str(idx), (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_LABEL, 1,
                    cv2.LINE_AA)


def draw_coordinate_panel(frame, landmarks):
    """Render a translucent overlay on the left side showing all coordinates.

    Each line shows:  ``<index> <NAME> x=... y=... z=...``

    Parameters
    ----------
    frame : numpy.ndarray
        The BGR image to draw on (modified in place).
    landmarks : list
        List of 21 ``NormalizedLandmark`` objects from MediaPipe.
    """
    # Semi-transparent dark rectangle as background for readability.
    overlay = frame.copy()
    panel_w = 420
    panel_h = 21 * 22 + 30  # 21 landmarks * 22 px line height + padding
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Title
    cv2.putText(frame, "Landmark Coordinates (normalized)", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_COORD, 1, cv2.LINE_AA)

    # One line per landmark.
    for idx, lm in enumerate(landmarks):
        text = f"{idx:2d} {LANDMARK_NAMES[idx]:<20s} x={lm.x:.3f} y={lm.y:.3f} z={lm.z:.3f}"
        y_pos = 40 + idx * 22
        cv2.putText(frame, text, (8, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_COORD, 1,
                    cv2.LINE_AA)


def print_coordinates_table(landmarks):
    """Print a formatted table of all landmark coordinates to the terminal.

    Parameters
    ----------
    landmarks : list
        List of 21 ``NormalizedLandmark`` objects from MediaPipe.
    """
    header = f"{'ID':>3s}  {'Name':<22s}  {'x':>8s}  {'y':>8s}  {'z':>8s}"
    separator = "-" * len(header)
    print(f"\n{separator}")
    print(header)
    print(separator)
    for idx, lm in enumerate(landmarks):
        print(f"{idx:3d}  {LANDMARK_NAMES[idx]:<22s}  {lm.x:8.4f}  {lm.y:8.4f}  {lm.z:8.4f}")
    print(separator)

# ---------------------------------------------------------------------------
# Hand pose heuristics
# ---------------------------------------------------------------------------
def distance(a, b):
    """Euclidean distance between two landmarks (using x, y)."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

# ---------------------------------------------------------------------------
# For simplicity, the "open hand" condition is defined as all fingertips being
# farther from the wrist than their respective middle joints.  This is a very
# basic heuristic and won't cover all hand poses, but it serves well for this demo.
# ---------------------------------------------------------------------------
def is_hand_open(landmarks):
    """Return True if all 5 fingers are extended (open hand)."""
    wrist = landmarks[0]

    # Thumb: TIP (4) farther from wrist than IP (3)
    thumb_open = distance(landmarks[4], wrist) > distance(landmarks[3], wrist)

    # Index: TIP (8) farther from wrist than PIP (6)
    index_open = distance(landmarks[8], wrist) > distance(landmarks[6], wrist)

    # Middle: TIP (12) farther from wrist than PIP (10)
    middle_open = distance(landmarks[12], wrist) > distance(landmarks[10], wrist)

    # Ring: TIP (16) farther from wrist than PIP (14)
    ring_open = distance(landmarks[16], wrist) > distance(landmarks[14], wrist)

    # Pinky: TIP (20) farther from wrist than PIP (18)
    pinky_open = distance(landmarks[20], wrist) > distance(landmarks[18], wrist)

    return all([thumb_open, index_open, middle_open, ring_open, pinky_open])

# ----------------------------------------------------------------------------
# For simplicity, the "closed fist" condition is defined as all fingertips being
# closer to the wrist than their respective middle joints.  This is a very
# basic heuristic and won't cover all hand poses, but it serves well for this demo.
# ----------------------------------------------------------------------------


def is_hand_closed(landmarks):
    """Return True if all 5 fingers are curled (fist)."""
    wrist = landmarks[0]

    thumb_closed = distance(landmarks[4], wrist) < distance(landmarks[3], wrist)
    index_closed = distance(landmarks[8], wrist) < distance(landmarks[6], wrist)
    middle_closed = distance(landmarks[12], wrist) < distance(landmarks[10], wrist)
    ring_closed = distance(landmarks[16], wrist) < distance(landmarks[14], wrist)
    pinky_closed = distance(landmarks[20], wrist) < distance(landmarks[18], wrist)

    return all([thumb_closed, index_closed, middle_closed, ring_closed, pinky_closed])


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

                    if is_hand_open(landmarks):
                        label = "OPEN HAND"
                        color = (0, 255, 0)
                    elif is_hand_closed(landmarks):
                        label = "CLOSED FIST"
                        color = (0, 0, 255)
                    else:
                        label = "PARTIAL"
                        color = (200, 200, 200)

                    cv2.putText(frame, label, (30, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

                    # Draw skeleton and per-landmark index numbers.
                    draw_skeleton(frame, landmarks, HAND_CONNECTIONS)

                    # Draw the coordinate overlay panel.
                    draw_coordinate_panel(frame, landmarks)

                    # Terminal output (continuous, ~every frame).
                    if printing_enabled:
                        print_coordinates_table(landmarks)
                else:
                    # No hand detected - show a hint.
                    cv2.putText(frame, "No hand detected", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2, 
                                cv2.LINE_AA)

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
                        print_coordinates_table(results.hand_landmarks[0])
                    else:
                        print("[No hand detected - nothing to snapshot]")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")


if __name__ == "__main__":
    main()

