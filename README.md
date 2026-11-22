# Vision Party

A webcam party game where your body is the controller. Computer vision
(MediaPipe Pose + YOLOv8n) recognizes poses and objects in real time and turns
them into short Solo or Duo challenges. CS 4337 Introduction to Computer Vision.

> Status: Day 1 — pose estimation, action rules and a live debug view.

## Setup

Requires Python 3.11 (MediaPipe has no wheels for 3.13+).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt -e .
```

(Plain `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt -e .` also works.)

On macOS, the first run asks for camera permission for your terminal app.

## Run

Debug view (skeleton, joint angles, detected actions, FPS):

```bash
.venv/bin/python -m vision_party.debug_view          # Solo
.venv/bin/python -m vision_party.debug_view --duo    # Duo, split screen
```

Keys: `d` toggle Solo/Duo · `a` toggle angle labels · `q` / `Esc` quit.

Tests (no camera needed):

```bash
.venv/bin/python -m pytest -q
```

## How it works

- **Input:** webcam frames (BGR, 1280×720).
- **Pose:** MediaPipe Pose gives 33 body landmarks per player. In Duo the raw
  frame is split in half and each half gets its own tracker, because MediaPipe
  tracks one person per image.
- **Mirroring:** pose runs on the raw frame so MediaPipe's left/right match the
  player's real hands; only the display is mirrored. Player 1 (screen left) is
  therefore the raw frame's right half.
- **Actions:** pure rule functions on landmarks. Distances are scaled by torso
  length, so rules work at any distance from the camera.
- **Output:** per-player booleans for each action, drawn on the debug view.

| Action | Rule |
|---|---|
| `LEFT_HAND_UP` / `RIGHT_HAND_UP` | That wrist above the nose, the other hand not up |
| `BOTH_HANDS_UP` | Both wrists above the nose |
| `SQUAT` | Average knee angle < 115° and hips dropped toward knee height |
| `ONE_LEG_UP` | One ankle clearly higher than the other |
| `T_POSE` | Straight elbows, wrists at shoulder height, arms spread wide |

All thresholds live in `src/vision_party/config.py`.

## Layout

```
src/vision_party/
  config.py      thresholds and camera settings
  capture.py     webcam, mirroring, player regions (Solo / Duo split)
  pose.py        MediaPipe wrapper, Pose type, joint angles
  actions.py     rule-based actions
  debug_view.py  live debug window
tests/           action-rule tests on synthetic skeletons
```
