# Vision Party

A webcam party game where your body is the controller. Computer vision
(MediaPipe Pose + YOLOv8n) recognizes poses and objects in real time and turns
them into short Solo or Duo challenges. CS 4337 Introduction to Computer Vision.

> Status: Day 2 — playable Reaction Battle in Solo and Duo, plus a live debug view.

## Setup

Requires Python 3.11 (MediaPipe has no wheels for 3.13+).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt -e .
```

(Plain `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt -e .` also works.)

On macOS, the first run asks for camera permission for your terminal app.

## Run

Play (5-round match):

```bash
.venv/bin/python -m vision_party                       # then press 1 (Solo) or 2 (Duo)
.venv/bin/python -m vision_party --p1 Ann --p2 Bob     # custom player names
```

Keys: `1` Solo · `2` Duo · `M` back to menu · `R` play again (after a match) · `Q` / `Esc` quit.
Options: `--rounds N`, `--camera N`, `--save-moments` (save each player's fastest-reaction photo to `moments/`).

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

### Match flow

Each round goes **intro → 3-2-1 countdown → active → result**; after the last
round a winner screen shows final scores. The engine (`engine.py`) is a pure
state machine: it takes poses and the current time and never touches the
camera or screen, which makes it testable without a webcam.

### Reaction Battle

The game calls a random action; the timer starts at "GO".

- An action must hold for 3 consecutive frames to count (filters single-frame
  noise). Reaction time is measured from the first of those frames.
- A player already doing the action at "GO" must drop it first, so holding a
  pose through the countdown doesn't win.
- **Solo:** succeed before the 5 s timeout. **Duo:** first player wins; same
  frame = tie.
- Points: 100 + speed bonus of up to 50 (full bonus at ≤ 0.4 s, falling to 0 at
  the timeout).

### Party extras

Small touches that make a match feel like a party. None of them change the score.

- **Ratings:** after each round, a pop-up rates every player who reacted:
  PERFECT! (≤ 0.6 s), GREAT! (≤ 1.0 s), GOOD (≤ 2.0 s) or OK.
- **Streaks:** "2 in a row", then "ON FIRE x3" for rounds won back to back,
  with a spoken "Ann is on fire!".
- **Party hats and a crown:** every player wears a party hat anchored to the
  eye and ear landmarks, so it follows the head and tilts with it. In Duo the
  player ahead on points wears a crown; in Solo you earn it by getting on fire.
- **Confetti:** bursts over the round winner and rains on the game-over screen.
- **Best-moment photos:** the game keeps the frame from each player's fastest
  reaction and shows it on the game-over screen.

Thresholds are in the "Fun extras" section of `config.py`.

## Layout

```
src/vision_party/
  config.py      thresholds and camera settings
  capture.py     webcam, mirroring, player regions (Solo / Duo split)
  pose.py        MediaPipe wrapper, Pose type, joint angles
  actions.py     rule-based actions
  games/         one module per game, all sharing the Game interface (base.py)
  engine.py      match state machine: rounds, phases, scores, announcements
  ui.py          drawing: skeletons, HUD, menu, countdown, results, winner screen
  fun.py         ratings, streak labels, best-moment photo picking (pure, tested)
  effects.py     party hats, crown, confetti
  main.py        game entry point (python -m vision_party)
  debug_view.py  live debug window
tests/           unit tests on synthetic skeletons (actions, Reaction Battle, engine, extras)
```
