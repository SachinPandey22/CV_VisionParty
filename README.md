# Vision Party

A webcam party game where your body is the controller. Computer vision
(MediaPipe Pose + YOLOv8n) recognizes poses and objects in real time and turns
them into short Solo or Duo challenges. CS 4337 Introduction to Computer Vision.

> Status: Day 4 — four games (Reaction Battle, Hold Challenge, Pose Battle, and Solo-only Bring to Camera with YOLOv8n), spoken voice, shorter timers in later rounds, party extras, a debug view and a pose-capture tool.

## Setup

Requires Python 3.11 (MediaPipe has no wheels for 3.13+).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt -e .
```

(Plain `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt -e .` also works.)

On macOS, the first run asks for camera permission for your terminal app.
If the camera can't be opened, allow it in System Settings → Privacy &
Security → Camera and restart the terminal.

The first Solo match downloads the YOLOv8n weights (~6 MB, from Ultralytics'
GitHub releases) into `models/`. After that, everything runs offline.

## Run

Play (5-round match):

```bash
.venv/bin/python -m vision_party                       # then press 1 (Solo) or 2 (Duo)
.venv/bin/python -m vision_party --p1 Ann --p2 Bob     # custom player names
```

Keys: `1` Solo · `2` Duo · `M` back to menu · `R` play again (after a match) · `Q` / `Esc` quit.
Options: `--rounds N`, `--camera N`, `--save-moments` (save each player's fastest-reaction photo to `moments/`), `--mute` (no voice), `--no-objects` (turn off YOLO and Bring to Camera).

Debug view (skeleton, joint angles, detected actions, FPS):

```bash
.venv/bin/python -m vision_party.debug_view          # Solo
.venv/bin/python -m vision_party.debug_view --duo    # Duo, split screen
```

Keys: `d` toggle Solo/Duo · `a` toggle angle labels · `q` / `Esc` quit.

Record the Pose Battle target poses (strike a pose, press 1-5, hold still):

```bash
.venv/bin/python -m vision_party.capture_poses
```

The live match % against every target is shown while you pose. Poses are saved
to `src/vision_party/targets.json` (commit it to share them). The repo ships
with 5 starter poses (Star, Warrior, Tree, Muscle, Superhero) built from limb
directions; re-record them with a real person for the best results.

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

- **Voice:** every instruction, countdown number and result is spoken. On
  macOS this uses the built-in `say` command; elsewhere pyttsx3. Speech runs on
  a background thread so the video never pauses, and stale lines are dropped
  so speech never falls behind the game.
- **Shorter timers:** each round's time limit shrinks by 7.5% per round, down
  to 70% (5 rounds: 100%, 92.5%, 85%, 77.5%, 70%). Hold Challenge is exempt
  because its 6 s goal is fixed.

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

### Hold Challenge

The game calls a pose to hold (both hands up, squat, one leg, T-pose).

- The hold timer starts the first time the pose is seen after "GO".
- **Dropout tolerance:** losing the pose for under 0.3 s (a missed detection
  or a wobble) is forgiven; a longer gap ends the hold.
- Holding 6 s is a full success. Round length 10 s.
- **Solo:** points scale with the hold (150 for the full 6 s). **Duo:** the
  longer hold wins (within 0.1 s = tie).

### Pose Battle

Copy the target pose shown in the panel at the top right.

- **Similarity:** the 8 joint angles (elbows, shoulders, hips, knees) are
  compared with the target's. Each joint scores 100% when it matches and falls
  linearly to 0% at 60° off; the pose score is the average over the joints
  visible in both. Using angles makes the score independent of the player's
  position, distance from the camera and height. At least 6 joints must be
  visible.
- A player matches at ≥ 75% for 5 frames in a row; the time is taken from the
  first of those frames.
- Points: 100 + up to 50 speed bonus + up to 50 accuracy bonus (75% → 0,
  100% → 50). **Solo:** match before the 8 s timeout. **Duo:** first to match wins.
- The result pop-up rates accuracy: PERFECT! (≥ 95%), GREAT! (≥ 88%),
  GOOD (≥ 81%), OK.

### Bring to Camera (Solo only)

The game names an object (bottle, cup, book or phone); bring it close to the
camera.

- **Detection:** a pretrained YOLOv8n (COCO classes, no training) finds the
  object. "Close" means its bounding box covers at least a set share of the
  frame: 8% for a bottle, 6% for a cup, 12% for a book, 4% for a phone.
- It must stay close for 3 frames in a row; the time is taken from the first.
- An object already close to the camera at "GO" must be moved away first.
- Points: 100 + up to 50 speed bonus (full bonus within 3 s). Round length 12 s.
- YOLO only runs during this game (from the countdown on), so the pose games
  keep their full frame rate. The model loads on a background thread when a
  Solo match starts. If it can't load, or `--no-objects` is set, the game is
  left out and the other three still work.
- Detection costs about 12 ms per 1280×720 frame on a laptop CPU.

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

## Evaluation

`python -m vision_party.evaluate` records the numbers for the report. Run it
from the project folder with one person in front of the camera; it shows
what to do for every trial (countdown, then GO).

```bash
python -m vision_party.evaluate all                                 # full run, ~25 min
python -m vision_party.evaluate all --condition lowlight --quick    # poor-lighting rerun, ~10 min
python -m vision_party.evaluate review-reaction                     # mark movement starts in the videos
python -m vision_party.evaluate summary                             # build results/summary.md
```

| Test | What it measures |
|---|---|
| `actions` | 20 trials per action + 10 "stand still" trials, shuffled: accuracy, latency from GO, false triggers |
| `fps` | full loop FPS (camera + pose + drawing) in Solo, Duo and Solo + YOLO, with pose / YOLO time per frame |
| `reaction` | 10 "raise your right hand at GO" trials after a random wait; each is saved as a video with frame timestamps |
| `review-reaction` | step through those videos (a/d, j/l) and press SPACE on the frame where the hand starts moving; the summary compares this with the detected reaction time |
| `similarity` | each target pose 3 times, scored against all 5 targets: correct-pose vs. wrong-pose scores and how often each passes the 75% threshold |
| `objects` | each object 5 times with the real Bring to Camera rules: success rate and time |

Options: `--condition NAME` labels a run (e.g. `normal`, `lowlight`, `busy`),
`--quick` uses fewer trials, `--trials N` overrides the count, `--camera N`.
Q stops a test and keeps the trials recorded so far.

Raw results (including the reaction videos) go to `results/raw/<condition>/`
and are not committed. `summary` writes `results/summary.md` and one
`*_summary.csv` per test; commit those for the report.

## Layout

```
src/vision_party/
  config.py      thresholds and camera settings
  capture.py     webcam, mirroring, player regions (Solo / Duo split)
  pose.py        MediaPipe wrapper, Pose type, joint angles
  actions.py     rule-based actions
  similarity.py  joint-angle pose similarity, target poses (load/save)
  objects.py     YOLOv8n object detector (lazy, background loading)
  voice.py       non-blocking text-to-speech
  targets.json   the 5 Pose Battle target poses
  games/         one module per game, all sharing the Game interface (base.py):
                 reaction.py, hold.py, pose_battle.py, bring.py
  engine.py      match state machine: rounds, phases, scores, announcements
  ui.py          drawing: skeletons, HUD, menu, countdown, results, winner screen
  fun.py         ratings, streak labels, best-moment photo picking (pure, tested)
  effects.py     party hats, crown, confetti
  main.py        game entry point (python -m vision_party)
  debug_view.py  live debug window
  capture_poses.py  record target poses from the webcam
  evaluate.py    evaluation command line (records trials, builds summary tables)
  evaluation/    runner.py: the camera trials; metrics.py: scoring + tables (pure)
tests/           unit tests on synthetic skeletons and detections (actions, all four games,
                 similarity, engine, timers, voice, extras)
```
