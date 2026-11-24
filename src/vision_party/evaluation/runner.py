"""Camera-based evaluation trials: one function per test in the plan.

Each test function takes a `session` (camera + pose + window, see Session)
and writes one CSV of raw results to results/raw/<condition>/. Keeping the
camera/UI in Session means the test logic can be unit-tested with a fake
session.
"""

import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np

from .. import config
from ..actions import ACTIONS, detect_actions, right_hand_up
from ..capture import Camera, mirror, player_regions
from ..games.bring import BringToCamera
from ..pose import PoseEstimator
from ..similarity import load_targets, pose_similarity
from .. import ui
from .metrics import NEUTRAL, judge_action_trial, reaction_from_flags, write_rows

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results"
RAW = RESULTS / "raw"

COUNTDOWN = 1.5
RECORD = 2.5
FEEDBACK = 1.0


class Quit(Exception):
    """The person pressed Q: stop and save what's been recorded so far."""


def _label(action: str) -> str:
    return "stand still" if action == NEUTRAL else action.replace("_", " ").lower()


def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


class Session:
    """Webcam, pose estimation and the window, shared by all tests."""

    def __init__(self, camera_index: int, duo: bool = False):
        self.cam = Camera(camera_index).__enter__()
        self.estimators: list[PoseEstimator] = []
        self.set_duo(duo)
        self.pose_ms = 0.0                # time spent on pose for the last frame
        cv2.namedWindow("Vision Party evaluation", cv2.WINDOW_NORMAL)

    def set_duo(self, duo: bool) -> None:
        for e in self.estimators:
            e.close()
        self.duo = duo
        self.estimators = [PoseEstimator(r) for r in player_regions(duo)]

    def frame(self):
        """Read one frame and run pose. Returns (raw, poses, now)."""
        raw = self.cam.read()
        now = time.perf_counter()
        t0 = time.perf_counter()
        poses = [e.process(raw) for e in self.estimators]
        self.pose_ms = 1000 * (time.perf_counter() - t0)
        return raw, poses, now

    def show(self, raw, poses, lines: list[tuple[str, tuple]] = (), big: str = "",
             big_color=ui.WHITE, reference=None) -> None:
        view = mirror(raw)
        if self.duo:
            ui.draw_divider(view)
        for i, pose in enumerate(poses):
            if pose is not None:
                ui.draw_skeleton(view, pose, ui.PLAYER_COLORS[i])
        h, w = view.shape[:2]
        ui.shade(view, 0, 0, w, 50 + 38 * len(lines), 0.6)
        for i, (text, color) in enumerate(lines):
            ui.text(view, text, (20, 40 + 38 * i), 0.85, color)
        if big:
            ui.center_text(view, big, w // 2, h // 2 + 40, 3.0, big_color, 6)
        if reference is not None:
            ui.draw_reference(view, reference)
        ui.text(view, "Q = stop and save", (20, h - 20), 0.6, ui.WHITE, 1)
        cv2.imshow("Vision Party evaluation", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise Quit

    def run_trial(self, header: str, prompt: str, on_frame, countdown=COUNTDOWN,
                  record=RECORD, wait: float = 0.0, reference=None) -> float:
        """Countdown -> optional random wait -> GO, then `record` seconds.

        on_frame(raw, poses, now, phase) is called for every frame, phase being
        'countdown', 'wait' or 'go'; return True during 'go' to end early.
        Returns the time of GO.
        """
        start = time.perf_counter()
        go = start + countdown + wait
        while True:
            raw, poses, now = self.frame()
            if now < start + countdown:
                phase, big, color = "countdown", str(int(start + countdown - now) + 1), ui.WHITE
            elif now < go:
                phase, big, color = "wait", "wait...", ui.YELLOW
            else:
                phase, big, color = "go", "GO!", ui.GREEN
            stop = on_frame(raw, poses, now, phase)
            self.show(raw, poses, [(header, ui.WHITE), (prompt, ui.YELLOW)], big, color, reference)
            if phase == "go" and (stop or now - go >= record):
                return go

    def message(self, lines: list[tuple[str, tuple]], seconds: float = FEEDBACK) -> None:
        """Show a message over the live camera for a moment."""
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            raw, poses, _ = self.frame()
            self.show(raw, poses, lines)

    def close(self) -> None:
        for e in self.estimators:
            e.close()
        self.cam.__exit__(None, None, None)
        cv2.destroyAllWindows()


def _out(condition: str, test: str) -> Path:
    return RAW / condition / f"{test}_{stamp()}.csv"


def _save(rows: list[dict], path: Path) -> Path | None:
    if rows:
        write_rows(path, rows)
        print(f"Saved {len(rows)} rows to {path}")
        return path
    return None


# --- 1. action accuracy -----------------------------------------------------
def run_actions(session, condition: str, trials: int = 20, seed: int = 0) -> Path | None:
    """`trials` trials per action, plus trials//2 'stand still' trials, shuffled."""
    session.set_duo(False)
    plan = [a for a in ACTIONS for _ in range(trials)] + [NEUTRAL] * max(1, trials // 2)
    random.Random(seed).shuffle(plan)
    rows: list[dict] = []
    path = _out(condition, "actions")
    try:
        for k, action in enumerate(plan, 1):
            frames, times = [], []

            def on_frame(raw, poses, now, phase):
                if phase == "go":
                    frames.append(detect_actions(poses[0]))
                    times.append(now)

            go = session.run_trial(f"Actions  {k}/{len(plan)}  [{condition}]",
                                   f"At GO: {_label(action)}  (stand normally until then)",
                                   on_frame)
            res = judge_action_trial(frames, times, go, action, config.REACTION_CONFIRM_FRAMES)
            rows.append({"condition": condition, "trial": k, "action": action,
                         "hit": int(res["hit"]),
                         "latency": "" if res["latency"] is None else round(res["latency"], 3),
                         "false_triggers": ";".join(res["false_triggers"]),
                         "frames": len(frames)})
            ok = "correct" if res["hit"] else "MISSED"
            session.message([(f"{_label(action)}: {ok}", ui.GREEN if res["hit"] else ui.RED),
                             ("Back to standing normally", ui.WHITE)])
    except Quit:
        pass
    return _save(rows, path)


# --- 2. frame rate ----------------------------------------------------------
def run_fps(session, condition: str, seconds: float = 15.0, detector=None) -> Path | None:
    """Full loop FPS (camera + pose + drawing) in Solo, Duo and Solo + YOLO."""
    rows: list[dict] = []
    path = _out(condition, "fps")
    modes = [("solo", False, False), ("duo", True, False)]
    if detector is not None:
        modes.append(("solo+yolo", False, True))
    try:
        for mode, duo, yolo in modes:
            session.set_duo(duo)
            who = "both players in view (one is fine)" if duo else "stay in view"
            session.message([(f"FPS test: {mode}", ui.YELLOW), (f"Get ready, {who}", ui.WHITE)], 3.0)
            n, pose_ms, yolo_ms = 0, 0.0, 0.0
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < seconds:
                raw, poses, _ = session.frame()
                pose_ms += session.pose_ms
                if yolo:
                    t1 = time.perf_counter()
                    detector.detect(raw)
                    yolo_ms += 1000 * (time.perf_counter() - t1)
                left = seconds - (time.perf_counter() - t0)
                session.show(raw, poses, [(f"Measuring FPS: {mode}", ui.YELLOW),
                                          (f"{left:.0f} s left - move around normally", ui.WHITE)])
                n += 1
            elapsed = time.perf_counter() - t0
            rows.append({"condition": condition, "mode": mode, "frames": n,
                         "seconds": round(elapsed, 2), "fps": round(n / elapsed, 2),
                         "pose_ms": round(pose_ms / max(n, 1), 2),
                         "yolo_ms": round(yolo_ms / max(n, 1), 2) if yolo else ""})
    except Quit:
        pass
    session.set_duo(False)
    return _save(rows, path)


# --- 3. reaction time vs. video frames --------------------------------------
def run_reaction(session, condition: str, trials: int = 10, seed: int = 0) -> Path | None:
    """Raise the right hand at GO (after a random wait). Each trial is saved as
    a video + per-frame timestamps, so the movement start can be marked by
    hand later (`review-reaction`) and compared with the detected time."""
    session.set_duo(False)
    rng = random.Random(seed)
    run = stamp()
    clip_dir = RAW / condition / f"reaction_{run}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    path = RAW / condition / f"reaction_{run}.csv"
    try:
        for k in range(1, trials + 1):
            clip = clip_dir / f"clip_{k:02d}.mp4"
            writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
            frames = []        # (index, time, flag)

            def on_frame(raw, poses, now, phase):
                if phase == "countdown":
                    return
                writer.write(cv2.resize(raw, (640, 360)))
                frames.append((len(frames), now, bool(poses[0] is not None and right_hand_up(poses[0]))))

            try:
                go = session.run_trial(f"Reaction  {k}/{trials}  [{condition}]",
                                       "Hands down. Raise your RIGHT hand as soon as you see GO!",
                                       on_frame, countdown=1.5, record=2.0, wait=rng.uniform(1.0, 3.0))
            finally:
                writer.release()
            times = [t for _, t, _ in frames]
            detected = reaction_from_flags([f for _, _, f in frames], times, go,
                                           config.REACTION_CONFIRM_FRAMES)
            go_frame = next((i for i, t, _ in frames if t >= go), len(frames))
            with open(clip.with_suffix(".csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "t_from_go", "right_hand_up"])
                for i, t, flag in frames:
                    w.writerow([i, round(t - go, 4), int(flag)])
            rows.append({"condition": condition, "trial": k, "clip": str(clip.relative_to(RAW)),
                         "go_frame": go_frame,
                         "detected_s": "" if detected is None else round(detected, 3),
                         "marked_frame": "", "marked_s": ""})
            msg = f"detected {detected:.2f}s" if detected is not None else "not detected"
            session.message([(f"Reaction: {msg}", ui.GREEN if detected else ui.RED),
                             ("Hands back down", ui.WHITE)])
    except Quit:
        pass
    return _save(rows, path)


def review_reaction(csv_path: Path) -> None:
    """Step through each recorded clip and mark the frame where the hand
    starts moving. Keys: d/a = next/previous frame, l/j = +/-10 frames,
    SPACE = mark this frame, n = skip clip, q = save and quit."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    cv2.namedWindow("Review reaction", cv2.WINDOW_NORMAL)
    try:
        for row in rows:
            clip = RAW / row["clip"]
            with open(clip.with_suffix(".csv"), newline="") as f:
                times = [float(r["t_from_go"]) for r in csv.DictReader(f)]
            cap = cv2.VideoCapture(str(clip))
            frames = []
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                frames.append(img)
            cap.release()
            if not frames:
                continue
            n = min(len(frames), len(times))
            i = int(row["go_frame"]) if row["go_frame"] else 0
            while True:
                i = max(0, min(n - 1, i))
                view = frames[i].copy()
                ui.shade(view, 0, 0, view.shape[1], 95, 0.6)
                ui.text(view, f"Trial {row['trial']}  frame {i}/{n - 1}  t = {times[i]:+.3f}s from GO",
                        (10, 28), 0.6, ui.WHITE, 1)
                det = row["detected_s"] or "-"
                mark = row["marked_s"] or "-"
                ui.text(view, f"detected {det}s   marked {mark}s", (10, 56), 0.6, ui.YELLOW, 1)
                ui.text(view, "a/d frame  j/l 10 frames  SPACE mark start  n next  q quit",
                        (10, 84), 0.5, ui.WHITE, 1)
                if times[i] >= 0 and (i == 0 or times[i - 1] < 0):
                    ui.text(view, "GO", (view.shape[1] - 70, 40), 1.0, ui.GREEN)
                cv2.imshow("Review reaction", view)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("d"):
                    i += 1
                elif key == ord("a"):
                    i -= 1
                elif key == ord("l"):
                    i += 10
                elif key == ord("j"):
                    i -= 10
                elif key == ord(" "):
                    row["marked_frame"], row["marked_s"] = i, round(times[i], 3)
                    break
                elif key == ord("n"):
                    break
                elif key in (ord("q"), 27):
                    raise Quit
    except Quit:
        pass
    finally:
        cv2.destroyAllWindows()
        write_rows(Path(csv_path), rows)
        print(f"Saved marks to {csv_path}")


# --- 4. pose similarity: correct vs. wrong poses ---------------------------
def run_similarity(session, condition: str, reps: int = 3, seed: int = 0) -> Path | None:
    """Do each target pose `reps` times. Every attempt is scored against all
    5 targets: the score for the pose being done is the 'correct' score, the
    other 4 are 'wrong-pose' scores."""
    session.set_duo(False)
    targets = load_targets()
    plan = [t for t in targets for _ in range(reps)]
    random.Random(seed).shuffle(plan)
    rows: list[dict] = []
    path = _out(condition, "similarity")
    try:
        for k, performed in enumerate(plan, 1):
            scores: dict[str, list[float]] = {t.key: [] for t in targets}

            def on_frame(raw, poses, now, phase):
                if phase == "go":
                    for t in targets:
                        s = pose_similarity(poses[0], t)
                        if s is not None:
                            scores[t.key].append(s)

            session.run_trial(f"Poses  {k}/{len(plan)}  [{condition}]",
                              f"Get into the pose: {performed.name} - hold it at GO",
                              on_frame, countdown=3.0, record=2.0, reference=performed)
            for t in targets:
                vals = scores[t.key]
                rows.append({"condition": condition, "trial": k, "performed": performed.key,
                             "target": t.key,
                             "best": round(max(vals), 1) if vals else "",
                             "mean": round(float(np.mean(vals)), 1) if vals else ""})
            own = scores[performed.key]
            best = f"{max(own):.0f}%" if own else "not visible"
            session.message([(f"{performed.name}: best match {best}", ui.YELLOW),
                             ("Relax", ui.WHITE)])
    except Quit:
        pass
    return _save(rows, path)


# --- 5. object success rate -------------------------------------------------
def run_objects(session, condition: str, detector, trials: int = 5, seed: int = 0) -> Path | None:
    """Bring each object to the camera `trials` times, using the game's own
    Bring to Camera rules (start with the object out of view or far away)."""
    session.set_duo(False)
    objects = [o for o in config.OBJECT_CLASSES for _ in range(trials)]
    random.Random(seed).shuffle(objects)
    rows: list[dict] = []
    path = _out(condition, "objects")
    try:
        for k, obj in enumerate(objects, 1):
            game = BringToCamera(1, random.Random(k), target=obj)
            state = {"started": False, "conf": 0.0}

            def on_frame(raw, poses, now, phase):
                if phase != "go":
                    return False
                if not state["started"]:
                    game.begin(now)
                    state["started"] = True
                dets = detector.detect(raw)
                for d in dets:
                    if d.name == obj:
                        state["conf"] = max(state["conf"], d.confidence)
                game.observe(dets)
                return game.update(poses, now)

            session.run_trial(f"Objects  {k}/{len(objects)}  [{condition}]",
                              f"Hide the {obj}. At GO, bring it close to the camera!",
                              on_frame, countdown=3.0, record=config.OBJECT_TIME_LIMIT + 0.5)
            t = game.time_taken
            rows.append({"condition": condition, "trial": k, "object": obj,
                         "success": int(t is not None),
                         "time_s": "" if t is None else round(t, 2),
                         "best_conf": round(state["conf"], 3)})
            msg = f"got it in {t:.1f}s" if t is not None else "not detected"
            session.message([(f"{obj}: {msg}", ui.GREEN if t is not None else ui.RED),
                             ("Put it away again", ui.WHITE)], 2.0)
    except Quit:
        pass
    return _save(rows, path)
