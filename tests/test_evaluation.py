"""Tests for the evaluation: trial scoring, summaries, and every test
procedure run through a fake camera session."""

import csv
import random

import numpy as np
import pytest

from vision_party import config
from vision_party.evaluation import metrics, runner
from vision_party.evaluation.metrics import (NEUTRAL, first_streak, judge_action_trial,
                                             reaction_from_flags, summarize)
from vision_party.objects import Detection
from vision_party.similarity import load_targets

from poses import LEFT_HAND, RIGHT_HAND, make_pose
from test_day3 import pose_from_target

STAND = make_pose()
DT = 1 / 30


# --- trial scoring ----------------------------------------------------------
def test_first_streak():
    t = [i * 0.1 for i in range(8)]
    assert first_streak([0, 1, 1, 0, 1, 1, 1, 0], t, 3) == pytest.approx(0.4)
    assert first_streak([1, 1, 0, 1, 1, 0], t, 3) is None


def test_judge_hit_latency_and_false_triggers():
    times = [i * 0.1 for i in range(6)]
    frames = [{"SQUAT": False, "T_POSE": False}] * 2 + [{"SQUAT": True, "T_POSE": True}] * 4
    r = judge_action_trial(frames, times, 0.0, "SQUAT", 3)
    assert r["hit"] and r["latency"] == pytest.approx(0.2)
    assert r["false_triggers"] == ["T_POSE"]


def test_both_hands_implies_one_hand():
    times = [i * 0.1 for i in range(4)]
    frames = [{"BOTH_HANDS_UP": True, "LEFT_HAND_UP": True, "RIGHT_HAND_UP": True}] * 4
    r = judge_action_trial(frames, times, 0.0, "BOTH_HANDS_UP", 3)
    assert r["hit"] and r["false_triggers"] == []


def test_neutral_trial():
    times = [i * 0.1 for i in range(4)]
    still = judge_action_trial([{"SQUAT": False}] * 4, times, 0.0, NEUTRAL, 3)
    moved = judge_action_trial([{"SQUAT": True}] * 4, times, 0.0, NEUTRAL, 3)
    assert still["hit"] and not moved["hit"] and moved["false_triggers"] == ["SQUAT"]


def test_reaction_needs_hand_down_first():
    times = [i * 0.1 for i in range(10)]
    # Hand already up at GO (t=0.2) and stays up: no valid reaction.
    assert reaction_from_flags([1] * 10, times, 0.2, 3) is None
    # Down after GO, raised at t=0.5.
    flags = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert reaction_from_flags(flags, times, 0.2, 3) == pytest.approx(0.3)


# --- summaries --------------------------------------------------------------
def _write(path, rows):
    metrics.write_rows(path, rows)


def test_summarize_builds_all_tables(tmp_path):
    raw = tmp_path / "raw"
    _write(raw / "normal" / "actions_1.csv", [
        {"condition": "normal", "trial": 1, "action": "SQUAT", "hit": 1, "latency": 0.5, "false_triggers": ""},
        {"condition": "normal", "trial": 2, "action": "SQUAT", "hit": 0, "latency": "", "false_triggers": "T_POSE"}])
    _write(raw / "lowlight" / "actions_2.csv", [
        {"condition": "lowlight", "trial": 1, "action": "SQUAT", "hit": 1, "latency": 0.7, "false_triggers": ""}])
    _write(raw / "normal" / "fps_1.csv", [
        {"condition": "normal", "mode": "solo", "frames": 300, "seconds": 10, "fps": 30, "pose_ms": 20, "yolo_ms": ""}])
    _write(raw / "normal" / "reaction_1.csv", [
        {"condition": "normal", "trial": 1, "clip": "c", "go_frame": 3, "detected_s": 0.5, "marked_frame": 5, "marked_s": 0.4},
        {"condition": "normal", "trial": 2, "clip": "c", "go_frame": 3, "detected_s": 0.6, "marked_frame": "", "marked_s": ""}])
    _write(raw / "normal" / "similarity_1.csv", [
        {"condition": "normal", "trial": 1, "performed": "STAR", "target": "STAR", "best": 90, "mean": 85},
        {"condition": "normal", "trial": 1, "performed": "STAR", "target": "TREE", "best": 40, "mean": 30}])
    _write(raw / "normal" / "objects_1.csv", [
        {"condition": "normal", "trial": 1, "object": "cup", "success": 1, "time_s": 2.5, "best_conf": 0.8},
        {"condition": "normal", "trial": 2, "object": "cup", "success": 0, "time_s": "", "best_conf": 0.2}])

    md = summarize(raw, tmp_path, threshold=75)
    acts = metrics.action_summary(metrics.read_rows(raw, "actions"))
    normal = next(r for r in acts if r["condition"] == "normal")
    assert normal["accuracy_pct"] == 50.0 and normal["trials_with_false_trigger"] == 1
    react = metrics.reaction_summary(metrics.read_rows(raw, "reaction"))[0]
    assert react["marked"] == 1 and react["mean_error_ms"] == pytest.approx(100)
    _, rates = metrics.similarity_summary(metrics.read_rows(raw, "similarity"), 75)
    assert rates[0]["correct_accepted_pct"] == 100 and rates[0]["wrong_accepted_pct"] == 0
    assert "| normal | cup | 2 | 1 | 50.0% |" in md
    assert (tmp_path / "summary.md").exists() and (tmp_path / "objects_summary.csv").exists()


def test_summarize_with_no_data(tmp_path):
    md = summarize(tmp_path / "raw", tmp_path, 75)
    assert "No data yet" in md


# --- the test procedures, with a fake camera session ------------------------
LABEL_POSES = {"left hand up": make_pose(LEFT_HAND), "right hand up": make_pose(RIGHT_HAND),
               "both hands up": make_pose({**LEFT_HAND, **RIGHT_HAND})}


class FakeSession:
    """Plays back synthetic poses instead of a camera. `respond(prompt,
    reference, t_since_go)` returns the pose shown after GO."""

    def __init__(self, respond):
        self.respond = respond
        self.now = 0.0
        self.pose_ms = 1.0
        self.duo = False
        self.raw = np.zeros((720, 1280, 3), np.uint8)

    def set_duo(self, duo):
        self.duo = duo

    def frame(self):
        self.now += DT
        return self.raw, [STAND] * (2 if self.duo else 1), self.now

    def show(self, *a, **k):
        pass

    def message(self, *a, **k):
        self.now += 1.0

    def run_trial(self, header, prompt, on_frame, countdown=1.5, record=2.5, wait=0.0, reference=None):
        start = self.now
        go = start + countdown + wait
        while True:
            self.now += DT
            phase = "countdown" if self.now < start + countdown else ("wait" if self.now < go else "go")
            pose = self.respond(prompt, reference, self.now - go) if phase == "go" else STAND
            stop = on_frame(self.raw, [pose], self.now, phase)
            if phase == "go" and (stop or self.now - go >= record):
                return go


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RAW", tmp_path / "raw")
    return tmp_path / "raw"


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_run_actions(raw_dir):
    def respond(prompt, ref, t):
        label = prompt.split("At GO: ")[1].split("  (")[0]
        return LABEL_POSES.get(label, STAND) if t > 0.3 else STAND
    path = runner.run_actions(FakeSession(respond), "normal", trials=2)
    rows = _rows(path)
    assert len(rows) == 6 * 2 + 1
    by = {}
    for r in rows:
        by.setdefault(r["action"], []).append(int(r["hit"]))
    assert by["RIGHT_HAND_UP"] == [1, 1] and by["BOTH_HANDS_UP"] == [1, 1]
    assert by["SQUAT"] == [0, 0]          # the fake player never squats
    assert by[NEUTRAL] == [1]
    assert path.parent.name == "normal"


def test_run_similarity(raw_dir):
    respond = lambda prompt, ref, t: pose_from_target(ref)
    rows = _rows(runner.run_similarity(FakeSession(respond), "normal", reps=1))
    targets = load_targets()
    assert len(rows) == len(targets) ** 2
    for r in rows:
        if r["performed"] == r["target"]:
            assert float(r["best"]) > 99
        else:
            assert float(r["best"]) < config.POSE_MATCH_THRESHOLD


def test_run_objects(raw_dir):
    class FakeDetector:
        def __init__(self, s):
            self.s = s
        def detect(self, raw):
            # Nothing for the first second after the trial starts, then a big box.
            if self.s.now - self.s.go_hint < 1.0:
                return []
            return [Detection(o, 0.9, (0, 0, 640, 360), 0.25) for o in config.OBJECT_CLASSES]

    s = FakeSession(lambda *a: STAND)
    orig = s.run_trial
    def run_trial(*a, **k):
        s.go_hint = s.now + k.get("countdown", 1.5)
        return orig(*a, **k)
    s.run_trial = run_trial
    rows = _rows(runner.run_objects(s, "normal", FakeDetector(s), trials=1))
    assert len(rows) == len(config.OBJECT_CLASSES)
    assert all(r["success"] == "1" for r in rows)
    assert all(0.9 < float(r["time_s"]) < 1.3 for r in rows)


def test_run_reaction_saves_clips_and_times(raw_dir):
    respond = lambda prompt, ref, t: make_pose(RIGHT_HAND) if t > 0.4 else STAND
    path = runner.run_reaction(FakeSession(respond), "normal", trials=2)
    rows = _rows(path)
    assert len(rows) == 2
    for r in rows:
        assert float(r["detected_s"]) == pytest.approx(0.4, abs=0.05)
        clip = raw_dir / r["clip"]
        assert clip.exists() and clip.with_suffix(".csv").exists()


def test_run_fps(raw_dir):
    class FakeDetector:
        def detect(self, raw):
            return []
    rows = _rows(runner.run_fps(FakeSession(None), "normal", seconds=0.05, detector=FakeDetector()))
    assert [r["mode"] for r in rows] == ["solo", "duo", "solo+yolo"]
    assert all(float(r["fps"]) > 0 for r in rows)


def test_quit_keeps_recorded_rows(raw_dir):
    calls = {"n": 0}
    def respond(prompt, ref, t):
        calls["n"] += 1
        if calls["n"] > 200:
            raise runner.Quit
        return make_pose(RIGHT_HAND)
    path = runner.run_actions(FakeSession(respond), "normal", trials=3)
    assert path is not None and 0 < len(_rows(path)) < 19
