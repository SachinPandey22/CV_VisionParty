"""Tests for Hold Challenge, pose similarity, Pose Battle and pose capture."""

import random

import numpy as np

from vision_party import config
from vision_party.capture_poses import average_target
from vision_party.fun import accuracy_rating
from vision_party.games import HoldChallenge, PoseBattle
from vision_party.pose import Pose
from vision_party.similarity import (BODY, TARGETS_FILE, load_targets, pose_similarity,
                                     save_targets, similarity)

from poses import LEFT_HAND, RIGHT_HAND, make_pose

STAND = make_pose()
HANDS_UP = make_pose({**LEFT_HAND, **RIGHT_HAND})
DT = 1 / 30


def run(game, schedule, start=0.0):
    """Feed (seconds, poses) chunks to a game. Returns (finished, time)."""
    t = start
    for seconds, poses in schedule:
        for _ in range(round(seconds / DT)):
            t += DT
            if game.update(poses, t):
                return True, t
    return False, t


# --- Hold Challenge ---------------------------------------------------------
def hold(n=1, **kw):
    g = HoldChallenge(n, random.Random(0), action="BOTH_HANDS_UP", **kw)
    g.begin(0.0)
    return g


def test_full_hold_succeeds_and_ends_round():
    g = hold()
    done, t = run(g, [(0.5, [STAND]), (config.HOLD_TARGET + 0.5, [HANDS_UP])])
    assert done
    res = g.result()
    assert res.winners == [0]
    assert res.points[0] == config.BASE_POINTS + config.SPEED_BONUS_MAX


def test_short_dropout_is_forgiven():
    g = hold()
    # Hold 2 s, lose the pose for 0.2 s (< HOLD_DROPOUT), hold 2 s more, then drop.
    done, _ = run(g, [(2.0, [HANDS_UP]), (0.2, [STAND]), (2.0, [HANDS_UP]), (1.0, [STAND])])
    assert done
    held = g.players[0].held(0)
    assert 4.0 < held < 4.4


def test_long_dropout_breaks_hold():
    g = hold()
    done, _ = run(g, [(2.0, [HANDS_UP]), (config.HOLD_DROPOUT + 0.2, [STAND]),
                      (3.0, [HANDS_UP])])
    assert done                                 # solo round ends when the hold breaks
    res = g.result()
    assert res.winners == []
    assert 1.9 < g.players[0].held(0) < 2.1
    assert 0 < res.points[0] < config.BASE_POINTS


def test_missing_pose_counts_as_losing_it():
    g = hold()
    run(g, [(1.0, [HANDS_UP]), (1.0, [None])])
    assert g.players[0].done


def test_duo_longer_hold_wins():
    g = hold(2)
    done, _ = run(g, [(1.5, [HANDS_UP, HANDS_UP]), (1.0, [STAND, HANDS_UP]),
                      (1.0, [STAND, STAND])])
    assert done
    res = g.result()
    assert res.winners == [1]
    assert res.points[1] > 0 and res.points[0] == 0
    assert "P2 wins" in res.message


def test_never_holding_times_out_with_no_points():
    g = hold(time_limit=2.0)
    done, _ = run(g, [(3.0, [STAND])])
    assert done
    res = g.result()
    assert res.points == [0] and res.message == "No hold!"


# --- similarity -------------------------------------------------------------
def pose_from_target(target, scale=180.0, offset=(640, 360)) -> Pose:
    pts = np.zeros((33, 2), np.float32)
    vis = np.zeros(33, np.float32)
    for k, (x, y) in target.points.items():
        pts[k] = [x * scale + offset[0], y * scale + offset[1]]
        vis[k] = 0.99
    return Pose(pts, vis)


def test_five_targets_load_and_are_distinct():
    targets = load_targets()
    assert len(targets) == 5
    for a in targets:
        assert len(a.angles) == 8 and set(a.points) == set(BODY)
        for b in targets:
            if a is not b:
                assert similarity(a.angles, b.angles) < config.POSE_MATCH_THRESHOLD


def test_similarity_perfect_and_scale_invariant():
    t = load_targets()[0]
    assert pose_similarity(pose_from_target(t), t) > 99
    # Same pose, smaller and somewhere else in the frame: same score.
    assert pose_similarity(pose_from_target(t, scale=90, offset=(300, 500)), t) > 99


def test_similarity_falls_with_angle_error():
    target = {"a": 90.0, "b": 90.0, "c": 90.0, "d": 90.0, "e": 90.0, "f": 90.0}
    assert similarity(dict(target), target) == 100
    half = {k: 90.0 + config.SIM_MAX_DIFF / 2 for k in target}
    assert abs(similarity(half, target) - 50) < 1e-6
    far = {k: 0.0 for k in target}
    assert similarity(far, target) == 0


def test_similarity_needs_enough_joints():
    target = {f"j{i}": 90.0 for i in range(8)}
    few = {f"j{i}": 90.0 for i in range(config.SIM_MIN_JOINTS - 1)}
    assert similarity(few, target) is None
    assert pose_similarity(None, load_targets()[0]) is None


def test_accuracy_rating():
    assert accuracy_rating(None) is None
    assert accuracy_rating(97)[0] == "PERFECT!"
    assert accuracy_rating(90)[0] == "GREAT!"
    assert accuracy_rating(83)[0] == "GOOD"
    assert accuracy_rating(76)[0] == "OK"


# --- Pose Battle ------------------------------------------------------------
def test_pose_battle_solo_match_scores_with_accuracy_bonus():
    target = load_targets()[1]
    g = PoseBattle(1, random.Random(0), target=target)
    g.begin(0.0)
    done, _ = run(g, [(0.5, [STAND]), (1.0, [pose_from_target(target)])])
    assert done
    res = g.result()
    assert res.winners == [0]
    assert abs(res.times[0] - 0.5) < 0.05
    assert res.accuracy[0] > 99
    # 100 base + speed bonus + full accuracy bonus
    assert res.points[0] >= config.BASE_POINTS + config.POSE_ACCURACY_MAX


def test_pose_battle_wrong_pose_does_not_match():
    targets = load_targets()
    g = PoseBattle(1, random.Random(0), target=targets[0], time_limit=1.0)
    g.begin(0.0)
    done, _ = run(g, [(1.5, [pose_from_target(targets[2])])])
    assert done
    res = g.result()
    assert res.winners == [] and res.points == [0]
    assert res.message.startswith("So close!")


def test_pose_battle_duo_first_to_match_wins():
    target = load_targets()[3]
    g = PoseBattle(2, random.Random(0), target=target)
    g.begin(0.0)
    done, _ = run(g, [(0.3, [STAND, STAND]), (1.0, [STAND, pose_from_target(target)])])
    assert done
    res = g.result()
    assert res.winners == [1] and res.points[0] == 0
    assert res.accuracy[0] is None and res.accuracy[1] > 99


def test_pose_battle_prompt_names_the_pose():
    target = load_targets()[0]
    g = PoseBattle(1, random.Random(0), target=target)
    assert target.name in g.prompt


# --- capture tool -----------------------------------------------------------
def test_average_target_and_save_roundtrip(tmp_path):
    base = load_targets()[0]
    poses = [pose_from_target(base, offset=(640 + d, 360)) for d in (-3, 0, 3)] + [None]
    t = average_target("STAR", "Star", poses)
    assert set(t.angles) == set(base.angles)
    assert similarity(t.angles, base.angles) > 99
    path = tmp_path / "targets.json"
    save_targets([t], path)
    again = load_targets(path)[0]
    assert again.name == "Star" and again.angles == t.angles and set(again.points) == set(BODY)


def test_average_target_with_no_frames():
    assert average_target("X", "X", [None, None]) is None


def test_targets_file_is_packaged():
    assert TARGETS_FILE.exists()


# --- every game's screens draw without errors --------------------------------
def test_ui_draws_every_game_in_every_phase():
    from vision_party import ui
    from vision_party.engine import Match, Phase
    from vision_party.fun import Moments
    from vision_party.games import ReactionBattle
    factories = [lambda n, r: ReactionBattle(n, r, action="RIGHT_HAND_UP"),
                 lambda n, r: HoldChallenge(n, r, action="BOTH_HANDS_UP"),
                 lambda n, r: PoseBattle(n, r, target=load_targets()[0])]
    for factory in factories:
        for names in (["Ann"], ["Ann", "Bob"]):
            m = Match(names, 0.0, rounds=1, games=[factory], rng=random.Random(0))
            seen, t = set(), 0.0
            poses = [HANDS_UP] * len(names)
            while m.phase is not Phase.FINISHED and t < 30:
                t += DT
                m.update(poses, t)
                img = np.zeros((720, 1280, 3), np.uint8)
                ui.draw_match(img, m, poses, t, Moments(len(names)))
                seen.add(m.phase)
            assert m.phase is Phase.FINISHED
            assert {Phase.INTRO, Phase.COUNTDOWN, Phase.ACTIVE, Phase.RESULT} <= seen
