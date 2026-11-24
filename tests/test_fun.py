"""Tests for the playful extras: ratings, streaks, props, confetti, photos."""

import random
import types

import numpy as np

from vision_party import config
from vision_party.effects import Confetti, draw_crown, draw_party_hat, draw_props, head_frame
from vision_party.engine import Match, Phase
from vision_party.fun import Moments, display_region, pick_frame, rating, streak_label
from vision_party.games import ReactionBattle
from vision_party.main import crown_for
from vision_party.pose import L

from poses import RIGHT_HAND, make_pose

FACE = {L.LEFT_EYE: (515, 140), L.RIGHT_EYE: (485, 140),
        L.LEFT_EAR: (535, 145), L.RIGHT_EAR: (465, 145)}
STAND = make_pose()
RIGHT = make_pose(RIGHT_HAND)
DT = 1 / 30


# --- ratings and streak badges ---------------------------------------------
def test_rating_levels():
    assert rating(None) is None
    assert rating(0.3) == ("PERFECT!", 0)
    assert rating(config.RATING_PERFECT) == ("PERFECT!", 0)
    assert rating(0.8) == ("GREAT!", 1)
    assert rating(1.5) == ("GOOD", 2)
    assert rating(4.0) == ("OK", 3)


def test_streak_label():
    assert streak_label(0) == "" and streak_label(1) == ""
    assert streak_label(2) == "2 in a row"
    assert streak_label(config.STREAK_ON_FIRE) == f"ON FIRE x{config.STREAK_ON_FIRE}"


# --- engine: streaks, leader, reaction times in results ---------------------
def _match(names, rounds):
    game = lambda n, rng: ReactionBattle(n, rng, action="RIGHT_HAND_UP")
    return Match(names, 0.0, rounds=rounds, games=[game], rng=random.Random(1))


def _play_round(m, t, poses_win):
    """Run one full round where `poses_win` is shown during ACTIVE."""
    idle = [STAND] * m.n_players
    for _ in range(int((config.INTRO_SECONDS + config.COUNTDOWN_SECONDS + 0.1) / DT)):
        t += DT
        m.update(idle, t)
    for _ in range(int(0.4 / DT)):
        t += DT
        m.update(poses_win, t)
    for _ in range(int((config.RESULT_SECONDS + 0.05) / DT)):
        t += DT
        m.update(idle, t)
    return t


def test_round_result_carries_reaction_times():
    m, t = _match(["Ann", "Bob"], rounds=2), 0.0
    for _ in range(int((config.INTRO_SECONDS + config.COUNTDOWN_SECONDS + 0.1) / DT)):
        t += DT
        m.update([STAND, STAND], t)
    for _ in range(int(0.4 / DT)):
        t += DT
        m.update([STAND, RIGHT], t)
    res = m.last_result
    assert res.times[0] is None and res.times[1] is not None
    assert rating(res.times[1])[0] == "PERFECT!"


def test_streaks_count_wins_and_reset():
    m, t = _match(["Ann", "Bob"], rounds=5), 0.0
    t = _play_round(m, t, [STAND, RIGHT])
    t = _play_round(m, t, [STAND, RIGHT])
    assert m.streaks == [0, 2]
    m.pop_announcements()
    t = _play_round(m, t, [STAND, RIGHT])
    assert m.streaks == [0, config.STREAK_ON_FIRE]
    assert "Bob is on fire!" in m.pop_announcements()
    t = _play_round(m, t, [RIGHT, STAND])
    assert m.streaks == [1, 0]


def test_leader_and_crown():
    m, t = _match(["Ann", "Bob"], rounds=3), 0.0
    assert m.leader() is None and crown_for(m) is None   # 0-0: nobody
    t = _play_round(m, t, [RIGHT, STAND])
    assert m.leader() == 0 and crown_for(m) == 0
    t = _play_round(m, t, [STAND, RIGHT])
    # Ann and Bob both won one round; points may differ by speed bonus, so
    # just check the crown follows the leader.
    assert crown_for(m) == m.leader()


def test_solo_crown_is_earned():
    m, t = _match(["Ann"], rounds=config.STREAK_ON_FIRE), 0.0
    t = _play_round(m, t, [RIGHT])
    assert crown_for(m) is None                          # one win is not enough
    for _ in range(config.STREAK_ON_FIRE - 1):
        t = _play_round(m, t, [RIGHT])
    assert m.phase is Phase.FINISHED and crown_for(m) == 0


# --- props ------------------------------------------------------------------
def test_head_frame_points_up_and_scales_with_ears():
    pose = make_pose(FACE)
    eye_mid, right, up, head_w = head_frame(pose, 1280)
    assert up[1] < 0                       # 'up' is toward the top of the image
    assert right[0] > 0
    assert abs(head_w - 70) < 1            # ear-to-ear distance


def test_head_frame_needs_eyes():
    assert head_frame(make_pose(), 1280) is None


def test_props_draw_on_the_head():
    img = np.zeros((720, 1280, 3), np.uint8)
    pose = make_pose(FACE)
    draw_party_hat(img, pose, (255, 0, 0))
    ys, xs = np.nonzero(img.any(axis=2))
    assert len(ys) > 0
    assert ys.max() < 140                  # entirely above the eyes
    nose_x_display = 1280 - 1 - 500
    assert abs(xs.mean() - nose_x_display) < 20

    img2 = np.zeros_like(img)
    draw_crown(img2, pose)
    assert img2.any()


def test_draw_props_skips_missing_players():
    img = np.zeros((720, 1280, 3), np.uint8)
    draw_props(img, [None, make_pose(FACE)], [(255, 0, 0), (0, 255, 0)], crown_for=None)
    assert img.any()


# --- confetti ---------------------------------------------------------------
def test_confetti_bursts_falls_and_is_capped():
    c = Confetti(seed=0)
    c.burst(640, 300)
    assert len(c) == config.CONFETTI_BURST
    for _ in range(20):
        c.burst(640, 300)
    assert len(c) == config.CONFETTI_MAX
    for _ in range(300):                   # 10 s at 30 FPS: everything falls off
        c.update(1 / 30, 720)
    assert len(c) == 0


def test_confetti_draws():
    c = Confetti(seed=0)
    c.burst(640, 300)
    img = np.zeros((720, 1280, 3), np.uint8)
    c.draw(img)
    assert img.any()


# --- best-moment photos -----------------------------------------------------
def _frame(value, w=1280):
    return np.full((720, w, 3), value, np.uint8)


def test_pick_frame_nearest_time():
    frames = [(1.0, _frame(1)), (1.1, _frame(2)), (1.2, _frame(3))]
    assert pick_frame(frames, 1.12)[0, 0, 0] == 2
    assert pick_frame([], 1.0) is None


def test_display_region():
    assert display_region(0, 1, 1280) == (0, 1280)
    assert display_region(0, 2, 1280) == (0, 640)
    assert display_region(1, 2, 1280) == (640, 1280)


def test_moments_keep_each_players_fastest():
    m = Moments(2, buffer=5)
    for k in range(5):
        m.push(k * 0.1, _frame(k))
    assert m.record(1, 0.8, 0.2)
    reaction, photo = m.best[1]
    assert reaction == 0.8 and photo.shape[1] == 640 and photo[0, 0, 0] == 2
    assert not m.record(1, 0.9, 0.3)       # slower: keep the old photo
    assert m.record(1, 0.5, 0.4)           # faster: replace it
    assert m.best[1][1][0, 0, 0] == 4
    assert m.best[0] is None


# --- the full app loop, with fake frames and time ---------------------------
def test_app_runs_full_match_with_effects(monkeypatch):
    from vision_party import main as app_mod
    clock = [0.0]
    monkeypatch.setattr(app_mod.time, "perf_counter", lambda: clock[0])
    import vision_party.engine as engine
    monkeypatch.setattr(engine, "DEFAULT_GAMES",
                        [lambda n, rng: ReactionBattle(n, rng, action="RIGHT_HAND_UP")])
    args = types.SimpleNamespace(p1="Ann", p2="Bob", rounds=1, camera=0, save_moments=False,
                                 mute=True, no_objects=True)
    app = app_mod.App(args)
    frame = np.zeros((720, 1280, 3), np.uint8)
    app.frame(frame, -1)
    app.frame(frame, ord("2"))

    # Fake pose estimators: Bob raises his hand once the round is active.
    class Fake:
        def __init__(self, i):
            self.i = i
        def process(self, raw):
            active = app.match.phase is not Phase.INTRO and app.match.phase is not Phase.COUNTDOWN
            return make_pose({**FACE, **RIGHT_HAND}) if (self.i == 1 and active) else make_pose(FACE)
        def close(self):
            pass
    app.estimators = [Fake(0), Fake(1)]

    for _ in range(400):
        clock[0] += DT
        view, running = app.frame(frame, -1)
        assert running and view.shape == frame.shape
        if app.match.phase is Phase.FINISHED:
            break
    assert app.match.phase is Phase.FINISHED
    assert app.moments.best[1] is not None   # Bob's photo was kept
    assert len(app.confetti) > 0
