"""Bring to Camera, solo-only filtering, shrinking timers and the voice queue."""

import random

import pytest

from vision_party import config
from vision_party.engine import DEFAULT_GAMES, Match
from vision_party.games import BringToCamera, HoldChallenge, ReactionBattle
from vision_party.objects import Detection
from vision_party.voice import Voice

DT = 1 / 30
N = config.OBJECT_CONFIRM_FRAMES
NEED = config.OBJECT_CLOSE_AREA["bottle"]


def det(name="bottle", area=0.01, conf=0.8):
    return Detection(name, conf, (100, 100, 200, 300), area)


FAR, CLOSE = [det(area=NEED / 4)], [det(area=NEED * 1.5)]


def bring_game(limit=12.0):
    g = BringToCamera(1, random.Random(0), target="bottle", time_limit=limit)
    g.begin(0.0)
    return g


def feed(g, frames, t0=0.0):
    t, done = t0, False
    for dets in frames:
        g.observe(dets)
        done = g.update([None], t)
        t += DT
        if done:
            break
    return done


# --- Bring to Camera ----------------------------------------------------------
def test_bring_success_time_from_first_close_frame():
    g = bring_game()
    assert feed(g, [FAR] * 30 + [CLOSE] * N)
    r = g.result()
    assert r.winners == [0] and r.points[0] >= config.BASE_POINTS
    assert g.time_taken == pytest.approx(30 * DT)


def test_bring_object_already_close_at_go_must_move_away_first():
    g = bring_game()
    assert not feed(g, [CLOSE] * 20)
    assert g.status(1.0) == ["move it away first"]
    assert feed(g, [FAR] + [CLOSE] * N, t0=1.0)


def test_bring_not_seen_at_all_counts_as_away():
    g = bring_game()
    assert feed(g, [[]] * 5 + [CLOSE] * N)


def test_bring_wrong_object_does_not_count():
    g = bring_game()
    assert not feed(g, [[det("cup", area=0.9)]] * 30)


def test_bring_flicker_does_not_count():
    g = bring_game()
    assert not feed(g, ([FAR] + [CLOSE] * (N - 1)) * 10)


def test_bring_waits_while_detector_loading():
    g = bring_game()
    assert not feed(g, [None] * 30)
    assert g.status(1.0) == ["starting object detector..."]
    assert not g._armed


def test_bring_timeout():
    g = bring_game(limit=1.0)
    assert feed(g, [FAR] * 40)
    r = g.result()
    assert r.winners == [] and r.points == [0]


def test_bring_prompt_and_phone_wording():
    g = BringToCamera(1, random.Random(0), target="cell phone")
    assert g.prompt == "Bring a phone to the camera!"


def test_bring_uses_largest_box_of_target():
    g = bring_game()
    feed(g, [[det(area=0.001), det(area=NEED / 2)]])
    assert g.closeness == pytest.approx(0.5)


# --- Engine: solo-only games and timer shrink ----------------------------------
def test_duo_never_gets_solo_only_games():
    m = Match(["A", "B"], 0.0, rng=random.Random(0))
    assert BringToCamera not in m.games
    assert BringToCamera in Match(["A"], 0.0).games


def test_only_solo_games_in_duo_raises():
    with pytest.raises(ValueError):
        Match(["A", "B"], 0.0, games=[BringToCamera])


def test_time_scale_shrinks_then_floors():
    scales = [Match.time_scale(i) for i in range(8)]
    assert scales[0] == 1.0
    assert all(a >= b for a, b in zip(scales, scales[1:]))
    assert min(scales) == config.TIMER_MIN_SCALE


def test_later_rounds_get_shorter_limits_but_hold_does_not():
    m = Match(["A"], 0.0, games=[ReactionBattle], rng=random.Random(0))
    m.round_index = 3
    m._next_round(0.0)   # round index 4
    assert m.game.effective_limit == pytest.approx(config.REACTION_TIMEOUT * Match.time_scale(4))

    h = Match(["A"], 0.0, games=[HoldChallenge], rng=random.Random(0))
    h.round_index = 3
    h._next_round(0.0)
    assert h.game.effective_limit == config.HOLD_TIME_LIMIT


def test_default_games_include_bring():
    assert BringToCamera in DEFAULT_GAMES


# --- Voice -----------------------------------------------------------------
def test_voice_drops_stale_lines(monkeypatch):
    v = Voice(enabled=False)
    v._backend = "say"                # pretend a real backend, but don't start speaking
    monkeypatch.setattr(v._wake, "put", lambda item: None)
    for i in range(10):
        v.say(str(i))
    assert list(v._pending) == [str(i) for i in range(10 - config.VOICE_MAX_QUEUE, 10)]


def test_voice_muted_never_queues():
    v = Voice(enabled=False)
    v.say("hello")
    assert not v._pending
    v.close()
