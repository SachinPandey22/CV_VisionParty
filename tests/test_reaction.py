import random

from vision_party import config
from vision_party.games import ReactionBattle

from poses import LEFT_HAND, RIGHT_HAND, make_pose

STAND = make_pose()
RIGHT = make_pose(RIGHT_HAND)
LEFT = make_pose(LEFT_HAND)
DT = 1 / 30
N = config.REACTION_CONFIRM_FRAMES


def game(n_players=1, timeout=5.0):
    g = ReactionBattle(n_players, random.Random(0), action="RIGHT_HAND_UP", timeout=timeout)
    g.begin(0.0)
    return g


def feed(g, frames, t0=0.0):
    """frames: list of per-frame pose lists. Returns (done, time after last frame)."""
    t = t0
    done = False
    for poses in frames:
        done = g.update(poses, t)
        t += DT
        if done:
            break
    return done, t


def test_solo_success_records_reaction_from_first_frame():
    g = game()
    # 10 frames standing, then the action held.
    done, _ = feed(g, [[STAND]] * 10 + [[RIGHT]] * N)
    assert done
    r = g.result()
    assert r.winners == [0]
    assert abs(g.players[0].reaction - 10 * DT) < 1e-6
    assert r.points[0] >= config.BASE_POINTS


def test_flicker_shorter_than_confirm_frames_does_not_count():
    g = game()
    frames = ([[STAND]] + [[RIGHT]] * (N - 1)) * 5
    done, _ = feed(g, frames)
    assert not done
    assert g.players[0].reaction is None


def test_holding_pose_through_countdown_must_drop_it_first():
    g = game()
    done, _ = feed(g, [[RIGHT]] * 20)
    assert not done, "holding the pose from before GO must not count"
    assert g.status(0.5) == ["drop the pose first"]
    done, _ = feed(g, [[STAND]] + [[RIGHT]] * N, t0=1.0)
    assert done


def test_wrong_action_does_not_count():
    g = game()
    done, _ = feed(g, [[LEFT]] * 30)
    assert not done


def test_solo_timeout_scores_zero():
    g = game(timeout=1.0)
    done, _ = feed(g, [[STAND]] * 40)
    assert done
    r = g.result()
    assert r.winners == [] and r.points == [0]


def test_duo_first_player_wins():
    g = game(n_players=2)
    frames = [[STAND, STAND]] * 5 + [[STAND, RIGHT]] * N
    done, _ = feed(g, frames)
    assert done
    r = g.result()
    assert r.winners == [1]
    assert r.points[0] == 0 and r.points[1] > 0
    assert r.message.startswith("P2 wins")


def test_duo_missing_player_does_not_crash():
    g = game(n_players=2)
    done, _ = feed(g, [[None, STAND]] * 3 + [[RIGHT, None]] * N)
    assert done and g.result().winners == [0]


def test_duo_same_frame_is_a_tie():
    g = game(n_players=2)
    done, _ = feed(g, [[STAND, STAND]] + [[RIGHT, RIGHT]] * N)
    r = g.result()
    assert done and r.winners == [0, 1] and r.message.startswith("Tie")


def test_faster_reaction_gets_more_points():
    fast, slow = game(), game()
    feed(fast, [[STAND]] * 3 + [[RIGHT]] * N)
    feed(slow, [[STAND]] * 60 + [[RIGHT]] * N)
    assert fast.result().points[0] > slow.result().points[0]
