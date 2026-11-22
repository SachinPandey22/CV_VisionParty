import random

from vision_party import config
from vision_party.engine import Match, Phase
from vision_party.games import ReactionBattle

from poses import RIGHT_HAND, make_pose

STAND = make_pose()
RIGHT = make_pose(RIGHT_HAND)
DT = 1 / 30


def right_hand_game(n, rng):
    return ReactionBattle(n, rng, action="RIGHT_HAND_UP")


class Clock:
    def __init__(self):
        self.t = 0.0

    def run(self, match, poses, seconds):
        for _ in range(int(seconds / DT)):
            self.t += DT
            match.update(poses, self.t)


def new_match(names, rounds=2):
    return Match(names, 0.0, rounds=rounds, games=[right_hand_game], rng=random.Random(1))


def test_phases_progress_and_scores_accumulate():
    m, clock = new_match(["Ann"]), Clock()
    assert m.phase is Phase.INTRO and m.round_index == 0
    clock.run(m, [STAND], config.INTRO_SECONDS + 0.05)
    assert m.phase is Phase.COUNTDOWN
    clock.run(m, [STAND], config.COUNTDOWN_SECONDS)
    assert m.phase is Phase.ACTIVE
    clock.run(m, [RIGHT], 0.3)
    assert m.phase is Phase.RESULT
    assert m.scores[0] > 0
    first = m.scores[0]

    clock.run(m, [STAND], config.RESULT_SECONDS + 0.05)
    assert m.phase is Phase.INTRO and m.round_index == 1
    clock.run(m, [STAND], config.INTRO_SECONDS + config.COUNTDOWN_SECONDS + 0.1)
    clock.run(m, [RIGHT], 0.3)
    clock.run(m, [STAND], config.RESULT_SECONDS + 0.05)
    assert m.phase is Phase.FINISHED
    assert m.scores[0] > first
    assert len(m.history) == 2


def test_countdown_values_and_announcements():
    m, clock = new_match(["Ann"]), Clock()
    said = m.pop_announcements()
    assert said == ["Raise your right hand!"]
    clock.run(m, [STAND], config.INTRO_SECONDS + config.COUNTDOWN_SECONDS + 0.1)
    said = m.pop_announcements()
    assert said == ["3", "2", "1", "Go!"]


def test_duo_winner_named_in_announcement():
    m, clock = new_match(["Ann", "Bob"], rounds=1), Clock()
    clock.run(m, [STAND, STAND], config.INTRO_SECONDS + config.COUNTDOWN_SECONDS + 0.1)
    m.pop_announcements()
    clock.run(m, [STAND, RIGHT], 0.3)
    assert m.phase is Phase.RESULT
    assert m.pop_announcements()[0].startswith("Bob wins")
    clock.run(m, [STAND, STAND], config.RESULT_SECONDS + 0.05)
    assert m.phase is Phase.FINISHED
    assert m.winners() == [1]
    assert m.pop_announcements() == ["Bob wins the match!"]


def test_holding_pose_during_countdown_does_not_win_instantly():
    m, clock = new_match(["Ann"]), Clock()
    clock.run(m, [RIGHT], config.INTRO_SECONDS + config.COUNTDOWN_SECONDS + 0.5)
    assert m.phase is Phase.ACTIVE
