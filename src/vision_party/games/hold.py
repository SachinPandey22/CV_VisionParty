"""Hold Challenge: get into the called pose and hold it as long as you can.

The hold timer starts the first time the pose is seen after GO. Losing the
pose for less than HOLD_DROPOUT seconds (a missed detection, a wobble) is
forgiven; a longer gap ends the hold. Holding HOLD_TARGET seconds is a full
success.

Solo: points scale with how long you held. Duo: the longer hold wins.
"""

import random

from .. import config
from ..actions import ACTIONS
from ..pose import Pose
from .base import Game, RoundResult

# Actions that make sense to hold, with their wording.
HOLD_PROMPTS = {
    "BOTH_HANDS_UP": "Keep both hands up!",
    "SQUAT": "Hold a squat!",
    "ONE_LEG_UP": "Balance on one leg!",
    "T_POSE": "Hold a T-pose!",
}


class _Holder:
    def __init__(self):
        self.start: float | None = None     # when the hold began
        self.last_seen: float | None = None  # last frame the pose was detected
        self.end: float | None = None        # when the hold broke (or was completed)

    @property
    def done(self) -> bool:
        return self.end is not None

    def held(self, now: float) -> float:
        """Seconds held so far (or in total, once done)."""
        if self.start is None:
            return 0.0
        stop = self.end if self.end is not None else now
        return max(0.0, stop - self.start)


class HoldChallenge(Game):
    title = "Hold Challenge"
    shrinks = False   # the hold goal is fixed, so a shorter round would be unfair

    def __init__(self, n_players: int, rng: random.Random, action: str | None = None,
                 goal: float = config.HOLD_TARGET,
                 time_limit: float = config.HOLD_TIME_LIMIT,
                 dropout: float = config.HOLD_DROPOUT):
        super().__init__(n_players, rng)
        self.action = action or rng.choice(sorted(HOLD_PROMPTS))
        self._rule = ACTIONS[self.action]
        self.goal = goal
        self._limit = time_limit
        self._dropout = dropout
        self.players = [_Holder() for _ in range(n_players)]

    @property
    def prompt(self) -> str:
        return HOLD_PROMPTS[self.action]

    @property
    def time_limit(self) -> float:
        return self._limit

    def update(self, poses: list[Pose | None], now: float) -> bool:
        for pose, h in zip(poses, self.players):
            if h.done:
                continue
            doing = pose is not None and self._rule(pose)
            if doing:
                if h.start is None:
                    h.start = now
                h.last_seen = now
                if h.held(now) >= self.goal:
                    h.end = h.start + self.goal       # full hold: done
            elif h.start is not None and now - h.last_seen > self._dropout:
                # Gap too long: the hold ended at the last frame it was seen.
                h.end = h.last_seen

        if self.time_left(now) <= 0:
            for h in self.players:
                if h.start is not None and not h.done:
                    h.end = min(now, h.last_seen + self._dropout)
            return True
        # Over early only once every player has finished a hold.
        return all(h.done for h in self.players)

    def _points(self, held: float) -> int:
        """Up to BASE + SPEED_BONUS_MAX (same maximum as the other games)."""
        full = config.BASE_POINTS + config.SPEED_BONUS_MAX
        return round(full * min(held, self.goal) / self.goal)

    def result(self) -> RoundResult:
        held = [h.held(h.end or 0.0) for h in self.players]
        details = [f"held {t:.1f}s" if t > 0 else "no hold" for t in held]
        points = [0] * self.n_players

        if self.n_players == 1:
            points[0] = self._points(held[0])
            if held[0] >= self.goal:
                return RoundResult([0], points, f"Rock solid! {self.goal:.0f}s", details)
            if held[0] > 0:
                return RoundResult([], points, f"Held {held[0]:.1f}s", details)
            return RoundResult([], points, "No hold!", details)

        best = max(held)
        if best <= 0:
            return RoundResult([], points, "Nobody held it!", details)
        # Holds within a tenth of a second count as a tie.
        winners = [i for i, t in enumerate(held) if best - t < 0.1]
        for i in winners:
            points[i] = self._points(held[i])
        if len(winners) == 1:
            msg = f"P{winners[0] + 1} wins! {best:.1f}s"
        else:
            msg = f"Tie! {best:.1f}s"
        return RoundResult(winners, points, msg, details)

    def status(self, now: float) -> list[str]:
        out = []
        for h in self.players:
            if h.start is None:
                out.append("get in position!")
            elif h.done:
                out.append(f"done: {h.held(now):.1f}s")
            else:
                out.append(f"holding {h.held(now):.1f}s / {self.goal:.0f}s")
        return out
