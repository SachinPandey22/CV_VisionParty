"""Pose Battle: copy the target pose shown on screen.

A player "matches" when their joint-angle similarity to the target stays at
or above POSE_MATCH_THRESHOLD for POSE_CONFIRM_FRAMES frames in a row. The
time is taken from the first of those frames.

Points: 100 base + up to 50 speed bonus + up to 50 for accuracy (how far the
best similarity went above the threshold).
Solo: match before the timer runs out. Duo: first to match wins.
"""

import random

from .. import config
from ..pose import Pose
from ..similarity import TargetPose, load_targets, pose_similarity
from .base import Game, RoundResult, speed_bonus


class _Copier:
    def __init__(self):
        self.score: float | None = None    # similarity this frame (0-100)
        self.best = 0.0                    # best similarity seen this round
        self.streak = 0
        self.streak_start: float | None = None
        self.time: float | None = None     # seconds from GO, once matched


class PoseBattle(Game):
    title = "Pose Battle"

    def __init__(self, n_players: int, rng: random.Random,
                 target: TargetPose | None = None,
                 targets: list[TargetPose] | None = None,
                 time_limit: float = config.POSE_TIME_LIMIT):
        super().__init__(n_players, rng)
        self.target = target or rng.choice(targets or load_targets())
        self._limit = time_limit
        self.players = [_Copier() for _ in range(n_players)]

    @property
    def prompt(self) -> str:
        return self.target.prompt

    @property
    def time_limit(self) -> float:
        return self._limit

    def update(self, poses: list[Pose | None], now: float) -> bool:
        for pose, c in zip(poses, self.players):
            if c.time is not None:
                continue
            c.score = pose_similarity(pose, self.target)
            if c.score is not None:
                c.best = max(c.best, c.score)
            if c.score is None or c.score < config.POSE_MATCH_THRESHOLD:
                c.streak, c.streak_start = 0, None
                continue
            if c.streak == 0:
                c.streak_start = now
            c.streak += 1
            if c.streak >= config.POSE_CONFIRM_FRAMES:
                c.time = c.streak_start - self.start_time

        matched = [c.time is not None for c in self.players]
        done = matched[0] if self.n_players == 1 else any(matched)
        return done or self.time_left(now) <= 0

    def _accuracy_bonus(self, best: float) -> int:
        span = 100 - config.POSE_MATCH_THRESHOLD
        frac = (best - config.POSE_MATCH_THRESHOLD) / span if span > 0 else 1.0
        return round(config.POSE_ACCURACY_MAX * max(0.0, min(1.0, frac)))

    def result(self) -> RoundResult:
        times = [c.time for c in self.players]
        details = [f"{c.time:.2f}s  {c.best:.0f}%" if c.time is not None else f"best {c.best:.0f}%"
                   for c in self.players]
        points = [0] * self.n_players
        valid = [t for t in times if t is not None]
        if not valid:
            best = max(c.best for c in self.players)
            msg = f"So close! {best:.0f}%" if best > 0 else "Nobody got it!"
            return RoundResult([], points, msg, details, times)

        first = min(valid)
        winners = [i for i, t in enumerate(times) if t is not None and t == first]
        for i in winners:
            points[i] = (config.BASE_POINTS
                         + speed_bonus(first, config.REACTION_FAST * 2, self.effective_limit)
                         + self._accuracy_bonus(self.players[i].best))
        accuracy = [c.best if c.time is not None else None for c in self.players]
        name = self.target.name
        if self.n_players == 1:
            msg = f"{name}! {self.players[0].best:.0f}% match"
        elif len(winners) == 1:
            msg = f"P{winners[0] + 1} nailed it! {self.players[winners[0]].best:.0f}%"
        else:
            msg = f"Tie! {first:.2f}s"
        return RoundResult(winners, points, msg, details, times, accuracy)

    def status(self, now: float) -> list[str]:
        out = []
        for c in self.players:
            if c.time is not None:
                out.append(f"matched {c.best:.0f}%")
            elif c.score is None:
                out.append("step into view")
            else:
                out.append(f"match {c.score:.0f}%")
        return out
