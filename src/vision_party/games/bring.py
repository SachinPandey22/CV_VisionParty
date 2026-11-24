"""Bring to Camera (Solo only): fetch the named object and hold it up close.

Success = YOLO sees the object and its box covers at least
OBJECT_CLOSE_AREA[object] of the frame for OBJECT_CONFIRM_FRAMES frames in a
row. Like Reaction Battle, the object must first be seen *not* close after GO
(or not at all), so leaving it next to the camera doesn't win instantly.

This game reads object detections, not poses: main.py runs the detector and
calls `observe` each frame before the engine calls `update`.
"""

import random

from .. import config
from ..objects import Detection
from ..pose import Pose
from .base import Game, RoundResult, speed_bonus

ARTICLE = {"bottle": "a bottle", "cup": "a cup", "book": "a book", "cell phone": "a phone"}


class BringToCamera(Game):
    title = "Bring to Camera"
    solo_only = True
    uses_objects = True

    def __init__(self, n_players: int, rng: random.Random, target: str | None = None,
                 time_limit: float = config.OBJECT_TIME_LIMIT):
        super().__init__(n_players, rng)
        self.target = target or rng.choice(config.OBJECT_CLASSES)
        self._limit = time_limit
        self._close_area = config.OBJECT_CLOSE_AREA[self.target]
        self._armed = False
        self._streak = 0
        self._streak_start: float | None = None
        self.time_taken: float | None = None
        self.closeness = 0.0          # latest area / required area, for the UI
        self.box: tuple[int, int, int, int] | None = None  # latest box of the target
        # None = no detector output yet (model still loading). The game waits
        # instead of treating that as "object not close", which would arm it.
        self._detections: list[Detection] | None = None

    @property
    def prompt(self) -> str:
        return f"Bring {ARTICLE[self.target]} to the camera!"

    @property
    def time_limit(self) -> float:
        return self._limit

    def observe(self, detections: list[Detection] | None) -> None:
        """Latest object detections for this frame."""
        self._detections = detections

    def update(self, poses: list[Pose | None], now: float) -> bool:
        if self.time_taken is None and self._detections is not None:
            matches = [d for d in self._detections if d.name == self.target]
            best = max(matches, key=lambda d: d.area_frac, default=None)
            self.box = best.box if best else None
            self.closeness = best.area_frac / self._close_area if best else 0.0
            close = self.closeness >= 1.0

            if not close:
                self._armed = True
                self._streak, self._streak_start = 0, None
            elif self._armed:
                if self._streak == 0:
                    self._streak_start = now
                self._streak += 1
                if self._streak >= config.OBJECT_CONFIRM_FRAMES:
                    self.time_taken = self._streak_start - self.start_time
        return self.time_taken is not None or self.time_left(now) <= 0

    def result(self) -> RoundResult:
        name = ARTICLE[self.target].split(" ", 1)[1]
        if self.time_taken is None:
            return RoundResult([], [0], f"No {name}? Time's up!", ["not close enough"], [None])
        t = self.time_taken
        points = config.BASE_POINTS + speed_bonus(t, config.OBJECT_FAST, self.effective_limit)
        return RoundResult([0], [points], f"Got the {name}! {t:.1f}s", [f"{t:.1f}s"], [t])

    def status(self, now: float) -> list[str]:
        if self.time_taken is not None:
            return [f"{self.time_taken:.1f}s"]
        if self._detections is None:
            return ["starting object detector..."]
        if not self._armed:
            return ["move it away first"]
        if self.closeness <= 0:
            return [f"looking for {ARTICLE[self.target]}..."]
        return [f"closer! {min(99, int(self.closeness * 100))}%"]
