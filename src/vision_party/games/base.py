"""Common interface for all games.

The engine drives a game like this:

    game = SomeGame(n_players, rng)   # picks the challenge; `prompt` is now set
    ... intro + countdown (engine) ...
    game.begin(now)                   # at "GO"
    while not game.update(poses, now): ...
    result = game.result()

Games never draw or speak; they only read poses and report state.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .. import config
from ..pose import Pose


@dataclass
class RoundResult:
    winners: list[int]             # indices of players who won (Duo) or succeeded (Solo)
    points: list[int]              # points earned this round, per player
    message: str                   # headline, e.g. "P1 wins!"
    details: list[str] = field(default_factory=list)  # one line per player
    times: list[float | None] = field(default_factory=list)  # reaction time per player, if any
    accuracy: list[float | None] = field(default_factory=list)  # pose match % per player, if any


class Game(ABC):
    title: str = "Game"
    solo_only: bool = False      # engine skips this game in Duo
    uses_objects: bool = False   # needs object detections via `observe` (see bring.py)
    shrinks: bool = True         # time limit gets shorter in later rounds

    def __init__(self, n_players: int, rng: random.Random):
        self.n_players = n_players
        self.rng = rng
        self.start_time: float | None = None
        # Set by the engine: later rounds get less time (see config.TIMER_*).
        self.time_scale = 1.0

    @property
    @abstractmethod
    def prompt(self) -> str:
        """Instruction shown and spoken before the round, e.g. 'Raise your right hand!'."""

    @property
    @abstractmethod
    def time_limit(self) -> float:
        """Seconds the round lasts at most."""

    def begin(self, now: float) -> None:
        self.start_time = now

    def elapsed(self, now: float) -> float:
        return 0.0 if self.start_time is None else now - self.start_time

    @property
    def effective_limit(self) -> float:
        """This round's actual time limit, after the per-round shrink."""
        return self.time_limit * self.time_scale

    def time_left(self, now: float) -> float:
        return max(0.0, self.effective_limit - self.elapsed(now))

    @abstractmethod
    def update(self, poses: list[Pose | None], now: float) -> bool:
        """Feed one frame of poses (one per player). Returns True when the round is over."""

    @abstractmethod
    def result(self) -> RoundResult:
        """Outcome of the finished round."""

    def status(self, now: float) -> list[str]:
        """Short live text per player, shown under their name during the round."""
        return [""] * self.n_players


def speed_bonus(seconds: float, fast: float, slow: float) -> int:
    """Full bonus at or below `fast` seconds, falling linearly to 0 at `slow`."""
    if seconds <= fast:
        return config.SPEED_BONUS_MAX
    if seconds >= slow:
        return 0
    return round(config.SPEED_BONUS_MAX * (slow - seconds) / (slow - fast))
