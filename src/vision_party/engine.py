"""Match engine: runs rounds through a fixed sequence of phases.

    INTRO -> COUNTDOWN -> ACTIVE -> RESULT -> (next round's INTRO | FINISHED)

The engine is a pure state machine: it never reads the camera, draws, or
speaks. Callers pass in poses and the current time, then read its state to
render. Text meant to be spoken is queued in `announcements`; main.py hands
it to the voice. Object games get their detections from main.py too (via
`game.observe`), so the engine stays free of models and I/O.
"""

import random
from enum import Enum, auto
from typing import Callable

from . import config
from .games import BringToCamera, Game, HoldChallenge, PoseBattle, ReactionBattle, RoundResult
from .pose import Pose

GameFactory = Callable[[int, random.Random], Game]

DEFAULT_GAMES: list[GameFactory] = [ReactionBattle, HoldChallenge, PoseBattle, BringToCamera]


class Phase(Enum):
    INTRO = auto()
    COUNTDOWN = auto()
    ACTIVE = auto()
    RESULT = auto()
    FINISHED = auto()


class Match:
    def __init__(self, names: list[str], now: float,
                 rounds: int = config.ROUNDS_PER_MATCH,
                 games: list[GameFactory] | None = None,
                 rng: random.Random | None = None):
        self.names = names
        self.n_players = len(names)
        self.rounds = rounds
        # Solo-only games (Bring to Camera) are left out of Duo matches.
        self.games = [g for g in (games or DEFAULT_GAMES)
                      if self.n_players == 1 or not getattr(g, "solo_only", False)]
        if not self.games:
            raise ValueError("no games available for this number of players")
        self.rng = rng or random.Random()
        self.scores = [0] * self.n_players
        self.streaks = [0] * self.n_players  # rounds won in a row, per player
        self.round_index = -1                 # 0-based; -1 before the first round
        self.game: Game | None = None
        self.last_result: RoundResult | None = None
        self.history: list[RoundResult] = []
        self.announcements: list[str] = []
        self._phase = Phase.INTRO
        self._phase_start = now
        self._last_countdown_said: int | None = None
        self._next_round(now)

    # --- state for the UI ---------------------------------------------------
    @property
    def phase(self) -> Phase:
        return self._phase

    def phase_elapsed(self, now: float) -> float:
        return now - self._phase_start

    def countdown_value(self, now: float) -> int:
        """3, 2, 1 during COUNTDOWN."""
        return max(1, config.COUNTDOWN_SECONDS - int(self.phase_elapsed(now)))

    def show_go(self, now: float) -> bool:
        return self._phase is Phase.ACTIVE and self.phase_elapsed(now) < config.GO_FLASH_SECONDS

    def winners(self) -> list[int]:
        """Match winners by total score (several on a tie)."""
        best = max(self.scores)
        return [i for i, s in enumerate(self.scores) if s == best]

    def leader(self) -> int | None:
        """The player strictly ahead on points (wears the crown), else None."""
        best = max(self.scores)
        if best <= 0:
            return None
        w = self.winners()
        return w[0] if len(w) == 1 else None

    @staticmethod
    def time_scale(round_index: int) -> float:
        """Time-limit multiplier for a round: later rounds are shorter."""
        return max(config.TIMER_MIN_SCALE, 1.0 - config.TIMER_SHRINK_PER_ROUND * round_index)

    def pop_announcements(self) -> list[str]:
        out, self.announcements = self.announcements, []
        return out

    # --- driving the match --------------------------------------------------
    def update(self, poses: list[Pose | None], now: float) -> None:
        t = self.phase_elapsed(now)
        if self._phase is Phase.INTRO and t >= config.INTRO_SECONDS:
            self._enter(Phase.COUNTDOWN, now)
        elif self._phase is Phase.COUNTDOWN:
            value = self.countdown_value(now)
            if value != self._last_countdown_said:
                self._last_countdown_said = value
                self.announcements.append(str(value))
            if t >= config.COUNTDOWN_SECONDS:
                self._enter(Phase.ACTIVE, now)
                self.announcements.append("Go!")
                self.game.begin(now)
                # Evaluate the GO frame too, so arming state starts correctly.
                self._step_game(poses, now)
        elif self._phase is Phase.ACTIVE:
            self._step_game(poses, now)
        elif self._phase is Phase.RESULT and t >= config.RESULT_SECONDS:
            if self.round_index + 1 >= self.rounds:
                self._enter(Phase.FINISHED, now)
                self.announcements.append(self._final_message())
            else:
                self._next_round(now)

    def _step_game(self, poses, now) -> None:
        if self.game.update(poses, now):
            result = self.game.result()
            for i, pts in enumerate(result.points):
                self.scores[i] += pts
            self.last_result = result
            self.history.append(result)
            self._enter(Phase.RESULT, now)
            self.announcements.append(self.named(result.message))
            self._update_streaks(result)

    def _update_streaks(self, result: RoundResult) -> None:
        """Count rounds won in a row and cheer when someone gets hot.
        Streaks are for fun only; they don't change the score."""
        for i in range(self.n_players):
            self.streaks[i] = self.streaks[i] + 1 if i in result.winners else 0
            n = self.streaks[i]
            if n == config.STREAK_ON_FIRE:
                self.announcements.append(f"{self.names[i]} is on fire!")
            elif n > config.STREAK_ON_FIRE:
                self.announcements.append(f"{n} in a row for {self.names[i]}!")

    def _next_round(self, now: float) -> None:
        self.round_index += 1
        factory = self.rng.choice(self.games)
        self.game = factory(self.n_players, self.rng)
        if self.game.shrinks:
            self.game.time_scale = self.time_scale(self.round_index)
        self.last_result = None
        self._last_countdown_said = None
        self._enter(Phase.INTRO, now)
        self.announcements.append(self.game.prompt)

    def _enter(self, phase: Phase, now: float) -> None:
        self._phase = phase
        self._phase_start = now

    def named(self, message: str) -> str:
        """Replace 'P1', 'P2' with player names."""
        for i, name in enumerate(self.names):
            message = message.replace(f"P{i + 1}", name)
        return message

    def _final_message(self) -> str:
        if self.n_players == 1:
            return f"Game over! {self.scores[0]} points."
        w = self.winners()
        if len(w) > 1:
            return "It's a tie!"
        return f"{self.names[w[0]]} wins the match!"
