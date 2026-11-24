"""Reaction Battle: do the called action as fast as possible.

Solo: succeed before the timer runs out; faster = more bonus points.
Duo:  first player to do it wins the round.

Anti-cheat: a player who is *already* doing the action at "GO" must drop it
first ("arming"), so you can't win by holding the pose through the countdown.
"""

import random

from .. import config
from ..actions import ACTIONS, PROMPTS
from ..pose import Pose
from .base import Game, RoundResult, speed_bonus


class _PlayerState:
    def __init__(self):
        self.armed = False             # has been out of the pose since GO
        self.streak = 0                # consecutive frames doing the action
        self.streak_start: float | None = None
        self.reaction: float | None = None  # seconds from GO, once confirmed


class ReactionBattle(Game):
    title = "Reaction Battle"

    def __init__(self, n_players: int, rng: random.Random, action: str | None = None,
                 timeout: float = config.REACTION_TIMEOUT):
        super().__init__(n_players, rng)
        self.action = action or rng.choice(sorted(ACTIONS))
        self._rule = ACTIONS[self.action]
        self._timeout = timeout
        self.players = [_PlayerState() for _ in range(n_players)]

    @property
    def prompt(self) -> str:
        return PROMPTS[self.action]

    @property
    def time_limit(self) -> float:
        return self._timeout

    def update(self, poses: list[Pose | None], now: float) -> bool:
        for pose, st in zip(poses, self.players):
            if st.reaction is not None:
                continue
            doing = pose is not None and self._rule(pose)
            if not doing:
                st.armed = True
                st.streak, st.streak_start = 0, None
                continue
            if not st.armed:
                continue
            if st.streak == 0:
                st.streak_start = now
            st.streak += 1
            if st.streak >= config.REACTION_CONFIRM_FRAMES:
                st.reaction = st.streak_start - self.start_time

        reacted = [st.reaction is not None for st in self.players]
        if self.n_players == 1:
            done = reacted[0]
        else:
            done = any(reacted)  # first confirmed reaction ends a Duo round
        return done or self.time_left(now) <= 0

    def result(self) -> RoundResult:
        times = [st.reaction for st in self.players]
        points = [0] * self.n_players
        details = [f"{t:.2f}s" if t is not None else "no reaction" for t in times]

        valid = [t for t in times if t is not None]
        if not valid:
            msg = "Too slow!" if self.n_players == 1 else "Nobody made it!"
            return RoundResult([], points, msg, details, times)

        best = min(valid)
        # Players confirmed in the same frame share the win.
        winners = [i for i, t in enumerate(times) if t is not None and t == best]
        for i in winners:
            points[i] = config.BASE_POINTS + speed_bonus(best, config.REACTION_FAST, self.effective_limit)
        if self.n_players == 1:
            msg = f"Nice! {best:.2f}s"
        elif len(winners) == 1:
            msg = f"P{winners[0] + 1} wins! {best:.2f}s"
        else:
            msg = f"Tie! {best:.2f}s"
        return RoundResult(winners, points, msg, details, times)

    def status(self, now: float) -> list[str]:
        out = []
        for st in self.players:
            if st.reaction is not None:
                out.append(f"{st.reaction:.2f}s")
            elif not st.armed:
                out.append("drop the pose first")
            else:
                out.append("")
        return out
