"""Pure helpers for the playful extras (no drawing, no camera).

Kept separate from the drawing code so the rules can be unit-tested.
"""

from collections import deque

import numpy as np

from . import config

# Ratings, best first. Level 0 = best; the UI picks a color per level.
RATINGS = ["PERFECT!", "GREAT!", "GOOD", "OK"]


def rating(seconds: float | None) -> tuple[str, int] | None:
    """Label and level for a reaction time, or None if the player didn't react."""
    if seconds is None:
        return None
    if seconds <= config.RATING_PERFECT:
        return RATINGS[0], 0
    if seconds <= config.RATING_GREAT:
        return RATINGS[1], 1
    if seconds <= config.RATING_GOOD:
        return RATINGS[2], 2
    return RATINGS[3], 3


def streak_label(n: int) -> str:
    """Short badge text for a win streak ('' when there is nothing to show)."""
    if n >= config.STREAK_ON_FIRE:
        return f"ON FIRE x{n}"
    if n >= 2:
        return f"{n} in a row"
    return ""


def display_region(player: int, n_players: int, width: int) -> tuple[int, int]:
    """x-range of a player's side of the *mirrored* screen.

    Duo: Player 1 is on the left of the screen (see capture.player_regions).
    """
    if n_players == 1:
        return 0, width
    half = width // 2
    return (0, half) if player == 0 else (half, width)


class Moments:
    """Keeps each player's fastest reaction as a photo for the winner screen.

    The app pushes every displayed frame during a round. When a round ends,
    the frame closest to GO + reaction time is cut to the player's side of the
    screen and kept if it beats that player's previous best.
    """

    def __init__(self, n_players: int, buffer: int = config.MOMENT_BUFFER_FRAMES):
        self.n_players = n_players
        self._frames: deque[tuple[float, np.ndarray]] = deque(maxlen=buffer)
        self.best: list[tuple[float, np.ndarray] | None] = [None] * n_players

    def push(self, t: float, frame: np.ndarray) -> None:
        self._frames.append((t, frame))

    def clear_buffer(self) -> None:
        self._frames.clear()

    def record(self, player: int, reaction: float, at_time: float) -> bool:
        """Keep the frame nearest `at_time` if `reaction` is this player's best.
        Returns True when a new best photo was stored."""
        frame = pick_frame(self._frames, at_time)
        if frame is None:
            return False
        prev = self.best[player]
        if prev is not None and prev[0] <= reaction:
            return False
        x0, x1 = display_region(player, self.n_players, frame.shape[1])
        self.best[player] = (reaction, frame[:, x0:x1].copy())
        return True


def pick_frame(frames, t: float) -> np.ndarray | None:
    """Frame whose timestamp is closest to t (None if there are no frames)."""
    if not frames:
        return None
    return min(frames, key=lambda item: abs(item[0] - t))[1]
