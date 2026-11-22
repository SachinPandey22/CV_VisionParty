"""Game modes. Each one implements `Game` from `base` so the engine can run any of them."""

from .base import Game, RoundResult
from .reaction import ReactionBattle

__all__ = ["Game", "RoundResult", "ReactionBattle"]
