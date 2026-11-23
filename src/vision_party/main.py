"""Vision Party game entry point.

Run:  python -m vision_party [--p1 NAME] [--p2 NAME] [--rounds N] [--camera 0]
                             [--save-moments]
Menu: 1 = Solo, 2 = Duo, Q / Esc = quit
After a match: R = play again, M = menu
"""

import argparse
import time
from pathlib import Path

import cv2

from . import config
from .capture import Camera, mirror, player_regions
from .effects import Confetti, draw_props
from .engine import Match, Phase
from .fun import Moments, display_region
from .pose import PoseEstimator
from . import ui

WINDOW = "Vision Party"


def crown_for(match: Match) -> int | None:
    """Who wears the crown. Duo: whoever is ahead on points. Solo: you earn it
    by getting on fire, and you keep it on the game-over screen if you scored."""
    if match.n_players == 2:
        return match.leader()
    if match.streaks[0] >= config.STREAK_ON_FIRE:
        return 0
    if match.phase is Phase.FINISHED and match.scores[0] > 0:
        return 0
    return None


class App:
    def __init__(self, args):
        self.args = args
        self.match: Match | None = None
        self.estimators: list[PoseEstimator] = []
        self.duo = False
        self.confetti = Confetti()
        self.moments: Moments | None = None
        self._last_phase: Phase | None = None
        self._last_now: float | None = None

    def start_match(self, duo: bool) -> None:
        self.close_estimators()
        self.duo = duo
        self.estimators = [PoseEstimator(r) for r in player_regions(duo)]
        names = [self.args.p1, self.args.p2] if duo else [self.args.p1]
        self.match = Match(names, time.perf_counter(), rounds=self.args.rounds)
        self.moments = Moments(len(names))
        self.confetti.clear()
        self._last_phase = None

    def to_menu(self) -> None:
        self.close_estimators()
        self.match = None
        self.confetti.clear()

    def close_estimators(self) -> None:
        for e in self.estimators:
            e.close()
        self.estimators = []

    def frame(self, raw, key: int) -> tuple[object, bool]:
        """Process one frame. Returns (image to show, keep running)."""
        if key in (ord("q"), 27):
            return None, False
        view = mirror(raw)

        if self.match is None:
            ui.draw_menu(view)
            if key == ord("1"):
                self.start_match(duo=False)
            elif key == ord("2"):
                self.start_match(duo=True)
            return view, True

        now = time.perf_counter()
        poses = [e.process(raw) for e in self.estimators]
        self.match.update(poses, now)
        for text in self.match.pop_announcements():
            print(f"[voice] {text}")      # replaced by text-to-speech on Day 4

        if self.duo:
            ui.draw_divider(view)
        for i, pose in enumerate(poses):
            if pose is not None:
                ui.draw_skeleton(view, pose, ui.PLAYER_COLORS[i])
        draw_props(view, poses, ui.PLAYER_COLORS, crown_for(self.match))
        if self.match.phase is Phase.ACTIVE:
            # Photo buffer: skeleton and hats in, but no text overlays.
            self.moments.push(now, view.copy())
        self._on_phase_change(view.shape[1], view.shape[0])

        ui.draw_match(view, self.match, poses, now, self.moments)
        self._draw_confetti(view, now)

        if self.match.phase is Phase.FINISHED:
            if key == ord("r"):
                self.start_match(self.duo)
            elif key == ord("m"):
                self.to_menu()
        elif key == ord("m"):
            self.to_menu()
        return view, True

    def _on_phase_change(self, width: int, height: int) -> None:
        """Effects that fire once when a round ends or the match finishes."""
        phase = self.match.phase
        if phase is self._last_phase:
            return
        self._last_phase = phase
        if phase is Phase.RESULT:
            res, game = self.match.last_result, self.match.game
            for i in res.winners:
                x0, x1 = display_region(i, self.match.n_players, width)
                self.confetti.burst((x0 + x1) / 2, height * 0.45)
                if i < len(res.times) and res.times[i] is not None:
                    self.moments.record(i, res.times[i], game.start_time + res.times[i])
            self.moments.clear_buffer()
        elif phase is Phase.FINISHED:
            for x in (0.25, 0.5, 0.75):
                self.confetti.burst(width * x, height * 0.6)
            if self.args.save_moments:
                self._save_moments()

    def _draw_confetti(self, view, now: float) -> None:
        dt = 0.0 if self._last_now is None else now - self._last_now
        self._last_now = now
        if self.match.phase is Phase.FINISHED:
            self.confetti.rain(view.shape[1], 2)
        self.confetti.update(dt, view.shape[0])
        self.confetti.draw(view)

    def _save_moments(self) -> None:
        out = Path("moments")
        out.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for i, best in enumerate(self.moments.best):
            if best is not None:
                path = out / f"{stamp}_{self.match.names[i]}_{best[0]:.2f}s.jpg"
                cv2.imwrite(str(path), best[1])
                print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision Party")
    parser.add_argument("--p1", default="P1", help="Player 1 name")
    parser.add_argument("--p2", default="P2", help="Player 2 name (Duo)")
    parser.add_argument("--rounds", type=int, default=config.ROUNDS_PER_MATCH)
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--save-moments", action="store_true",
                        help="save each player's fastest-reaction photo to moments/")
    args = parser.parse_args()

    app = App(args)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    with Camera(args.camera) as cam:
        key = -1
        while True:
            view, running = app.frame(cam.read(), key)
            if not running:
                break
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
    app.close_estimators()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
