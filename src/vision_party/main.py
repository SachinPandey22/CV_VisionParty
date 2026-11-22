"""Vision Party game entry point.

Run:  python -m vision_party [--p1 NAME] [--p2 NAME] [--rounds N] [--camera 0]
Menu: 1 = Solo, 2 = Duo, Q / Esc = quit
After a match: R = play again, M = menu
"""

import argparse
import time

import cv2

from . import config
from .capture import Camera, mirror, player_regions
from .engine import Match, Phase
from .pose import PoseEstimator
from . import ui

WINDOW = "Vision Party"


class App:
    def __init__(self, args):
        self.args = args
        self.match: Match | None = None
        self.estimators: list[PoseEstimator] = []
        self.duo = False

    def start_match(self, duo: bool) -> None:
        self.close_estimators()
        self.duo = duo
        self.estimators = [PoseEstimator(r) for r in player_regions(duo)]
        names = [self.args.p1, self.args.p2] if duo else [self.args.p1]
        self.match = Match(names, time.perf_counter(), rounds=self.args.rounds)

    def to_menu(self) -> None:
        self.close_estimators()
        self.match = None

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
        ui.draw_match(view, self.match, poses, now)

        if self.match.phase is Phase.FINISHED:
            if key == ord("r"):
                self.start_match(self.duo)
            elif key == ord("m"):
                self.to_menu()
        elif key == ord("m"):
            self.to_menu()
        return view, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision Party")
    parser.add_argument("--p1", default="P1", help="Player 1 name")
    parser.add_argument("--p2", default="P2", help="Player 2 name (Duo)")
    parser.add_argument("--rounds", type=int, default=config.ROUNDS_PER_MATCH)
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
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
