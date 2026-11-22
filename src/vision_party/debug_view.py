"""Day 1 debug view: live skeleton, joint angles and detected actions.

Run:  python -m vision_party.debug_view [--duo] [--camera 0]
Keys: d = toggle Solo/Duo, a = toggle angle labels, q / Esc = quit
"""

import argparse
import time

import cv2

from .actions import detect_actions
from .capture import Camera, mirror, player_regions
from .pose import PoseEstimator
from .ui import PLAYER_COLORS, draw_skeleton

ON, OFF = (80, 220, 80), (150, 150, 150)


def draw_panel(img, x: int, label: str, color, actions: dict[str, bool]) -> None:
    cv2.putText(img, label, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    for i, (name, on) in enumerate(actions.items()):
        cv2.putText(img, ("* " if on else "  ") + name, (x, 75 + 28 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, ON if on else OFF, 2, cv2.LINE_AA)


def run(duo: bool, camera_index: int) -> None:
    show_angles = True
    estimators = [PoseEstimator(r) for r in player_regions(duo)]
    last = time.perf_counter()
    fps = 0.0
    with Camera(camera_index) as cam:
        while True:
            raw = cam.read()
            poses = [e.process(raw) for e in estimators]
            view = mirror(raw)
            h, w = view.shape[:2]

            if duo:
                cv2.line(view, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
            for i, pose in enumerate(poses):
                color = PLAYER_COLORS[i]
                if pose is not None:
                    draw_skeleton(view, pose, color, show_angles)
                panel_x = 20 + i * (w // 2)
                label = f"P{i + 1}" + ("" if pose is not None else " (not visible)")
                draw_panel(view, panel_x, label, color, detect_actions(pose))

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            cv2.putText(view, f"{fps:.1f} FPS  [{'DUO' if duo else 'SOLO'}]  d=mode a=angles q=quit",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Vision Party - debug", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                show_angles = not show_angles
            if key == ord("d"):
                duo = not duo
                for e in estimators:
                    e.close()
                estimators = [PoseEstimator(r) for r in player_regions(duo)]

    for e in estimators:
        e.close()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--duo", action="store_true", help="two players, split screen")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    args = parser.parse_args()
    run(args.duo, args.camera)


if __name__ == "__main__":
    main()
