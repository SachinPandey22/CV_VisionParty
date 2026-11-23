"""Record the target poses for Pose Battle from the webcam.

Run:  python -m vision_party.capture_poses [--camera 0]
Keys: 1-5 = record that pose (3-second countdown, then ~0.5 s of frames are
      averaged), q / Esc = quit.

The live match % against every target is shown on the left, which also makes
this a handy way to check the similarity scores. Recorded poses are written to
src/vision_party/targets.json; commit that file to share them.
"""

import argparse
import time

import cv2
import numpy as np

from .capture import FULL, Camera, mirror
from .pose import PoseEstimator, joint_angles
from .similarity import (TargetPose, load_targets, normalized_points, pose_similarity,
                         save_targets)
from .ui import GREEN, PLAYER_COLORS, YELLOW, center_text, draw_reference, draw_skeleton, text

COUNTDOWN = 3.0
SAMPLE_FRAMES = 15


def average_target(key: str, name: str, poses) -> TargetPose | None:
    """Average joint angles and landmark positions over several frames, so one
    jittery frame doesn't define the target. None if no usable frames."""
    poses = [p for p in poses if p is not None]
    if not poses:
        return None
    angles: dict[str, list[float]] = {}
    for p in poses:
        for joint, value in joint_angles(p).items():
            if value is not None:
                angles.setdefault(joint, []).append(value)
    # Keep joints seen in at least half the frames.
    kept = {j: round(float(np.mean(v)), 1) for j, v in angles.items() if len(v) * 2 >= len(poses)}
    pts = [normalized_points(p) for p in poses]
    mean_pts = {k: [round(float(np.mean([d[k][i] for d in pts])), 4) for i in (0, 1)]
                for k in pts[0]}
    return TargetPose(key, name, kept, mean_pts)


def run(camera_index: int) -> None:
    targets = load_targets()
    estimator = PoseEstimator(FULL)
    slot: int | None = None            # index being recorded
    t_start = 0.0
    samples: list = []
    message, message_until = "", 0.0
    cv2.namedWindow("Capture poses", cv2.WINDOW_NORMAL)
    with Camera(camera_index) as cam:
        while True:
            raw = cam.read()
            now = time.perf_counter()
            pose = estimator.process(raw)
            view = mirror(raw)
            if pose is not None:
                draw_skeleton(view, pose, PLAYER_COLORS[0])

            text(view, "Press 1-5 to record a pose, Q to quit", (20, 40), 0.8, YELLOW)
            for i, t in enumerate(targets):
                score = pose_similarity(pose, t)
                label = f"{i + 1}  {t.name:10s} {'--' if score is None else f'{score:3.0f}%'}"
                text(view, label, (20, 85 + 34 * i), 0.75, GREEN if score and score >= 75 else (255, 255, 255))
            shown = targets[slot] if slot is not None else None
            if shown is not None:
                draw_reference(view, shown)

            h, w = view.shape[:2]
            if slot is not None:
                left = COUNTDOWN - (now - t_start)
                if left > 0:
                    center_text(view, str(int(left) + 1), w // 2, h // 2, 4.0, (255, 255, 255), 8)
                    center_text(view, f"Get into: {targets[slot].name}", w // 2, h // 2 + 80, 1.2, YELLOW, 3)
                else:
                    samples.append(pose)
                    center_text(view, "Hold still...", w // 2, h // 2, 1.6, YELLOW, 4)
                    if len(samples) >= SAMPLE_FRAMES:
                        old = targets[slot]
                        new = average_target(old.key, old.name, samples)
                        if new is None or len(new.angles) < 6:
                            message = "Couldn't see your whole body - step back and try again"
                        else:
                            targets[slot] = new
                            save_targets(targets)
                            message = f"Saved {new.name}!"
                        message_until, slot, samples = now + 2.5, None, []
            if now < message_until:
                center_text(view, message, w // 2, h - 40, 1.0, GREEN, 2)

            cv2.imshow("Capture poses", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if slot is None and ord("1") <= key < ord("1") + len(targets):
                slot, t_start, samples = key - ord("1"), now, []
    estimator.close()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Pose Battle target poses")
    parser.add_argument("--camera", type=int, default=0)
    run(parser.parse_args().camera)


if __name__ == "__main__":
    main()
