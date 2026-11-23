"""Joint-angle pose similarity and the target poses for Pose Battle.

Comparing joint *angles* (not landmark positions) makes the score independent
of where the player stands, how far they are from the camera, and how tall
they are. Each of the 8 joints (elbows, shoulders, hips, knees) scores
100% when it matches the target angle and falls linearly to 0% at
SIM_MAX_DIFF degrees off; the pose score is the average over the joints
visible in both poses.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .pose import JOINTS, L, Pose, joint_angles

TARGETS_FILE = Path(__file__).with_name("targets.json")

# Landmarks kept for drawing the reference stick figure.
BODY = [L.NOSE, L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_ELBOW, L.RIGHT_ELBOW,
        L.LEFT_WRIST, L.RIGHT_WRIST, L.LEFT_HIP, L.RIGHT_HIP,
        L.LEFT_KNEE, L.RIGHT_KNEE, L.LEFT_ANKLE, L.RIGHT_ANKLE]


@dataclass
class TargetPose:
    key: str
    name: str                         # shown to players, e.g. "Star"
    angles: dict[str, float]          # joint name -> degrees
    points: dict[int, list[float]]    # landmark id -> [x, y], body-normalized

    @property
    def prompt(self) -> str:
        return f"Strike the pose: {self.name}!"


def similarity(player: dict[str, float | None], target: dict[str, float]) -> float | None:
    """Score 0-100 of how well `player` angles match `target` angles.
    None if fewer than SIM_MIN_JOINTS joints can be compared."""
    scores = []
    for joint, want in target.items():
        got = player.get(joint)
        if got is None or want is None:
            continue
        diff = abs(got - want)
        scores.append(max(0.0, 1.0 - diff / config.SIM_MAX_DIFF))
    if len(scores) < config.SIM_MIN_JOINTS:
        return None
    return 100.0 * float(np.mean(scores))


def pose_similarity(pose: Pose | None, target: TargetPose) -> float | None:
    if pose is None:
        return None
    return similarity(joint_angles(pose), target.angles)


def normalized_points(pose: Pose) -> dict[int, list[float]]:
    """Body landmarks relative to the hip midpoint, in torso lengths."""
    hip = (pose.pt(L.LEFT_HIP) + pose.pt(L.RIGHT_HIP)) / 2
    torso = pose.torso_length()
    return {int(i): [round(float(v), 4) for v in (pose.pt(i) - hip) / torso] for i in BODY}


def target_from_pose(key: str, name: str, pose: Pose) -> TargetPose:
    angles = {k: round(v, 1) for k, v in joint_angles(pose).items() if v is not None}
    return TargetPose(key, name, angles, normalized_points(pose))


def load_targets(path: Path = TARGETS_FILE) -> list[TargetPose]:
    data = json.loads(Path(path).read_text())
    return [TargetPose(t["key"], t["name"], t["angles"],
                       {int(k): v for k, v in t["points"].items()})
            for t in data["poses"]]


def save_targets(targets: list[TargetPose], path: Path = TARGETS_FILE) -> None:
    data = {"joints": list(JOINTS), "poses": [
        {"key": t.key, "name": t.name, "angles": t.angles,
         "points": {str(k): v for k, v in t.points.items()}} for t in targets]}
    Path(path).write_text(json.dumps(data, indent=2) + "\n")
