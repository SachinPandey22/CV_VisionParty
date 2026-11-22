"""Rule-based body actions.

Each rule is a pure function Pose -> bool. Image y grows downward, so
"above" means a *smaller* y. Distances are scaled by torso length (see config).
A rule returns False when the landmarks it needs aren't visible.
"""

from typing import Callable

import numpy as np

from . import config
from .pose import L, Pose, angle


def _hand_up(pose: Pose, wrist: int) -> bool:
    if not pose.visible(wrist, L.NOSE):
        return False
    margin = config.HAND_UP_MARGIN * pose.torso_length()
    return pose.pt(wrist)[1] < pose.pt(L.NOSE)[1] - margin


def left_hand_up(pose: Pose) -> bool:
    """Only the left hand is up. Exclusive, so raising both hands can't game a
    'left hand' reaction round."""
    return _hand_up(pose, L.LEFT_WRIST) and not _hand_up(pose, L.RIGHT_WRIST)


def right_hand_up(pose: Pose) -> bool:
    return _hand_up(pose, L.RIGHT_WRIST) and not _hand_up(pose, L.LEFT_WRIST)


def both_hands_up(pose: Pose) -> bool:
    return _hand_up(pose, L.LEFT_WRIST) and _hand_up(pose, L.RIGHT_WRIST)


def squat(pose: Pose) -> bool:
    needed = (L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE, L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE)
    if not pose.visible(*needed):
        return False
    lk = angle(pose.pt(L.LEFT_HIP), pose.pt(L.LEFT_KNEE), pose.pt(L.LEFT_ANKLE))
    rk = angle(pose.pt(L.RIGHT_HIP), pose.pt(L.RIGHT_KNEE), pose.pt(L.RIGHT_ANKLE))
    hip_y = (pose.pt(L.LEFT_HIP)[1] + pose.pt(L.RIGHT_HIP)[1]) / 2
    knee_y = (pose.pt(L.LEFT_KNEE)[1] + pose.pt(L.RIGHT_KNEE)[1]) / 2
    hips_dropped = (knee_y - hip_y) < config.SQUAT_HIP_KNEE_GAP * pose.torso_length()
    return (lk + rk) / 2 < config.SQUAT_KNEE_ANGLE and hips_dropped


def one_leg_up(pose: Pose) -> bool:
    if not pose.visible(L.LEFT_ANKLE, L.RIGHT_ANKLE):
        return False
    gap = abs(pose.pt(L.LEFT_ANKLE)[1] - pose.pt(L.RIGHT_ANKLE)[1])
    return gap > config.ONE_LEG_ANKLE_GAP * pose.torso_length()


def t_pose(pose: Pose) -> bool:
    needed = (L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST,
              L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST)
    if not pose.visible(*needed):
        return False
    torso = pose.torso_length()
    for sh, el, wr in ((L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST),
                       (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST)):
        if angle(pose.pt(sh), pose.pt(el), pose.pt(wr)) < config.T_POSE_ELBOW_ANGLE:
            return False
        if abs(pose.pt(wr)[1] - pose.pt(sh)[1]) > config.T_POSE_WRIST_HEIGHT_TOL * torso:
            return False
    spread = np.linalg.norm(pose.pt(L.LEFT_WRIST) - pose.pt(L.RIGHT_WRIST))
    return spread > config.T_POSE_SPREAD * torso


ACTIONS: dict[str, Callable[[Pose], bool]] = {
    "LEFT_HAND_UP": left_hand_up,
    "RIGHT_HAND_UP": right_hand_up,
    "BOTH_HANDS_UP": both_hands_up,
    "SQUAT": squat,
    "ONE_LEG_UP": one_leg_up,
    "T_POSE": t_pose,
}

# Spoken / on-screen wording for each action (used by the games later).
PROMPTS = {
    "LEFT_HAND_UP": "Raise your left hand!",
    "RIGHT_HAND_UP": "Raise your right hand!",
    "BOTH_HANDS_UP": "Both hands up!",
    "SQUAT": "Squat!",
    "ONE_LEG_UP": "Stand on one leg!",
    "T_POSE": "Make a T-pose!",
}


def detect_actions(pose: Pose | None) -> dict[str, bool]:
    """Evaluate every action. With no pose, everything is False."""
    if pose is None:
        return {name: False for name in ACTIONS}
    return {name: rule(pose) for name, rule in ACTIONS.items()}
