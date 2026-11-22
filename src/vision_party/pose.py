"""Pose estimation (MediaPipe) and joint-angle geometry.

Landmarks are stored in *pixel* coordinates of the full raw frame. Pixels
(not MediaPipe's 0-1 normalized values) are used because normalized x and y
are scaled by different amounts (width vs. height), which would distort angles.
"""

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from . import config
from .capture import Region

_mp_pose = mp.solutions.pose
L = _mp_pose.PoseLandmark          # e.g. L.LEFT_WRIST == 15
CONNECTIONS = _mp_pose.POSE_CONNECTIONS


@dataclass
class Pose:
    points: np.ndarray      # (33, 2) x, y in raw-frame pixels
    visibility: np.ndarray  # (33,) 0..1

    def visible(self, *ids: int) -> bool:
        return all(self.visibility[i] >= config.MIN_VISIBILITY for i in ids)

    def pt(self, idx: int) -> np.ndarray:
        return self.points[idx]

    def torso_length(self) -> float:
        """Shoulder midpoint to hip midpoint, in pixels. Used as the body scale."""
        sh = (self.pt(L.LEFT_SHOULDER) + self.pt(L.RIGHT_SHOULDER)) / 2
        hp = (self.pt(L.LEFT_HIP) + self.pt(L.RIGHT_HIP)) / 2
        return float(np.linalg.norm(sh - hp)) or 1.0


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC at vertex B, in degrees (0-180)."""
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# Joint name -> (point A, vertex B, point C). Used by actions and pose similarity.
JOINTS = {
    "left_elbow":     (L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST),
    "right_elbow":    (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST),
    "left_shoulder":  (L.LEFT_HIP, L.LEFT_SHOULDER, L.LEFT_ELBOW),
    "right_shoulder": (L.RIGHT_HIP, L.RIGHT_SHOULDER, L.RIGHT_ELBOW),
    "left_hip":       (L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_KNEE),
    "right_hip":      (L.RIGHT_SHOULDER, L.RIGHT_HIP, L.RIGHT_KNEE),
    "left_knee":      (L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE),
    "right_knee":     (L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE),
}


def joint_angles(pose: Pose) -> dict[str, float | None]:
    """All joint angles in degrees; None where a landmark isn't visible."""
    out: dict[str, float | None] = {}
    for name, (a, b, c) in JOINTS.items():
        out[name] = angle(pose.pt(a), pose.pt(b), pose.pt(c)) if pose.visible(a, b, c) else None
    return out


class PoseEstimator:
    """Wraps one MediaPipe Pose tracker for one player region.

    Each player gets their own estimator because MediaPipe keeps tracking
    state between frames and only tracks a single person per image.
    """

    def __init__(self, region: Region):
        self.region = region
        self._pose = _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.MODEL_COMPLEXITY,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def process(self, raw_frame: np.ndarray) -> Pose | None:
        crop, x_offset = self.region.crop(raw_frame)
        h, w = crop.shape[:2]
        result = self._pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks is None:
            return None
        lm = result.pose_landmarks.landmark
        points = np.array([[p.x * w + x_offset, p.y * h] for p in lm], dtype=np.float32)
        vis = np.array([p.visibility for p in lm], dtype=np.float32)
        return Pose(points, vis)

    def close(self) -> None:
        self._pose.close()
