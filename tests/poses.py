"""Synthetic skeletons for tests (raw-frame pixels, person facing camera)."""

import numpy as np

from vision_party.pose import L, Pose


def make_pose(overrides=None) -> Pose:
    """A neutral standing pose (arms down, legs straight), torso length 200px."""
    p = {
        L.NOSE: (500, 150),
        L.LEFT_SHOULDER: (560, 250), L.RIGHT_SHOULDER: (440, 250),
        L.LEFT_ELBOW: (570, 350), L.RIGHT_ELBOW: (430, 350),
        L.LEFT_WRIST: (575, 440), L.RIGHT_WRIST: (425, 440),
        L.LEFT_HIP: (540, 450), L.RIGHT_HIP: (460, 450),
        L.LEFT_KNEE: (540, 600), L.RIGHT_KNEE: (460, 600),
        L.LEFT_ANKLE: (540, 750), L.RIGHT_ANKLE: (460, 750),
    }
    p.update(overrides or {})
    points = np.zeros((33, 2), dtype=np.float32)
    vis = np.zeros(33, dtype=np.float32)
    for idx, xy in p.items():
        points[idx] = xy
        vis[idx] = 0.99
    return Pose(points, vis)


RIGHT_HAND = {L.RIGHT_ELBOW: (430, 160), L.RIGHT_WRIST: (425, 60)}
LEFT_HAND = {L.LEFT_ELBOW: (570, 160), L.LEFT_WRIST: (575, 60)}
