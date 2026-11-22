"""Action rules on hand-built skeletons (no camera needed).

Coordinates are raw-frame pixels for a person facing the camera, so the
person's LEFT side is on the image's RIGHT (larger x). y grows downward.
"""

import numpy as np
import pytest

from vision_party.actions import detect_actions
from vision_party.capture import Region, player_regions
from vision_party.pose import L, Pose, angle


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


def active(pose):
    return {k for k, v in detect_actions(pose).items() if v}


def test_standing_triggers_nothing():
    assert active(make_pose()) == set()


def test_no_pose_is_all_false():
    assert not any(detect_actions(None).values())


def test_left_hand_up_is_exclusive():
    assert active(make_pose({L.LEFT_ELBOW: (570, 160), L.LEFT_WRIST: (575, 60)})) == {"LEFT_HAND_UP"}


def test_right_hand_up():
    assert active(make_pose({L.RIGHT_ELBOW: (430, 160), L.RIGHT_WRIST: (425, 60)})) == {"RIGHT_HAND_UP"}


def test_both_hands_up_not_single_hand():
    pose = make_pose({L.LEFT_ELBOW: (570, 160), L.LEFT_WRIST: (575, 60),
                        L.RIGHT_ELBOW: (430, 160), L.RIGHT_WRIST: (425, 60)})
    assert active(pose) == {"BOTH_HANDS_UP"}


def test_squat():
    # Hips drop to just above the knees, knees bend forward.
    pose = make_pose({
        L.NOSE: (500, 300),
        L.LEFT_SHOULDER: (560, 400), L.RIGHT_SHOULDER: (440, 400),
        L.LEFT_ELBOW: (570, 480), L.RIGHT_ELBOW: (430, 480),
        L.LEFT_WRIST: (575, 550), L.RIGHT_WRIST: (425, 550),
        L.LEFT_HIP: (540, 600), L.RIGHT_HIP: (460, 600),
        L.LEFT_KNEE: (600, 640), L.RIGHT_KNEE: (400, 640),
        L.LEFT_ANKLE: (560, 750), L.RIGHT_ANKLE: (440, 750),
    })
    assert "SQUAT" in active(pose)


def test_one_leg_up():
    pose = make_pose({L.LEFT_KNEE: (560, 540), L.LEFT_ANKLE: (550, 620)})
    assert active(pose) == {"ONE_LEG_UP"}


def test_t_pose():
    pose = make_pose({L.LEFT_ELBOW: (680, 255), L.LEFT_WRIST: (800, 260),
                        L.RIGHT_ELBOW: (320, 255), L.RIGHT_WRIST: (200, 260)})
    assert active(pose) == {"T_POSE"}


def test_hidden_landmarks_return_false():
    pose = make_pose({L.LEFT_ELBOW: (570, 160), L.LEFT_WRIST: (575, 60)})
    pose.visibility[L.LEFT_WRIST] = 0.1
    assert not detect_actions(pose)["LEFT_HAND_UP"]


def test_rules_are_scale_invariant():
    big = make_pose({L.LEFT_ELBOW: (680, 255), L.LEFT_WRIST: (800, 260),
                       L.RIGHT_ELBOW: (320, 255), L.RIGHT_WRIST: (200, 260)})
    small = Pose(big.points * 0.4, big.visibility)
    assert detect_actions(big) == detect_actions(small)


def test_angle():
    assert angle(np.array([1, 0]), np.array([0, 0]), np.array([0, 1])) == pytest.approx(90)
    assert angle(np.array([1, 0]), np.array([0, 0]), np.array([-1, 0])) == pytest.approx(180)


def test_duo_regions_map_screen_left_to_raw_right():
    p1, p2 = player_regions(duo=True)
    assert p1 == Region(0.5, 1.0) and p2 == Region(0.0, 0.5)
    frame = np.zeros((10, 100, 3), dtype=np.uint8)
    crop, offset = p1.crop(frame)
    assert crop.shape[1] == 50 and offset == 50
