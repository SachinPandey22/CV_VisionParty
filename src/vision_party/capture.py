"""Webcam capture and player regions.

Pose estimation always runs on the *raw* (un-mirrored) camera frame so that
MediaPipe's LEFT/RIGHT labels match the player's real left and right hand.
The frame is mirrored only for display, so players see themselves like in a
mirror.

Because of the mirroring, the player standing on the LEFT of the screen is in
the RIGHT half of the raw frame. `player_regions` handles that mapping.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class Region:
    """A vertical slice of the raw frame, as fractions of its width."""
    x0: float
    x1: float

    def crop(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        """Return the slice of `frame` and its left offset in pixels."""
        w = frame.shape[1]
        left, right = int(self.x0 * w), int(self.x1 * w)
        return frame[:, left:right], left


FULL = Region(0.0, 1.0)


def player_regions(duo: bool) -> list[Region]:
    """Regions of the raw frame for each player, in player order.

    Duo: Player 1 is on the left of the (mirrored) screen, which is the right
    half of the raw frame.
    """
    if not duo:
        return [FULL]
    return [Region(0.5, 1.0), Region(0.0, 0.5)]


class Camera:
    def __init__(self, index: int = config.CAMERA_INDEX,
                 width: int = config.FRAME_WIDTH, height: int = config.FRAME_HEIGHT):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {index}. On macOS, allow camera access "
                "for your terminal app in System Settings > Privacy & Security > Camera.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray:
        """Return the next raw (un-mirrored) BGR frame."""
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera stopped returning frames.")
        return frame

    def release(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def mirror(frame: np.ndarray) -> np.ndarray:
    """Flip horizontally for display."""
    return cv2.flip(frame, 1)
