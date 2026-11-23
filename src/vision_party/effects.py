"""Party props (hats, crown) and confetti, drawn on the display frame.

Props are anchored to the face landmarks so they follow the player's head,
including when it tilts. Everything works in *display* (mirrored) pixels.
"""

import cv2
import numpy as np

from . import config
from .pose import L, Pose

GOLD = (40, 200, 255)          # BGR
GOLD_DARK = (20, 120, 190)
WHITE = (255, 255, 255)
CONFETTI_COLORS = np.array([
    (80, 80, 255), (60, 220, 255), (90, 220, 90), (255, 170, 60),
    (230, 90, 230), (255, 255, 255), (40, 200, 255),
], dtype=np.uint8)


# --- head geometry ----------------------------------------------------------
def head_frame(pose: Pose, width: int):
    """Where to put a hat: (eye midpoint, unit 'right', unit 'up', head width).

    Head width comes from the ears when both are visible, otherwise from the
    eyes (head width is roughly 2.5x the distance between the eyes). 'Up' is
    perpendicular to the eye line, pointing away from the nose, so the hat
    tilts with the head. Returns None if the eyes aren't visible.
    """
    if not pose.visible(L.LEFT_EYE, L.RIGHT_EYE, L.NOSE):
        return None
    pts = pose.points.astype(np.float32).copy()
    pts[:, 0] = width - 1 - pts[:, 0]             # mirror to display coords
    le, re, nose = pts[L.LEFT_EYE], pts[L.RIGHT_EYE], pts[L.NOSE]
    eye_mid = (le + re) / 2
    eye_vec = le - re
    eye_dist = float(np.linalg.norm(eye_vec))
    if eye_dist < 1:
        return None
    right = eye_vec / eye_dist
    if right[0] < 0:                              # keep 'right' pointing screen-right
        right = -right
    up = np.array([right[1], -right[0]], dtype=np.float32)
    if np.dot(up, eye_mid - nose) < 0:            # point away from the nose
        up = -up
    if pose.visible(L.LEFT_EAR, L.RIGHT_EAR):
        head_w = float(np.linalg.norm(pts[L.LEFT_EAR] - pts[L.RIGHT_EAR]))
    else:
        head_w = 2.5 * eye_dist
    head_w = max(head_w, 2.0 * eye_dist)          # ears can be hidden by hair
    return eye_mid, right, up, head_w


def _shape(origin, right, up, w, h, local) -> np.ndarray:
    """Map local (x across, y up) coords in [-.5,.5] x [0,1] to pixels."""
    pts = [origin + right * (x * w) + up * (y * h) for x, y in local]
    return np.array(pts, dtype=np.int32)


def draw_party_hat(img, pose: Pose, color) -> None:
    frame = head_frame(pose, img.shape[1])
    if frame is None:
        return
    eye_mid, right, up, hw = frame
    base = eye_mid + up * (0.62 * hw)             # roughly the top of the forehead
    w, h = 0.85 * hw, 1.15 * hw
    cone = _shape(base, right, up, w, h, [(-0.5, 0), (0.5, 0), (0, 1)])
    cv2.fillPoly(img, [cone], color, cv2.LINE_AA)
    # Two white stripes across the cone.
    for y in (0.3, 0.6):
        half = 0.5 * (1 - y)
        a, b = _shape(base, right, up, w, h, [(-half, y), (half, y)])
        cv2.line(img, tuple(a), tuple(b), WHITE, max(2, int(hw * 0.06)), cv2.LINE_AA)
    cv2.polylines(img, [cone], True, (40, 40, 40), 2, cv2.LINE_AA)
    tip = tuple(_shape(base, right, up, w, h, [(0, 1)])[0])
    cv2.circle(img, tip, max(4, int(hw * 0.11)), WHITE, -1, cv2.LINE_AA)


def draw_crown(img, pose: Pose) -> None:
    frame = head_frame(pose, img.shape[1])
    if frame is None:
        return
    eye_mid, right, up, hw = frame
    base = eye_mid + up * (0.62 * hw)
    w, h = 0.95 * hw, 0.6 * hw
    local = [(-0.5, 0), (0.5, 0), (0.5, 1), (0.25, 0.5), (0, 1.15), (-0.25, 0.5), (-0.5, 1)]
    crown = _shape(base, right, up, w, h, local)
    cv2.fillPoly(img, [crown], GOLD, cv2.LINE_AA)
    cv2.polylines(img, [crown], True, GOLD_DARK, 2, cv2.LINE_AA)
    r = max(3, int(hw * 0.07))
    for p in _shape(base, right, up, w, h, [(0.5, 1), (0, 1.15), (-0.5, 1)]):
        cv2.circle(img, tuple(p), r, (80, 80, 255), -1, cv2.LINE_AA)   # red jewels
    band = _shape(base, right, up, w, h, [(-0.3, 0.22), (0, 0.22), (0.3, 0.22)])
    for p in band:
        cv2.circle(img, tuple(p), r, (230, 120, 60), -1, cv2.LINE_AA)  # blue jewels


def draw_props(img, poses: list[Pose | None], colors, crown_for: int | None) -> None:
    """Party hat on every visible player; the leader gets a crown instead."""
    for i, pose in enumerate(poses):
        if pose is None:
            continue
        if i == crown_for:
            draw_crown(img, pose)
        else:
            draw_party_hat(img, pose, colors[i])


# --- confetti ---------------------------------------------------------------
class Confetti:
    """Simple particle system: bursts that fly up and fall with gravity."""

    GRAVITY = 900.0            # px / s^2
    DRAG = 0.6                 # fraction of velocity lost per second

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        # columns: x, y, vx, vy, size, color index, spin phase
        self.p = np.zeros((0, 7), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.p)

    def burst(self, x: float, y: float, n: int = config.CONFETTI_BURST) -> None:
        r = self.rng
        angle = r.uniform(-np.pi * 0.9, -np.pi * 0.1, n)    # mostly upward
        speed = r.uniform(300, 950, n)
        new = np.column_stack([
            np.full(n, x) + r.normal(0, 15, n),
            np.full(n, y) + r.normal(0, 15, n),
            np.cos(angle) * speed,
            np.sin(angle) * speed,
            r.uniform(5, 11, n),
            r.integers(0, len(CONFETTI_COLORS), n),
            r.uniform(0, 2 * np.pi, n),
        ]).astype(np.float32)
        self._add(new)

    def rain(self, width: int, n: int) -> None:
        """Confetti falling from the top edge (used on the winner screen)."""
        r = self.rng
        new = np.column_stack([
            r.uniform(0, width, n), r.uniform(-40, 0, n),
            r.normal(0, 60, n), r.uniform(50, 200, n),
            r.uniform(5, 11, n), r.integers(0, len(CONFETTI_COLORS), n),
            r.uniform(0, 2 * np.pi, n),
        ]).astype(np.float32)
        self._add(new)

    def _add(self, new: np.ndarray) -> None:
        self.p = np.vstack([self.p, new])[-config.CONFETTI_MAX:]

    def update(self, dt: float, height: int) -> None:
        if not len(self.p) or dt <= 0:
            return
        dt = min(dt, 0.1)                        # don't jump after a slow frame
        p = self.p
        p[:, 2:4] *= max(0.0, 1 - self.DRAG * dt)
        p[:, 3] += self.GRAVITY * dt
        p[:, 3] = np.minimum(p[:, 3], 380)        # paper falls slowly
        p[:, 0:2] += p[:, 2:4] * dt
        p[:, 6] += 8 * dt
        self.p = p[p[:, 1] < height + 20]

    def draw(self, img) -> None:
        for x, y, _, _, size, ci, spin in self.p:
            # Flutter: the piece's height shrinks and grows as it "spins".
            hh = max(1, int(size * abs(np.cos(spin)) * 0.6))
            x0, y0 = int(x), int(y)
            color = tuple(int(c) for c in CONFETTI_COLORS[int(ci)])
            cv2.rectangle(img, (x0, y0), (x0 + int(size), y0 + hh), color, -1)

    def clear(self) -> None:
        self.p = self.p[:0]
