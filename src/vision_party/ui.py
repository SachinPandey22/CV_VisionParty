"""Drawing helpers for the OpenCV window. Nothing here changes game state."""

import cv2
import numpy as np

from . import config, fun
from .engine import Match, Phase
from .pose import CONNECTIONS, JOINTS, Pose, joint_angles
from .similarity import TargetPose

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, RED, YELLOW = (90, 220, 90), (70, 70, 230), (60, 220, 255)
PLAYER_COLORS = [(255, 170, 60), (60, 200, 255)]   # BGR: P1 blue, P2 orange
ORANGE = (40, 140, 255)
# One color per rating level: PERFECT, GREAT, GOOD, OK.
RATING_COLORS = [(60, 220, 255), (90, 220, 90), (255, 200, 80), (200, 200, 200)]


# --- primitives -------------------------------------------------------------
def text(img, s: str, org, scale=0.8, color=WHITE, thickness=2) -> None:
    """Text with a dark outline so it stays readable over any background."""
    cv2.putText(img, s, org, FONT, scale, BLACK, thickness + 4, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thickness, cv2.LINE_AA)


def center_text(img, s: str, cx: int, y: int, scale=1.0, color=WHITE, thickness=2) -> None:
    (w, _), _ = cv2.getTextSize(s, FONT, scale, thickness)
    text(img, s, (cx - w // 2, y), scale, color, thickness)


def shade(img, x0, y0, x1, y1, alpha=0.55) -> None:
    """Darken a rectangle (for banners behind text)."""
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha)).astype(np.uint8)


# --- skeleton ---------------------------------------------------------------
def to_display(pose: Pose, width: int) -> np.ndarray:
    """Raw-frame pixel coords -> mirrored display coords."""
    pts = pose.points.copy()
    pts[:, 0] = width - 1 - pts[:, 0]
    return pts.astype(int)


def draw_skeleton(img, pose: Pose, color, show_angles: bool = False) -> None:
    pts = to_display(pose, img.shape[1])
    for a, b in CONNECTIONS:
        if pose.visible(a, b):
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 3, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if pose.visible(i):
            cv2.circle(img, tuple(p), 4, WHITE, -1, cv2.LINE_AA)
    if show_angles:
        for name, value in joint_angles(pose).items():
            if value is not None:
                vx = pts[JOINTS[name][1]]
                cv2.putText(img, f"{value:.0f}", (vx[0] + 6, vx[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def draw_object_box(img, game) -> None:
    """Bring to Camera: box around the target object, green once it's close."""
    box = getattr(game, "box", None)
    if box is None:
        return
    w = img.shape[1]
    x0, y0, x1, y1 = box
    x0, x1 = w - 1 - x1, w - 1 - x0          # raw -> mirrored display coords
    close = game.closeness >= 1.0
    color = GREEN if close else YELLOW
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
    # Label under the box: above it would sit behind the prompt banner.
    label_y = min(img.shape[0] - 30, y1 + 30)
    text(img, f"{game.target} {min(100, int(game.closeness * 100))}%", (x0, label_y), 0.7, color)


def draw_divider(img) -> None:
    h, w = img.shape[:2]
    cv2.line(img, (w // 2, 0), (w // 2, h), WHITE, 2)


# --- screens ----------------------------------------------------------------
def draw_menu(img) -> None:
    h, w = img.shape[:2]
    shade(img, 0, 0, w, h, 0.5)
    center_text(img, "VISION PARTY", w // 2, h // 2 - 80, 2.0, YELLOW, 4)
    center_text(img, "1  Solo        2  Duo        Q  Quit", w // 2, h // 2 + 10, 1.0)
    center_text(img, "Stand back so your whole body is in view", w // 2, h // 2 + 70, 0.7)


def draw_match(img, match: Match, poses: list[Pose | None], now: float,
               moments=None) -> None:
    h, w = img.shape[:2]
    game = match.game

    # Top bar: round, game title, scores.
    shade(img, 0, 0, w, 60)
    text(img, f"Round {match.round_index + 1}/{match.rounds}  -  {game.title}", (20, 40), 0.8)
    for i, name in enumerate(match.names):
        y = 95
        label = f"{name}: {match.scores[i]}"
        if poses[i] is None:
            label += "  (not visible)"
        if match.n_players == 2:
            x = 20 + i * (w // 2)
        else:   # right-aligned, so longer labels never run off the screen
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.8, 2)
            x = w - 20 - tw
        text(img, label, (x, y), 0.8, PLAYER_COLORS[i])
        badge = fun.streak_label(match.streaks[i])
        if badge:
            hot = match.streaks[i] >= config.STREAK_ON_FIRE
            text(img, badge, (x, y + 32), 0.7, ORANGE if hot else WHITE)

    phase = match.phase
    if phase is Phase.INTRO:
        _banner(img, game.prompt, YELLOW)
    elif phase is Phase.COUNTDOWN:
        _banner(img, game.prompt, YELLOW, small=True)
        center_text(img, str(match.countdown_value(now)), w // 2, h // 2 + 60, 4.0, WHITE, 8)
    elif phase is Phase.ACTIVE:
        if match.show_go(now):
            center_text(img, "GO!", w // 2, h // 2 + 60, 4.0, GREEN, 8)
        _banner(img, game.prompt, YELLOW, small=True)
        _timer_bar(img, game.time_left(now) / game.effective_limit)
        _player_lines(img, match, game.status(now))
    elif phase is Phase.RESULT and match.last_result:
        res = match.last_result
        color = GREEN if res.winners else RED
        _banner(img, match.named(res.message), color)
        _player_lines(img, match, [
            f"{d}  +{p}" if p else d for d, p in zip(res.details, res.points)])
        _ratings(img, match, res, match.phase_elapsed(now))
    elif phase is Phase.FINISHED:
        draw_winner(img, match, moments)

    # Pose Battle: show the pose to copy (drawn last so nothing covers it).
    target = getattr(game, "target", None)
    if isinstance(target, TargetPose) and phase in (Phase.INTRO, Phase.COUNTDOWN, Phase.ACTIVE):
        draw_reference(img, target)


def _ratings(img, match: Match, res, t: float) -> None:
    """Big PERFECT! / GREAT! / GOOD / OK under each player who reacted.
    It pops in large and settles to its normal size in the first 0.25 s."""
    h, w = img.shape[:2]
    pop = 1.0 + 0.8 * max(0.0, 1.0 - t / 0.25)
    for i in range(match.n_players):
        if res.accuracy:                       # Pose Battle: rate the accuracy
            r = fun.accuracy_rating(res.accuracy[i])
        else:                                  # otherwise rate reaction time
            r = fun.rating(res.times[i] if i < len(res.times) else None)
        if r is None:
            continue
        label, level = r
        cx = w // 2 if match.n_players == 1 else (w // 4 if i == 0 else 3 * w // 4)
        center_text(img, label, cx, h // 2 + 110, 1.6 * pop, RATING_COLORS[level],
                    max(3, int(4 * pop)))


def draw_winner(img, match: Match, moments=None) -> None:
    h, w = img.shape[:2]
    shade(img, 0, 0, w, h, 0.6)
    center_text(img, "GAME OVER", w // 2, h // 2 - 120, 2.0, YELLOW, 4)
    if match.n_players == 2:
        win = match.winners()
        head = "It's a tie!" if len(win) > 1 else f"{match.names[win[0]]} wins!"
        center_text(img, head, w // 2, h // 2 - 40, 1.4, GREEN, 3)
    for i, name in enumerate(match.names):
        center_text(img, f"{name}: {match.scores[i]} pts", w // 2, h // 2 + 30 + 50 * i,
                    1.1, PLAYER_COLORS[i])
    if moments is not None:
        _moment_photos(img, match, moments)
    center_text(img, "R  Play again      M  Menu      Q  Quit", w // 2, h - 25, 0.8)


def _moment_photos(img, match: Match, moments) -> None:
    """Each player's fastest-reaction photo, in a row near the bottom."""
    h, w = img.shape[:2]
    shots = [(i, b) for i, b in enumerate(moments.best) if b is not None]
    if not shots:
        return
    top = h // 2 + 30 + 50 * match.n_players
    box_h = h - 60 - top
    if box_h < 60:
        return
    thumbs = []
    for i, (secs, photo) in shots:
        scale = box_h / photo.shape[0]
        thumbs.append((i, secs, cv2.resize(photo, (int(photo.shape[1] * scale), box_h))))
    gap = 30
    total = sum(t.shape[1] for _, _, t in thumbs) + gap * (len(thumbs) - 1)
    x = max(0, (w - total) // 2)
    for i, secs, t in thumbs:
        tw = min(t.shape[1], w - x)
        if tw <= 0:
            break
        img[top:top + box_h, x:x + tw] = t[:, :tw]
        cv2.rectangle(img, (x, top), (x + tw - 1, top + box_h - 1), PLAYER_COLORS[i], 4)
        shade(img, x, top + box_h - 52, x + tw, top + box_h, 0.5)
        text(img, match.names[i], (x + 8, top + box_h - 30), 0.55, PLAYER_COLORS[i], 1)
        text(img, f"fastest {secs:.2f}s", (x + 8, top + box_h - 9), 0.55, WHITE, 1)
        x += tw + gap


# Bones of the reference stick figure (landmark id pairs).
_REF_BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
              (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


def draw_reference(img, target, box_w: int = 230, box_h: int = 300) -> None:
    """The target pose as a stick figure in a panel at the top right.
    Drawn mirrored, so it looks like the player's own mirror image."""
    h, w = img.shape[:2]
    x0, y0 = w - box_w - 15, 210          # below the instruction banner
    shade(img, x0, y0, x0 + box_w, y0 + box_h, 0.65)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), YELLOW, 2)
    center_text(img, f"Copy: {target.name}", x0 + box_w // 2, y0 + 30, 0.7, YELLOW, 2)

    pts = {k: np.array([-v[0], v[1]], dtype=np.float32) for k, v in target.points.items()}
    allp = np.array(list(pts.values()))
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    area_top, area_h = y0 + 50, box_h - 65
    scale = min((box_w - 40) / max(hi[0] - lo[0], 1e-3), area_h / max(hi[1] - lo[1], 1e-3))
    cx = x0 + box_w / 2 - (lo[0] + hi[0]) / 2 * scale
    cy = area_top + area_h / 2 - (lo[1] + hi[1]) / 2 * scale
    px = {k: (int(cx + p[0] * scale), int(cy + p[1] * scale)) for k, p in pts.items()}
    for a, b in _REF_BONES:
        if a in px and b in px:
            cv2.line(img, px[a], px[b], WHITE, 4, cv2.LINE_AA)
    if 0 in px and 11 in px and 12 in px:
        head_r = max(6, int(0.18 * scale))
        cv2.circle(img, px[0], head_r, WHITE, 3, cv2.LINE_AA)


def _banner(img, s: str, color, small: bool = False) -> None:
    h, w = img.shape[:2]
    y = 170 if small else h // 2
    scale = 1.1 if small else 1.6
    shade(img, 0, y - 50, w, y + 25)
    center_text(img, s, w // 2, y, scale, color, 3)


def _timer_bar(img, frac: float) -> None:
    h, w = img.shape[:2]
    frac = max(0.0, min(1.0, frac))
    cv2.rectangle(img, (0, h - 14), (w, h), (60, 60, 60), -1)
    color = GREEN if frac > 0.5 else (YELLOW if frac > 0.2 else RED)
    cv2.rectangle(img, (0, h - 14), (int(w * frac), h), color, -1)


def _player_lines(img, match: Match, lines: list[str]) -> None:
    h, w = img.shape[:2]
    for i, line in enumerate(lines):
        if not line:
            continue
        cx = w // 2 if match.n_players == 1 else (w // 4 if i == 0 else 3 * w // 4)
        center_text(img, line, cx, h - 50, 1.0, PLAYER_COLORS[i], 2)
