"""Drawing helpers for the OpenCV window. Nothing here changes game state."""

import cv2
import numpy as np

from .engine import Match, Phase
from .pose import CONNECTIONS, JOINTS, Pose, joint_angles

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, RED, YELLOW = (90, 220, 90), (70, 70, 230), (60, 220, 255)
PLAYER_COLORS = [(255, 170, 60), (60, 200, 255)]   # BGR: P1 blue, P2 orange


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


def draw_match(img, match: Match, poses: list[Pose | None], now: float) -> None:
    h, w = img.shape[:2]
    game = match.game

    # Top bar: round, game title, scores.
    shade(img, 0, 0, w, 60)
    text(img, f"Round {match.round_index + 1}/{match.rounds}  -  {game.title}", (20, 40), 0.8)
    for i, name in enumerate(match.names):
        x = 20 + i * (w // 2) if match.n_players == 2 else w - 260
        y = 95
        label = f"{name}: {match.scores[i]}"
        if poses[i] is None:
            label += "  (not visible)"
        text(img, label, (x, y), 0.8, PLAYER_COLORS[i])

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
        _timer_bar(img, game.time_left(now) / game.time_limit)
        _player_lines(img, match, game.status(now))
    elif phase is Phase.RESULT and match.last_result:
        res = match.last_result
        color = GREEN if res.winners else RED
        _banner(img, match.named(res.message), color)
        _player_lines(img, match, [
            f"{d}  +{p}" if p else d for d, p in zip(res.details, res.points)])
    elif phase is Phase.FINISHED:
        draw_winner(img, match)


def draw_winner(img, match: Match) -> None:
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
    center_text(img, "R  Play again      M  Menu      Q  Quit", w // 2, h - 60, 0.8)


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
