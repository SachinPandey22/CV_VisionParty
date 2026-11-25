"""Scoring and summary tables for the evaluation (pure: no camera, no UI).

Raw results are one CSV per test run under results/raw/<condition>/. The
`summarize` function turns all of them into the tables used in the report.
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

# Raising both hands also makes the one-hand rules true; that's expected, not
# a false trigger.
IMPLIED = {"BOTH_HANDS_UP": {"LEFT_HAND_UP", "RIGHT_HAND_UP"}}
NEUTRAL = "NEUTRAL"           # "stand still" trials: any detected action is a false trigger


# --- judging single trials --------------------------------------------------
def first_streak(flags: list[bool], times: list[float], need: int) -> float | None:
    """Time of the first frame of the first run of `need` True flags in a row."""
    run, start = 0, None
    for flag, t in zip(flags, times):
        if flag:
            if run == 0:
                start = t
            run += 1
            if run >= need:
                return start
        else:
            run = 0
    return None


def judge_action_trial(frames: list[dict[str, bool]], times: list[float], go: float,
                       target: str, need: int) -> dict:
    """Score one action trial.

    frames: detect_actions() output per frame recorded after GO.
    hit: the target action was detected for `need` frames in a row (for a
    NEUTRAL trial, hit means *nothing* was detected). latency: seconds from GO.
    false_triggers: other actions that were detected for `need` frames.
    """
    names = sorted({k for f in frames for k in f})
    fired = {}
    for name in names:
        t = first_streak([f.get(name, False) for f in frames], times, need)
        if t is not None:
            fired[name] = t - go
    allowed = {target} | IMPLIED.get(target, set())
    false = sorted(n for n in fired if n not in allowed)
    if target == NEUTRAL:
        return {"hit": not fired, "latency": None, "false_triggers": false}
    return {"hit": target in fired, "latency": fired.get(target), "false_triggers": false}


def reaction_from_flags(flags: list[bool], times: list[float], go: float, need: int) -> float | None:
    """Detected reaction time, using the game's rule: the hand must be down at
    some point after GO before the raise counts (no head starts)."""
    armed = False
    run, start = 0, None
    for flag, t in zip(flags, times):
        if t < go:
            continue
        if not flag:
            armed, run = True, 0
            continue
        if not armed:
            continue
        if run == 0:
            start = t
        run += 1
        if run >= need:
            return start - go
    return None


# --- CSV helpers ------------------------------------------------------------
def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_rows(raw_dir: Path, test: str) -> list[dict]:
    """All rows of one test, from results/raw/<condition>/<test>_*.csv."""
    rows = []
    for path in sorted(Path(raw_dir).glob(f"*/{test}_*.csv")):
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _num(v) -> float | None:
    return None if v in (None, "", "None") else float(v)


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def _fmt(v, digits=2, suffix="") -> str:
    return "–" if v is None else f"{v:.{digits}f}{suffix}"


# --- summaries --------------------------------------------------------------
def action_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["action"])].append(r)
    out = []
    for (cond, action), rs in sorted(groups.items()):
        hits = sum(int(r["hit"]) for r in rs)
        false = sum(1 for r in rs if r["false_triggers"])
        out.append({
            "condition": cond, "action": action, "trials": len(rs), "correct": hits,
            "accuracy_pct": round(100 * hits / len(rs), 1),
            "mean_latency_s": _mean(_num(r["latency"]) for r in rs),
            "trials_with_false_trigger": false,
        })
    return out


def fps_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["mode"])].append(r)
    out = []
    for (cond, mode), rs in sorted(groups.items()):
        out.append({
            "condition": cond, "mode": mode, "runs": len(rs),
            "fps": _mean(_num(r["fps"]) for r in rs),
            "pose_ms": _mean(_num(r["pose_ms"]) for r in rs),
            "yolo_ms": _mean(_num(r["yolo_ms"]) for r in rs),
        })
    return out


def reaction_summary(rows: list[dict]) -> list[dict]:
    """Detected reaction time vs. the frame a person marked in the video."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["condition"]].append(r)
    out = []
    for cond, rs in sorted(groups.items()):
        detected = [r for r in rs if _num(r["detected_s"]) is not None]
        marked = [r for r in detected if _num(r["marked_s"]) is not None]
        errs = [_num(r["detected_s"]) - _num(r["marked_s"]) for r in marked]
        out.append({
            "condition": cond, "trials": len(rs), "detected": len(detected),
            "marked": len(marked),
            "mean_detected_s": _mean(_num(r["detected_s"]) for r in detected),
            "mean_error_ms": _mean(1000 * e for e in errs),
            "mean_abs_error_ms": _mean(1000 * abs(e) for e in errs),
            "max_abs_error_ms": max((1000 * abs(e) for e in errs), default=None),
        })
    return out


def similarity_summary(rows: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    """(matrix rows, accept-rate rows). Matrix: mean best score of each
    performed pose against each target. Correct = performed pose vs. its own
    target; wrong = vs. every other target."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r["condition"], r["performed"], r["target"])].append(_num(r["best"]))
    matrix = [{"condition": c, "performed": p, "target": t, "mean_best_pct": _mean(v), "n": len(v)}
              for (c, p, t), v in sorted(cells.items())]
    rates = []
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        correct = [_num(r["best"]) for r in rs if r["performed"] == r["target"]]
        wrong = [_num(r["best"]) for r in rs if r["performed"] != r["target"]]
        acc = lambda vals: 100 * sum(v >= threshold for v in vals) / len(vals) if vals else None
        rates.append({
            "condition": cond,
            "correct_mean_pct": _mean(correct), "wrong_mean_pct": _mean(wrong),
            "correct_accepted_pct": acc(correct),   # want high
            "wrong_accepted_pct": acc(wrong),       # want low
            "n_correct": len(correct), "n_wrong": len(wrong),
        })
    return matrix, rates


def object_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["object"])].append(r)
    out = []
    for (cond, obj), rs in sorted(groups.items()):
        ok = [r for r in rs if int(r["success"])]
        out.append({
            "condition": cond, "object": obj, "trials": len(rs), "success": len(ok),
            "success_pct": round(100 * len(ok) / len(rs), 1),
            "mean_time_s": _mean(_num(r["time_s"]) for r in ok),
            "mean_best_conf": _mean(_num(r["best_conf"]) for r in rs),
        })
    return out


# --- markdown ---------------------------------------------------------------
def md_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    """columns: (key, header, format) where format is '', 'n1', 'n2', 'pct'."""
    if not rows:
        return "_No data yet._\n"
    def cell(v, f):
        if f == "n0":
            return _fmt(v, 0)
        if f == "n1":
            return _fmt(v, 1)
        if f == "n2":
            return _fmt(v, 2)
        if f == "pct":
            return _fmt(v, 1, "%")
        return str(v)
    head = "| " + " | ".join(h for _, h, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(cell(r.get(k), f) for k, _, f in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body]) + "\n"


def summarize(raw_dir: Path, out_dir: Path, threshold: float) -> str:
    """Build every summary table, write CSVs + summary.md, return the markdown."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    acts = action_summary(read_rows(raw_dir, "actions"))
    fps = fps_summary(read_rows(raw_dir, "fps"))
    react = reaction_summary(read_rows(raw_dir, "reaction"))
    matrix, rates = similarity_summary(read_rows(raw_dir, "similarity"), threshold)
    objs = object_summary(read_rows(raw_dir, "objects"))
    for name, rows in [("actions", acts), ("fps", fps), ("reaction", react),
                       ("similarity_matrix", matrix), ("similarity_rates", rates),
                       ("objects", objs)]:
        write_rows(out_dir / f"{name}_summary.csv", rows)

    md = ["# Vision Party — evaluation results", "",
          "Generated by `python -m vision_party.evaluate summary` from the raw CSVs in `results/raw/`.", "",
          "## Action recognition", "",
          md_table(acts, [("condition", "Condition", ""), ("action", "Action", ""),
                          ("trials", "Trials", ""), ("correct", "Correct", ""),
                          ("accuracy_pct", "Accuracy", "pct"),
                          ("mean_latency_s", "Mean latency (s)", "n2"),
                          ("trials_with_false_trigger", "Trials w/ false trigger", "")]),
          "NEUTRAL = stand still; correct means no action was detected.", "",
          "## Frame rate", "",
          md_table(fps, [("condition", "Condition", ""), ("mode", "Mode", ""), ("runs", "Runs", ""),
                         ("fps", "FPS", "n1"), ("pose_ms", "Pose (ms/frame)", "n1"),
                         ("yolo_ms", "YOLO (ms/frame)", "n1")]),
          "## Reaction time vs. video frames", "",
          md_table(react, [("condition", "Condition", ""), ("trials", "Trials", ""),
                           ("detected", "Detected", ""), ("marked", "Marked", ""),
                           ("mean_detected_s", "Mean reaction (s)", "n2"),
                           ("mean_error_ms", "Mean error (ms)", "n0"),
                           ("mean_abs_error_ms", "Mean |error| (ms)", "n0"),
                           ("max_abs_error_ms", "Max |error| (ms)", "n0")]),
          "Error = detected reaction time − movement start marked by hand in the recorded video "
          "(positive = the game detected it later).", "",
          f"## Pose similarity (match threshold {threshold:.0f}%)", "",
          md_table(rates, [("condition", "Condition", ""),
                           ("correct_mean_pct", "Correct pose: mean score", "pct"),
                           ("correct_accepted_pct", "Correct pose: accepted", "pct"),
                           ("wrong_mean_pct", "Wrong pose: mean score", "pct"),
                           ("wrong_accepted_pct", "Wrong pose: accepted", "pct")]),
          "Full score matrix (performed pose × target): `results/similarity_matrix_summary.csv`.", "",
          "## Bring to Camera (object detection)", "",
          md_table(objs, [("condition", "Condition", ""), ("object", "Object", ""),
                          ("trials", "Trials", ""), ("success", "Success", ""),
                          ("success_pct", "Success rate", "pct"),
                          ("mean_time_s", "Mean time (s)", "n1"),
                          ("mean_best_conf", "Mean best confidence", "n2")]),
          ]
    text = "\n".join(md)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(text)
    return text
