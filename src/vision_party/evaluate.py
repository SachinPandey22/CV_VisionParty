"""Evaluation for the report.

Run from the project folder:

  python -m vision_party.evaluate all                    full run (~30 min)
  python -m vision_party.evaluate all --condition lowlight --quick
                                                         poor-lighting rerun (~10 min)
  python -m vision_party.evaluate review-reaction        mark movement starts in the videos
  python -m vision_party.evaluate summary                build results/summary.md

Single tests: actions, fps, reaction, similarity, objects.
Raw results go to results/raw/<condition>/ (not committed: the reaction
videos show the players); the summary tables go to results/ (commit these).
"""

import argparse
from pathlib import Path

from . import config
from .evaluation import runner
from .evaluation.metrics import summarize

TESTS = ["actions", "fps", "reaction", "similarity", "objects"]

# (full, quick) trial counts per test
COUNTS = {"actions": (20, 5), "reaction": (10, 5), "similarity": (3, 1), "objects": (5, 2)}


def _detector():
    from .objects import ObjectDetector
    det = ObjectDetector()
    print("Loading YOLOv8n (downloads ~6 MB the first time)...")
    det.load()
    if not det.ready:
        print(f"YOLO not available ({det.error}); skipping object tests.")
        return None
    return det


def run_tests(tests: list[str], args) -> None:
    det = _detector() if ("objects" in tests or "fps" in tests) else None
    session = runner.Session(args.camera)
    n = lambda test: args.trials or COUNTS[test][1 if args.quick else 0]
    try:
        for test in tests:
            try:
                if test == "actions":
                    runner.run_actions(session, args.condition, n(test))
                elif test == "fps":
                    runner.run_fps(session, args.condition, 8.0 if args.quick else 15.0, det)
                elif test == "reaction":
                    runner.run_reaction(session, args.condition, n(test))
                elif test == "similarity":
                    runner.run_similarity(session, args.condition, n(test))
                elif test == "objects":
                    if det is None:
                        continue
                    session.message([("Next: objects. Get a bottle, cup, book and phone,", runner.ui.YELLOW),
                                     ("and keep them out of view.", runner.ui.WHITE)], 5.0)
                    runner.run_objects(session, args.condition, det, n(test))
            except runner.Quit:
                pass
            if test != tests[-1]:
                session.message([(f"Done: {test}. Take a short break.", runner.ui.GREEN),
                                 (f"Next: {tests[tests.index(test) + 1]}", runner.ui.WHITE)], 4.0)
    except runner.Quit:
        print("Stopped.")
    finally:
        session.close()


def latest_reaction_csv(condition: str) -> Path | None:
    files = sorted((runner.RAW / condition).glob("reaction_*.csv"))
    return files[-1] if files else None


def main() -> None:
    p = argparse.ArgumentParser(description="Vision Party evaluation (see module docstring)")
    p.add_argument("test", choices=["all", *TESTS, "review-reaction", "summary"])
    p.add_argument("--condition", default="normal",
                   help="label for this run, e.g. normal, lowlight, busy (default: normal)")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    p.add_argument("--trials", type=int, default=None, help="override the number of trials")
    p.add_argument("--quick", action="store_true", help="fewer trials (for the poor-lighting rerun)")
    args = p.parse_args()

    if args.test == "summary":
        print(summarize(runner.RAW, runner.RESULTS, config.POSE_MATCH_THRESHOLD))
        print(f"\nWrote {runner.RESULTS / 'summary.md'} and the *_summary.csv tables.")
    elif args.test == "review-reaction":
        path = latest_reaction_csv(args.condition)
        if path is None:
            print(f"No reaction recordings for condition '{args.condition}' yet.")
        else:
            runner.review_reaction(path)
    else:
        run_tests(TESTS if args.test == "all" else [args.test], args)


if __name__ == "__main__":
    main()
