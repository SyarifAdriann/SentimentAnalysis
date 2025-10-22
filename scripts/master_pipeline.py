"""Master orchestration entry-point for the Twitter Airline Sentiment project.

This script coordinates execution for each defined phase. Individual phase
implementations live in the src/ package and may be invoked independently.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import subprocess
import sys

PHASES: Tuple[Tuple[str, str], ...] = (
    ("phase0", "Phase 0 – System Check & Environment Setup"),
    ("phase1", "Phase 1 – Data Loading & Validation"),
    ("phase2", "Phase 2 – Exploratory Data Analysis"),
    ("phase3", "Phase 3 – Sentiment Analysis & Pattern Discovery"),
    ("phase4", "Phase 4 – Predictive Modeling"),
    ("phase5", "Phase 5 – Visualization Creation"),
    ("phase6", "Phase 6 – Report Generation"),
    ("phase7", "Phase 7 – Database Setup"),
    ("phase8", "Phase 8 – Web Dashboard Development"),
    ("phase9", "Phase 9 – Deployment"),
    ("phase10", "Phase 10 – Automated Testing"),
    ("phase11", "Phase 11 – Final Integration & Launch"),
)

SRC_RUNNER = Path("src") / "pipeline" / "run_phase.py"


def _ensure_runner_exists() -> None:
    if not SRC_RUNNER.exists():
        print(f"[WARN] Phase runner module not found at {SRC_RUNNER}.", file=sys.stderr)
        print("       Please implement src/pipeline/run_phase.py before executing phases.")


def run_phase(phase: str) -> None:
    _ensure_runner_exists()
    if not SRC_RUNNER.exists():
        return

    cmd = [sys.executable, str(SRC_RUNNER), "--phase", phase]
    print(f"[INFO] Executing {phase} via {SRC_RUNNER}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Phase '{phase}' failed with exit code {exc.returncode}.", file=sys.stderr)
        raise


def run_multiple(phases: Iterable[str]) -> None:
    for phase in phases:
        run_phase(phase)


def run_all() -> None:
    phase_ids = [phase for phase, _ in PHASES]
    run_multiple(phase_ids)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project orchestration helper")
    parser.add_argument(
        "--phase",
        choices=[phase for phase, _ in PHASES],
        help="Run only the specified phase",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available phases",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list:
        print("Available phases:")
        for phase, description in PHASES:
            print(f"- {phase}: {description}")
        return

    if args.phase:
        run_phase(args.phase)
    else:
        run_all()


if __name__ == "__main__":
    main()
