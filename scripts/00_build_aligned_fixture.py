#!/usr/bin/env python3
"""Generate the exact MC100 fixture aligned with the frozen FROST-DOA scores."""

from __future__ import annotations
import argparse
from pathlib import Path
from frost_doa.fixture import generate_aligned_mc100_fixture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/test_fixture_mc100_aligned.h5")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=20261030)
    args = parser.parse_args()
    path = generate_aligned_mc100_fixture(args.output, args.trials, args.seed_base)
    print(f"Aligned fixture written to {Path(path).resolve()}")


if __name__ == "__main__":
    main()
