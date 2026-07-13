#!/usr/bin/env python3
"""Verify hashes of the aligned benchmark fixture and frozen FROST reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = {
    "data/test_fixture_mc100_aligned.h5": "2ec17bd1f434d442cb668dcf477198517bdef107b1a7aa384eb5623ffd0c4358",
    "reference/frost_doa_frozen_sample_scores.csv": "6ae65e0054dda2c6a950b622e170136c82df3b08b73e1a75ecef8dc6e668d6d8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("audit/asset_hashes.json"))
    args = parser.parse_args()
    rows = []
    passed = True
    for relative, expected in EXPECTED.items():
        path = args.root / relative
        actual = sha256(path) if path.exists() else None
        ok = actual == expected
        passed &= ok
        rows.append(
            {"path": relative, "expected_sha256": expected, "actual_sha256": actual, "passed": ok}
        )
    report = {"passed": passed, "assets": rows}
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
