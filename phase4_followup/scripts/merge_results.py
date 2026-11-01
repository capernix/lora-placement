#!/usr/bin/env python3
"""Validate and copy follow-up artifacts without overwriting Phase 4 results."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from validate_result import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT / "phase4-18runs")
    parser.add_argument("--force", action="store_true", help="explicitly replace duplicate completed runs")
    args = parser.parse_args()
    records = validate(args.incoming, args.manifest)
    args.destination.mkdir(parents=True, exist_ok=True)
    for record in records:
        source = args.incoming / record["run_id"]
        target = args.destination / record["run_id"]
        if target.exists() and not args.force:
            raise SystemExit(f"REFUSING TO OVERWRITE EXISTING RUN: {target}; use --force explicitly")
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.copytree(source, target)
        print(f"MERGED {record['run_id']}")
    print(f"PASS: merged {len(records)} validated run(s) into {args.destination}")


if __name__ == "__main__":
    main()
