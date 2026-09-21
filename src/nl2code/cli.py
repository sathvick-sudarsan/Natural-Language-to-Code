"""Explicit data validation and preparation commands."""

import argparse
import json
from pathlib import Path

from nl2code.data.contract import DataError, load_records
from nl2code.data.prepare import prepare
from nl2code.data.split import assign_splits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nl2code")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="validate or prepare the v1 dataset")
    actions = data.add_subparsers(dest="action", required=True)
    validate = actions.add_parser("validate", help="validate and count rows")
    validate.add_argument("--input", type=Path, required=True)
    prepare_command = actions.add_parser("prepare", help="write deterministic split artifacts")
    prepare_command.add_argument("--input", type=Path, required=True)
    prepare_command.add_argument("--output", type=Path, required=True)
    prepare_command.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.action == "validate":
            records, counts = load_records(args.input)
            counts["leakage_groups"] = sum(assign_splits(records, 42).values())
            print(json.dumps(counts, sort_keys=True))
        else:
            manifest = prepare(args.input, args.output, args.seed)
            print(
                json.dumps(
                    {
                        key: manifest[key]
                        for key in (
                            "canonical_row_count",
                            "leakage_group_count",
                            "row_counts",
                            "group_counts",
                        )
                    },
                    sort_keys=True,
                )
            )
    except (DataError, OSError) as error:
        parser.exit(2, f"nl2code: {error}\n")
