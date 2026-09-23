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
    seq2seq = commands.add_parser("seq2seq", help="train or use the attention Seq2Seq baseline")
    seq_actions = seq2seq.add_subparsers(dest="action", required=True)
    train_command = seq_actions.add_parser("train", help="train or resume from an epoch boundary")
    train_command.add_argument("--data-dir", type=Path, required=True)
    train_command.add_argument("--config", type=Path, required=True)
    train_command.add_argument("--run-dir", type=Path, required=True)
    train_command.add_argument("--resume", type=Path)
    infer_command = seq_actions.add_parser("infer", help="greedy ad-hoc inference")
    infer_command.add_argument("--checkpoint", type=Path, required=True)
    infer_command.add_argument("--text", required=True)
    predict_command = seq_actions.add_parser("predict", help="write strict M0 prediction files")
    predict_command.add_argument("--data-dir", type=Path, required=True)
    predict_command.add_argument("--checkpoint", type=Path, required=True)
    predict_command.add_argument("--split", choices=("train", "validation", "test"), required=True)
    predict_command.add_argument("--output", type=Path, required=True)
    predict_command.add_argument("--metadata-output", type=Path, required=True)
    predict_command.add_argument("--selection", type=Path)
    evaluation = commands.add_parser("evaluate", help="evaluate a strict prediction file")
    evaluation.add_argument("--data-dir", type=Path, required=True)
    evaluation.add_argument("--split", choices=("train", "validation", "test"), required=True)
    evaluation.add_argument("--predictions", type=Path, required=True)
    evaluation.add_argument("--metadata", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "data" and args.action == "validate":
            records, counts = load_records(args.input)
            counts["leakage_groups"] = sum(assign_splits(records, 42).values())
            print(json.dumps(counts, sort_keys=True))
        elif args.command == "data":
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
        elif args.command == "seq2seq":
            if args.action == "train":
                from nl2code.seq2seq.train import train

                print(json.dumps(train(args.data_dir, args.config, args.run_dir, args.resume)))
            elif args.action == "infer":
                from nl2code.seq2seq.predict import infer

                print(infer(args.checkpoint, args.text))
            else:
                from nl2code.seq2seq.predict import predict

                print(
                    json.dumps(
                        predict(
                            args.data_dir,
                            args.checkpoint,
                            args.split,
                            args.output,
                            args.metadata_output,
                            args.selection,
                        )
                    )
                )
        else:
            from nl2code.data.contract import sha256
            from nl2code.evaluation.metrics import evaluate
            from nl2code.evaluation.predictions import validate_predictions
            from nl2code.seq2seq.dataset import load_split
            from nl2code.seq2seq.runtime import write_json

            manifest, manifest_hash, rows, split_hash = load_split(args.data_dir, args.split)
            prediction_bytes = args.predictions.read_bytes()
            predictions = [json.loads(line) for line in prediction_bytes.splitlines()]
            metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
            validate_predictions(rows, predictions, metadata, manifest_hash, split_hash, args.split)
            result = {
                "schema_version": "evaluation-v1",
                "split": metadata["split"],
                "data_manifest_sha256": manifest_hash,
                "canonical_dataset_sha256": manifest["canonical_sha256"],
                "split_file_sha256": split_hash,
                "prediction_file_sha256": sha256(prediction_bytes),
                "config_sha256": metadata["config_sha256"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "generation_settings": metadata["generation_settings"],
                "prediction_record_count": metadata["record_count"],
                "metrics": evaluate(rows, predictions),
            }
            write_json(args.output, result)
            print(json.dumps(result, sort_keys=True))
    except (DataError, ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"nl2code: {error}\n")
