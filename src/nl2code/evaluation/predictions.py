"""Strict M0 prediction file contract."""

import json
import re
from pathlib import Path

from nl2code.data.contract import stable_json

REQUIRED_METADATA = {
    "schema_version",
    "model_family",
    "split",
    "record_count",
    "data_manifest_sha256",
    "split_file_sha256",
    "config_sha256",
    "checkpoint_sha256",
    "generation_settings",
    "code_revision",
}
PREDICTION_FIELDS = {"record_id", "normalized_intent", "prediction"}


def validate_predictions(rows, predictions, metadata, manifest_sha, split_sha, split):
    if (
        not isinstance(metadata, dict)
        or set(metadata) != REQUIRED_METADATA
        or any(
            metadata[key] != expected
            for key, expected in (
                ("schema_version", "prediction-v1"),
                ("split", split),
                ("record_count", len(rows)),
                ("data_manifest_sha256", manifest_sha),
                ("split_file_sha256", split_sha),
            )
        )
    ):
        raise ValueError("prediction metadata mismatch")
    if not all(
        isinstance(metadata[key], str) and re.fullmatch(r"[0-9a-f]{64}", metadata[key])
        for key in (
            "data_manifest_sha256",
            "split_file_sha256",
            "config_sha256",
            "checkpoint_sha256",
        )
    ):
        raise ValueError("invalid prediction SHA-256 identity")
    settings = metadata["generation_settings"]
    if (
        not isinstance(metadata["model_family"], str)
        or not metadata["model_family"]
        or not isinstance(metadata["code_revision"], str)
        or not metadata["code_revision"]
        or not isinstance(settings, dict)
        or settings.get("strategy") != "greedy"
        or type(settings.get("max_tokens")) is not int
        or settings["max_tokens"] < 1
    ):
        raise ValueError("incompatible prediction metadata")
    expected = {row["record_id"]: row["normalized_intent"] for row in rows}
    if len(expected) != len(rows):
        raise ValueError("duplicate record ID in split")
    seen = {}
    by_intent = {}
    for prediction in predictions:
        if set(prediction) != PREDICTION_FIELDS or not all(
            isinstance(value, str) for value in prediction.values()
        ):
            raise ValueError("invalid prediction row")
        record_id = prediction["record_id"]
        intent = prediction["normalized_intent"]
        if record_id in seen or record_id not in expected or expected[record_id] != intent:
            raise ValueError("duplicate, unknown, or mismatched prediction ID")
        if intent in by_intent and by_intent[intent] != prediction["prediction"]:
            raise ValueError("inconsistent prediction for normalized intent")
        seen[record_id] = prediction
        by_intent[intent] = prediction["prediction"]
    if set(seen) != set(expected) or len(seen) != metadata["record_count"]:
        raise ValueError("missing prediction IDs")
    if list(seen) != sorted(seen):
        raise ValueError("predictions must be sorted by record ID")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_predictions(path: Path, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(stable_json(row) + b"\n" for row in predictions))
