import hashlib
import json

import pytest

from nl2code.cli import main
from nl2code.data.contract import sha256, stable_json
from nl2code.evaluation.metrics import evaluate
from nl2code.evaluation.predictions import validate_predictions, write_predictions

ROWS = [
    {"record_id": "a", "normalized_intent": "sort", "snippet": "x = 1\n"},
    {"record_id": "b", "normalized_intent": "sort", "snippet": "x=1"},
    {"record_id": "c", "normalized_intent": "blank", "snippet": "pass"},
]
PREDICTIONS = [
    {"record_id": "a", "normalized_intent": "sort", "prediction": "x=1\r\n"},
    {"record_id": "b", "normalized_intent": "sort", "prediction": "x=1\r\n"},
    {"record_id": "c", "normalized_intent": "blank", "prediction": "  "},
]
META = {
    "schema_version": "prediction-v1",
    "model_family": "seq2seq",
    "split": "validation",
    "record_count": 3,
    "data_manifest_sha256": "a" * 64,
    "split_file_sha256": "b" * 64,
    "config_sha256": "c" * 64,
    "checkpoint_sha256": "d" * 64,
    "generation_settings": {"strategy": "greedy", "max_tokens": 8},
    "code_revision": "rev",
}


def test_multi_reference_metrics_and_blank_syntax():
    result = evaluate(ROWS, PREDICTIONS)
    assert result["intent_any_reference_exact_match"] == {
        "matched": 1,
        "total": 2,
        "value": 0.5,
    }
    assert result["row_exact_match"]["matched"] == 1
    assert result["row_exact_match"]["diagnostic"] is True
    assert result["python_syntax_validity"]["valid"] == 1
    assert result["empty_prediction_count"] == 1
    assert result["multi_reference_group_count"] == 1


@pytest.mark.parametrize(
    "changed",
    [
        PREDICTIONS[:2],
        PREDICTIONS + [PREDICTIONS[0]],
        PREDICTIONS[:2] + [{**PREDICTIONS[2], "record_id": "unknown"}],
        [{**PREDICTIONS[0], "normalized_intent": "wrong"}] + PREDICTIONS[1:],
        [PREDICTIONS[0], {**PREDICTIONS[1], "prediction": "other"}, PREDICTIONS[2]],
    ],
)
def test_prediction_rows_reject_contract_errors(changed):
    with pytest.raises(ValueError):
        validate_predictions(ROWS, changed, META, "a" * 64, "b" * 64, "validation")


def test_metadata_rejects_mismatch():
    with pytest.raises(ValueError):
        validate_predictions(
            ROWS,
            PREDICTIONS,
            {**META, "split_file_sha256": "wrong"},
            "a" * 64,
            "b" * 64,
            "validation",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("config_sha256", "short"),
        ("checkpoint_sha256", "z" * 64),
        ("data_manifest_sha256", None),
        ("split_file_sha256", "A" * 64),
        ("generation_settings", None),
        ("generation_settings", {}),
        ("generation_settings", {"strategy": "greedy"}),
        ("generation_settings", {"strategy": "beam", "max_tokens": 8}),
        ("generation_settings", {"strategy": "greedy", "max_tokens": 0}),
    ],
)
def test_prediction_metadata_rejects_invalid_identity(field, value):
    with pytest.raises(ValueError):
        validate_predictions(
            ROWS,
            PREDICTIONS,
            {**META, field: value},
            "a" * 64,
            "b" * 64,
            "validation",
        )


@pytest.mark.parametrize(
    "field",
    ["data_manifest_sha256", "split_file_sha256", "config_sha256", "checkpoint_sha256"],
)
def test_prediction_metadata_rejects_missing_digest(field):
    metadata = {key: value for key, value in META.items() if key != field}
    with pytest.raises(ValueError):
        validate_predictions(ROWS, PREDICTIONS, metadata, "a" * 64, "b" * 64, "validation")


def test_prediction_metadata_rejects_missing_generation_object():
    metadata = {key: value for key, value in META.items() if key != "generation_settings"}
    with pytest.raises(ValueError):
        validate_predictions(ROWS, PREDICTIONS, metadata, "a" * 64, "b" * 64, "validation")


def _evaluation_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    split_bytes = b"".join(stable_json(row) + b"\n" for row in ROWS)
    (data_dir / "validation.jsonl").write_bytes(split_bytes)
    manifest = {
        "manifest_version": "v1",
        "schema_version": "v1",
        "canonical_sha256": "e" * 64,
        "file_sha256": {"validation.jsonl": sha256(split_bytes)},
        "row_counts": {"validation": len(ROWS)},
    }
    manifest_bytes = stable_json(manifest) + b"\n"
    (data_dir / "manifest.json").write_bytes(manifest_bytes)
    metadata = {
        **META,
        "data_manifest_sha256": sha256(manifest_bytes),
        "split_file_sha256": sha256(split_bytes),
    }
    predictions_path = tmp_path / "predictions.jsonl"
    write_predictions(predictions_path, PREDICTIONS)
    metadata_path = tmp_path / "predictions.meta.json"
    metadata_path.write_bytes(stable_json(metadata) + b"\n")
    return data_dir, predictions_path, metadata_path, manifest, metadata


def _evaluate_cli(data_dir, predictions_path, metadata_path, output):
    main(
        [
            "evaluate",
            "--data-dir",
            str(data_dir),
            "--split",
            "validation",
            "--predictions",
            str(predictions_path),
            "--metadata",
            str(metadata_path),
            "--output",
            str(output),
        ]
    )


def test_saved_results_identify_exact_prediction_bytes(tmp_path):
    data_dir, predictions_path, metadata_path, manifest, metadata = _evaluation_files(tmp_path)
    output = tmp_path / "results.json"
    original = predictions_path.read_bytes()
    _evaluate_cli(data_dir, predictions_path, metadata_path, output)
    result = json.loads(output.read_text())
    assert result["schema_version"] == "evaluation-v1"
    assert result["split"] == "validation"
    assert result["data_manifest_sha256"] == metadata["data_manifest_sha256"]
    assert result["canonical_dataset_sha256"] == manifest["canonical_sha256"]
    assert result["split_file_sha256"] == metadata["split_file_sha256"]
    assert result["config_sha256"] == metadata["config_sha256"]
    assert result["checkpoint_sha256"] == metadata["checkpoint_sha256"]
    assert result["generation_settings"] == metadata["generation_settings"]
    assert result["prediction_record_count"] == len(ROWS)
    assert result["prediction_file_sha256"] == hashlib.sha256(original).hexdigest()
    assert result["metrics"]["intent_any_reference_exact_match"]["matched"] == 1
    predictions_path.write_bytes(original.replace(b"\n", b"  \n"))
    _evaluate_cli(data_dir, predictions_path, metadata_path, output)
    changed = json.loads(output.read_text())
    assert changed["prediction_file_sha256"] == sha256(predictions_path.read_bytes())
    assert changed["prediction_file_sha256"] != result["prediction_file_sha256"]


def test_invalid_metadata_produces_no_results(tmp_path):
    data_dir, predictions_path, metadata_path, _, metadata = _evaluation_files(tmp_path)
    metadata_path.write_bytes(stable_json({**metadata, "config_sha256": "bad"}) + b"\n")
    output = tmp_path / "results.json"
    with pytest.raises(SystemExit):
        _evaluate_cli(data_dir, predictions_path, metadata_path, output)
    assert not output.exists()


def test_base_import_without_torch():
    import nl2code
    import nl2code.cli

    assert nl2code and nl2code.cli
