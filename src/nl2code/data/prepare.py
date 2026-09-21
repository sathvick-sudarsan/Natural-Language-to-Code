"""Deterministic, local-only data artifact preparation."""

from pathlib import Path

from nl2code.data.contract import FIELDS, SCHEMA_VERSION, load_records, sha256, stable_json
from nl2code.data.normalize import NORMALIZATION_VERSION
from nl2code.data.split import RATIOS, SPLIT_POLICY, SPLITS, assign_splits

MANIFEST_VERSION = "v1"


def _jsonl(rows: list[dict[str, str]]) -> bytes:
    return b"".join(stable_json(row) + b"\n" for row in rows)


def prepare(source: Path, output: Path, seed: int = 42) -> dict:
    """Validate, group, split, and write byte-stable files to output."""
    source = Path(source)
    output = Path(output)
    records, counts = load_records(source)
    source_sha256 = sha256(source.read_bytes())
    group_counts = assign_splits(records, seed)
    canonical_rows = [{key: record[key] for key in FIELDS} for record in records]
    canonical_sha256 = sha256(_jsonl(canonical_rows))
    row_counts = {split: sum(record["split"] == split for record in records) for split in SPLITS}
    files = {}
    for split in SPLITS:
        rows = [
            {key: record[key] for key in (*FIELDS, "normalized_intent", "record_id", "group_id")}
            for record in records
            if record["split"] == split
        ]
        files[f"{split}.jsonl"] = _jsonl(rows)
    files["splits.jsonl"] = _jsonl(
        [{key: record[key] for key in ("record_id", "group_id", "split")} for record in records]
    )
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "split_policy": SPLIT_POLICY,
        "seed": seed,
        "requested_split_ratios": RATIOS,
        "source_sha256": source_sha256,
        "canonical_sha256": canonical_sha256,
        "raw_row_count": counts["raw_rows"],
        "canonical_row_count": counts["canonical_rows"],
        "duplicate_rows_removed": counts["duplicates_removed"],
        "leakage_group_count": sum(group_counts.values()),
        "row_counts": row_counts,
        "group_counts": group_counts,
        "file_sha256": {name: sha256(content) for name, content in files.items()},
    }
    files["manifest.json"] = stable_json(manifest) + b"\n"
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (output / name).write_bytes(content)
    return manifest
