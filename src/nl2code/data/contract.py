"""Read, validate, and deduplicate the v1 CSV without coercing IDs."""

import csv
import hashlib
import json
import unicodedata
from pathlib import Path

from nl2code.data.normalize import canonical_text, normalize_intent, normalize_newlines

SCHEMA_VERSION = "v1"
FIELDS = ("question_id", "intent", "rewritten_intent", "snippet")


class DataError(ValueError):
    """An input dataset violates the v1 contract."""


def stable_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_records(source: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Return sorted unique canonical records and raw/duplicate row counts."""
    unique: dict[bytes, dict[str, str]] = {}
    raw_rows = 0
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if header is None:
                raise DataError("CSV is empty; required columns are missing")
            missing = sorted(set(FIELDS) - set(header))
            unexpected = sorted(set(header) - set(FIELDS))
            if missing or unexpected or len(header) != len(FIELDS):
                duplicate = len(header) != len(set(header))
                raise DataError(
                    f"invalid columns: missing={missing}, unexpected={unexpected}, "
                    f"duplicates={duplicate}"
                )
            for row_number, values in enumerate(reader, start=2):
                raw_rows += 1
                if len(values) != len(FIELDS):
                    raise DataError(
                        f"row {row_number}: expected {len(FIELDS)} fields, got {len(values)}"
                    )
                original = dict(zip(header, values, strict=True))
                record = {
                    "question_id": unicodedata.normalize("NFKC", original["question_id"]).strip(),
                    "intent": canonical_text(original["intent"]),
                    "rewritten_intent": canonical_text(original["rewritten_intent"]),
                    "snippet": normalize_newlines(original["snippet"]),
                }
                for required in ("question_id", "intent", "snippet"):
                    if not record[required].strip():
                        raise DataError(f"row {row_number}: blank {required}")
                canonical = stable_json(record)
                record["normalized_intent"] = normalize_intent(record["intent"])
                record["record_id"] = sha256(canonical)
                unique[canonical] = record
    except (OSError, UnicodeError, csv.Error) as error:
        raise DataError(f"cannot read {source}: {error}") from error
    records = sorted(unique.values(), key=lambda record: record["record_id"])
    return records, {
        "raw_rows": raw_rows,
        "canonical_rows": len(records),
        "duplicates_removed": raw_rows - len(records),
    }
