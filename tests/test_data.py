"""Offline checks for the v1 data contract and split artifacts."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from nl2code.data.contract import DataError, load_records
from nl2code.data.normalize import normalize_intent
from nl2code.data.prepare import prepare

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("question_id", "intent", "rewritten_intent", "snippet")


def write_csv(path, rows, fields=FIELDS):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row(question_id, intent, snippet, rewritten_intent=""):
    return dict(
        question_id=question_id,
        intent=intent,
        rewritten_intent=rewritten_intent,
        snippet=snippet,
    )


SAMPLE = [
    row(" A ", "Use X", "x = 1\nprint(x)"),
    row("A", "Other task", "print(2)"),
    row("C", "  OTHER\tTASK  ", "print(3)"),
    row("D", "Separate", "print(4)"),
    row("D", "Separate", "print(4)"),
]


class DataTests(unittest.TestCase):
    def test_question_id_keeps_internal_carriage_return(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            write_csv(source, [row("  A\rB  ", "task", "pass")])
            records, _ = load_records(source)
            self.assertEqual(records[0]["question_id"], "A\rB")

    def test_schema_failures(self):
        for fields, rows, message in (
            (FIELDS[:-1], [dict(question_id="1", intent="x", rewritten_intent="")], "missing"),
            ((*FIELDS, "extra"), [dict(row("1", "x", "x"), extra="x")], "unexpected"),
            (FIELDS, [row("  ", "x", "x")], "question_id"),
            (FIELDS, [row("1", " \t ", "x")], "intent"),
            (FIELDS, [row("1", "x", " \n ")], "snippet"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "input.csv"
                write_csv(source, rows, fields)
                with self.assertRaisesRegex(DataError, message):
                    load_records(source)

    def test_normalization_and_code_preservation(self):
        self.assertEqual(normalize_intent("  Ａ\r\nB\t  C  "), "a b c")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            write_csv(source, [row(" ００１ ", "  Ａ  B\r\n", "  x = 'Ａ'\r\n    print(x)  ")])
            records, counts = load_records(source)
            self.assertEqual(counts["raw_rows"], 1)
            self.assertEqual(records[0]["question_id"], "001")
            self.assertEqual(records[0]["intent"], "A  B")
            self.assertEqual(records[0]["snippet"], "  x = 'Ａ'\n    print(x)  ")

    def test_transitive_groups_duplicates_and_disjointness(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "input.csv"
            write_csv(source, SAMPLE)
            manifest = prepare(source, base / "out", seed=42)
            self.assertEqual(manifest["raw_row_count"], 5)
            self.assertEqual(manifest["canonical_row_count"], 4)
            self.assertEqual(manifest["duplicate_rows_removed"], 1)
            self.assertEqual(manifest["leakage_group_count"], 2)
            split_rows = {
                split: [
                    json.loads(line)
                    for line in (base / "out" / f"{split}.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                for split in ("train", "validation", "test")
            }
            combined = [record for records in split_rows.values() for record in records]
            connected = [record for record in combined if record["question_id"] in {"A", "C"}]
            self.assertEqual(len(connected), 3)
            self.assertEqual(len({record["group_id"] for record in connected}), 1)
            self.assertEqual(
                len(
                    {
                        next(split for split, rows in split_rows.items() if record in rows)
                        for record in connected
                    }
                ),
                1,
            )
            for key in ("question_id", "normalized_intent"):
                for left, right in (
                    ("train", "validation"),
                    ("train", "test"),
                    ("validation", "test"),
                ):
                    self.assertFalse(
                        {r[key] for r in split_rows[left]} & {r[key] for r in split_rows[right]}
                    )

    def test_byte_determinism_and_row_order_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "one.csv"
            reversed_source = base / "reversed.csv"
            write_csv(source, SAMPLE)
            write_csv(reversed_source, list(reversed(SAMPLE)))
            for name, path in (
                ("first", source),
                ("second", source),
                ("reversed", reversed_source),
            ):
                prepare(path, base / name)
            files = (
                "train.jsonl",
                "validation.jsonl",
                "test.jsonl",
                "splits.jsonl",
                "manifest.json",
            )
            for name in files:
                self.assertEqual(
                    (base / "first" / name).read_bytes(), (base / "second" / name).read_bytes()
                )
                if name != "manifest.json":
                    self.assertEqual(
                        (base / "first" / name).read_bytes(),
                        (base / "reversed" / name).read_bytes(),
                    )
            first = json.loads((base / "first" / "manifest.json").read_text(encoding="utf-8"))
            reversed_manifest = json.loads(
                (base / "reversed" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first["canonical_sha256"], reversed_manifest["canonical_sha256"])
            self.assertNotEqual(first["source_sha256"], reversed_manifest["source_sha256"])
            self.assertEqual(
                {key: value for key, value in first.items() if key != "source_sha256"},
                {key: value for key, value in reversed_manifest.items() if key != "source_sha256"},
            )
            for name, digest in first["file_sha256"].items():
                self.assertEqual(
                    digest, hashlib.sha256((base / "first" / name).read_bytes()).hexdigest()
                )

    def test_real_tracked_dataset_has_no_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = prepare(ROOT / "Data" / "mbpp_conala.csv", output)
            self.assertEqual(sum(manifest["row_counts"].values()), manifest["canonical_row_count"])
            split_rows = {
                split: [
                    json.loads(line)
                    for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                for split in ("train", "validation", "test")
            }
            for key in ("question_id", "normalized_intent"):
                keys = [{record[key] for record in split_rows[split]} for split in split_rows]
                self.assertFalse(keys[0] & keys[1] | keys[0] & keys[2] | keys[1] & keys[2])

    def test_imports_have_no_file_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            code = "\n".join(
                f"import {name}"
                for name in (
                    "nl2code",
                    "nl2code.__main__",
                    "nl2code.cli",
                    "nl2code.data.contract",
                    "nl2code.data.normalize",
                    "nl2code.data.split",
                    "nl2code.data.prepare",
                )
            )
            subprocess.run([sys.executable, "-B", "-c", code], cwd=directory, check=True)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "input.csv"
            write_csv(source, SAMPLE)
            commands = (
                ("--help",),
                ("data", "validate", "--input", str(source)),
                ("data", "prepare", "--input", str(source), "--output", str(base / "out")),
            )
            for args in commands:
                with self.subTest(args=args):
                    result = subprocess.run(
                        [sys.executable, "-m", "nl2code", *args], capture_output=True, text=True
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((base / "out" / "manifest.json").is_file())
            write_csv(source, [row("", "x", "x")])
            result = subprocess.run(
                [sys.executable, "-m", "nl2code", "data", "validate", "--input", str(source)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("question_id", result.stderr)


if __name__ == "__main__":
    unittest.main()
