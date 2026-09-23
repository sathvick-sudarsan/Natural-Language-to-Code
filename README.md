# Natural-Language-to-Code

This repository began as a university project translating natural-language requests into Python code. The assignment called for an attention-based Seq2Seq model as the simpler baseline and a transformer/CodeT5 approach as the advanced model. Those experiments are preserved as historical work in [legacy/academic](legacy/academic/README.md) and on the original team branches.

**Current status:** M0 provides a reproducible, leakage-safe data foundation. M1A adds local Seq2Seq training and evaluation infrastructure, but no measured M1 test result or trained artifact. Historical numerical results are not currently reproducible or resume-safe and should not be cited as validated performance.

## Install

Python **3.12 only** (`>=3.12,<3.13`) is supported for M0. From the repository root:
Check that `python --version` reports 3.12 before creating the environment. On Windows, `py -3.12 -m venv .venv` selects 3.12 explicitly when another version is the default; on Linux, use `python3.12 -m venv .venv` if needed.

```bash
python -m venv .venv
# Activate .venv using your shell: for PowerShell, .venv\Scripts\Activate.ps1;
# for bash/zsh, source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Validate and prepare data

The tracked `Data/mbpp_conala.csv` is the v1 input. All preparation is local and uses only the Python standard library.

```bash
nl2code data validate --input Data/mbpp_conala.csv
nl2code data prepare --input Data/mbpp_conala.csv --output artifacts/data/m0 --seed 42
```

`python -m nl2code` accepts the same commands. Preparation writes deterministic JSONL split and membership files plus a fingerprinted manifest. It assigns connected leakage groups as indivisible units, keeping shared question IDs and normalized intents within one split. See [data provenance and policy](docs/data-provenance.md) for the precise contract and limits. Generated `artifacts/` files are ignored by Git.

## Check the foundation

```bash
ruff format --check .
ruff check .
pytest
```

CI runs these checks, package imports, validation, and preparation on Python 3.12 for Windows and Ubuntu. No ML dependency or model download is needed after the development install.

## M1A Seq2Seq infrastructure

Install the separate `seq2seq` extra with `python -m pip install -e ".[dev,seq2seq]"`.
After M0 preparation, run:

```bash
nl2code seq2seq train --data-dir artifacts/data/m0 --config configs/seq2seq/m1-baseline.json --run-dir artifacts/runs/seq2seq/m1-baseline-seed42
nl2code seq2seq infer --checkpoint artifacts/runs/seq2seq/m1-baseline-seed42/best.pt --text "sort a list in reverse order"
nl2code seq2seq predict --data-dir artifacts/data/m0 --checkpoint artifacts/runs/seq2seq/m1-baseline-seed42/best.pt --split validation --output artifacts/runs/seq2seq/validation.jsonl --metadata-output artifacts/runs/seq2seq/validation.meta.json
nl2code evaluate --data-dir artifacts/data/m0 --split validation --predictions artifacts/runs/seq2seq/validation.jsonl --metadata artifacts/runs/seq2seq/validation.meta.json --output artifacts/runs/seq2seq/validation-results.json
```

See [Seq2Seq baseline](docs/seq2seq-baseline.md) for resume and trusted
checkpoint rules and [evaluation protocol](docs/evaluation-protocol.md) for
strict prediction files and multi-reference exact match. The CPU model checks
run separately with `pytest -q tests_seq2seq`.

## Layout

- `src/nl2code/`: canonical package and `nl2code` CLI.
- `Data/`: tracked university-era CSV inputs; the directory name is preserved.
- `tests/`: offline contract, determinism, leakage, import, and CLI checks.
- `docs/data-provenance.md`: source uncertainty and versioned data policy.
- `legacy/academic/`: preserved academic code, notebook, and old requirements.
- `.github/workflows/ci.yml`: CPU-only validation on Windows and Ubuntu.

The historical 600k retrieval corpus remains on `Vineeth-branch` and is not included on `main`.
