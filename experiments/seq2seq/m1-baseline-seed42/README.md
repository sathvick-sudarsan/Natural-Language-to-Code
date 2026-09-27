# M1 measured attention-Seq2Seq baseline (`m1-baseline-seed42`)

This is the first measured attention-Seq2Seq baseline for this repository. The
architecture is adapted from the preserved university baseline
(`legacy/academic/baseline_seq2seq.py`); see
[Seq2Seq baseline](../../../docs/seq2seq-baseline.md). Scoring follows the
[evaluation protocol](../../../docs/evaluation-protocol.md).

## Test results

| Metric | Numerator | Denominator | Value |
|---|---:|---:|---:|
| Intent-level any-reference exact match (primary) | 0 | 322 | 0.0 |
| Python syntax validity (primary) | 50 | 322 | 0.15527950310559005 |
| Row-level exact match (secondary diagnostic) | 0 | 399 | 0.0 |

| Count | Value |
|---|---:|
| Test records | 399 |
| Unique normalized intents | 322 |
| Multi-reference intent groups | 45 |
| Maximum references for one intent | 6 |
| Empty predictions | 0 |

The values above are copied from [`results.test.json`](results.test.json),
produced by `nl2code evaluate`. The 399 test records form 316 M0 leakage
groups; one leakage group can contain several distinct normalized intents.

## Method

- Data: M0's leakage-safe deterministic split (`connected-components-hash-v1`,
  seed 42) of `Data/mbpp_conala.csv`: 3,006 train, 399 validation, and 399 test
  records out of 3,804 canonical records.
- Exactly one run was performed, using the committed
  `configs/seq2seq/m1-baseline.json` unmodified: seed 42, bidirectional GRU
  encoder and GRU decoder with additive attention, embeddings 256, hidden 512,
  dropout 0.5, `source-word-regex-v1` source tokens (max 128),
  `python-char-v1` target characters (max 512), batch 32, 10 epochs, Adam with
  learning rate 0.001, teacher forcing 0.5, gradient clipping 1.0. There was no
  hyperparameter tuning, no additional seed, and no rerun.
- Training read only `train.jsonl`, `validation.jsonl`, and `manifest.json`.
  All 10 epochs completed without interruption or resume.
- The checkpoint with the minimum validation token NLL was selected (epoch 3,
  validation token NLL 3.7058126119462447). The test split was first accessed
  only after training and selection finished, to generate predictions from the
  selected `best.pt` using greedy decoding with at most 512 tokens. The
  evaluator was run once.

| Epoch | Training token NLL | Validation token NLL |
|---:|---:|---:|
| 1 | 3.1317 | 3.7574 |
| 2 | 2.7904 | 3.8242 |
| 3 | 2.6326 | **3.7058** (selected) |
| 4 | 2.4960 | 3.8363 |
| 5 | 2.4129 | 3.8529 |
| 6 | 2.3133 | 3.8828 |
| 7 | 2.2478 | 4.0545 |
| 8 | 2.1997 | 4.0316 |
| 9 | 2.1468 | 4.0166 |
| 10 | 2.0795 | 3.9774 |

These values are rounded from [`training-history.jsonl`](training-history.jsonl).

## Metric definitions

- **Intent-level any-reference exact match** is the primary metric. The unit
  is the unique `normalized_intent`. The single deterministic prediction for an
  intent is compared with every unique reference for that intent. It matches
  only when the texts are equal after CRLF/CR-to-LF newline normalization and
  stripping outer whitespace.
- **Python syntax validity** counts, per unique intent, predictions that are
  non-empty after stripping and that `ast.parse` accepts.
- **Row-level exact match** compares each row's prediction with that row's own
  reference, using the same normalization. It is a secondary diagnostic.

Generated programs were not executed. The dataset contains no established
executable test suite, so no functional correctness or pass@k claim is made.
BLEU and ROUGE were not computed.

## Environment and provenance

| Item | Value |
|---|---|
| Code revision | `c6fa082e4ad487035eff4af9376a271209891268` |
| Branch | `feat/m1b-measured-seq2seq`; worktree clean at experiment start |
| Python | 3.12.7 (CPython, 64-bit) |
| Operating system | Windows-11-10.0.26200-SP0 |
| PyTorch | 2.7.0+cpu (from `pip install -e ".[dev,seq2seq]"`) |
| Execution device | CPU, Intel Core Ultra 9 285H, 16 logical processors |
| CUDA available | false (CPU-only PyTorch build; no CUDA version applies) |
| GPU | None used. The host has an NVIDIA GeForce RTX 5070 Ti Laptop GPU, which the CPU-only build did not use. |
| Seed | 42 |
| Training wall clock | 2026-09-26T23:14:30Z to 2026-09-27T00:24:38Z |
| Test prediction wall clock | 2026-09-27T00:25:06Z to 2026-09-27T00:25:32Z |
| M0 manifest SHA-256 | `596cf2478e41dcd6954d6cd76533c97f81eed6616b9d635f48828c35c2970ebc` |
| Canonical dataset SHA-256 | `1817fa2215fa940438cefcdec548a91dc9f69dbcac1779418cb991faf962103f` |
| Train split SHA-256 | `27069c30b3fbe6c509afb063baadc20f13f8699e234d3282db27d3e2b94ec795` |
| Validation split SHA-256 | `ad3e69de5e99e82641633a9177b362eac6811b5432f82ba26a7797990b7abc6a` |
| Test split SHA-256 | `a3059d44ef8b4fa1669f990bba30fa5358a8216845051af2f99902f32319ab83` |
| Config SHA-256 | `04baeb19794b51d54e9dd88ef97755c278eae5fc1d7215cdfbeae3802bd7e2b8` |
| Vocabulary SHA-256 | `3c9096f36ae8f3257a23f5f48d5b81c09a2dcf187c340844786a2b85b096c41b` |
| Selected checkpoint SHA-256 | `26528846772cde09183555c6c5f6702e8028b1c33bd979c5bbe78f468c276308` |
| Generation settings | greedy, `max_tokens` 512 |
| Test prediction file SHA-256 | `fe690e1b2508e7c1d1dcbea39dd9d59acea3c95ea6117fedb71a98810d8004f9` |

The config and vocabulary identities are the tooling's SHA-256 values over
sorted-key compact JSON, not over file bytes. The Python, OS, PyTorch, CUDA
availability, and code revision values also appear in
[`environment.json`](environment.json). The implementation seeds its random
number generators, but bitwise-identical results on different hardware, thread
counts, or library versions are not claimed.

## Files

Machine-readable files are byte copies of the run outputs:
`environment.json`, `vocab.json`, `training-history.jsonl`, `selection.json`,
`best-checkpoint.json`, `predictions.test.jsonl`,
`predictions.test.meta.json`, and `results.test.json` come from
`artifacts/runs/seq2seq/m1-baseline-seed42/`, and `data-manifest.json` is
`artifacts/data/m0/manifest.json`. `config.json` is the committed
`configs/seq2seq/m1-baseline.json`. Its parsed content equals the run copy,
which differed only in Windows CRLF line endings. `.gitattributes` disables
line-ending conversion so the committed bytes keep their recorded SHA-256
values. Checkpoints (`best.pt`, `last.pt`, `best-previous.pt`) are not
committed.

## Reproduce

```bash
nl2code data prepare --input Data/mbpp_conala.csv --output artifacts/data/m0 --seed 42
nl2code seq2seq train --data-dir artifacts/data/m0 --config configs/seq2seq/m1-baseline.json --run-dir artifacts/runs/seq2seq/m1-baseline-seed42
nl2code seq2seq predict --data-dir artifacts/data/m0 --checkpoint artifacts/runs/seq2seq/m1-baseline-seed42/best.pt --selection artifacts/runs/seq2seq/m1-baseline-seed42/selection.json --split test --output artifacts/runs/seq2seq/m1-baseline-seed42/predictions.test.jsonl --metadata-output artifacts/runs/seq2seq/m1-baseline-seed42/predictions.test.meta.json
nl2code evaluate --data-dir artifacts/data/m0 --split test --predictions artifacts/runs/seq2seq/m1-baseline-seed42/predictions.test.jsonl --metadata artifacts/runs/seq2seq/m1-baseline-seed42/predictions.test.meta.json --output artifacts/runs/seq2seq/m1-baseline-seed42/results.test.json
```
