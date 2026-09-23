# M1A attention Seq2Seq infrastructure

This canonical implementation adapts the architecture in
`legacy/academic/baseline_seq2seq.py`; the academic files remain preserved and
are never imported. It retains source embedding, a bidirectional GRU encoder,
additive attention, a GRU decoder, the context/hidden/embedding output
projection, teacher forcing, and gradient clipping. The encoder packs source
sequences using their true lengths, and attention masks padding.

The historical script downloaded NLTK data and initialized data, vocabularies,
training, and files at import time. The canonical modules do none of those.
They also avoid module-level vocabulary globals. Python targets are Unicode
characters, so code case, punctuation, indentation, and internal whitespace
are retained. Source intents use M0 normalization, casefolding, and the
`\w+|[^\w\s]` regex. Ad-hoc text receives the same M0 intent normalization.
The source vocabulary has PAD/UNK followed by train tokens sorted by descending
frequency and token. The target has PAD/UNK/SOS/EOS followed by sorted train
characters. Both vocabularies are train-only. Target limits reserve SOS/EOS;
the run environment records truncation and OOV counts. Generated UNK is
rendered as the Unicode replacement character; PAD and SOS are never rendered.

Install Python 3.12 and `python -m pip install -e ".[dev,seq2seq]"`. The
`seq2seq` extra pins PyTorch 2.7.0. Prepare M0 data first:

```bash
nl2code data prepare --input Data/mbpp_conala.csv --output artifacts/data/m0 --seed 42
nl2code seq2seq train --data-dir artifacts/data/m0 --config configs/seq2seq/m1-baseline.json --run-dir artifacts/runs/seq2seq/m1-baseline-seed42
nl2code seq2seq train --data-dir artifacts/data/m0 --config configs/seq2seq/m1-baseline.json --run-dir artifacts/runs/seq2seq/m1-baseline-seed42 --resume artifacts/runs/seq2seq/m1-baseline-seed42/last.pt
nl2code seq2seq infer --checkpoint artifacts/runs/seq2seq/m1-baseline-seed42/best.pt --text "sort a list in reverse order"
nl2code seq2seq predict --data-dir artifacts/data/m0 --checkpoint artifacts/runs/seq2seq/m1-baseline-seed42/best.pt --split validation --output artifacts/runs/seq2seq/validation.jsonl --metadata-output artifacts/runs/seq2seq/validation.meta.json
```

Training verifies M0 manifest and train/validation file hashes; it never opens
the test split. Adam uses the JSON config settings. Training and validation
NLL sum over non-PAD target tokens and divide by token count. Only minimum
validation token NLL selects `best.pt`. The ignored run directory contains the
exact config, vocabulary, environment and preprocessing counts, JSONL history,
`last.pt`, `best.pt`, `best-checkpoint.json`, and `selection.json`.
Epoch files are atomically replaced. A newer best checkpoint and its metadata
are saved before `last.pt`, which commits the epoch. An ignored
`best-previous.pt` preserves the prior selection during that commit. Resume
checks the selected checkpoint hash, epoch, score, and sidecars against the
committed `last.pt`; it rolls back unfinished newer sidecars and history.

An epoch-boundary checkpoint stores model, optimizer, configuration,
vocabularies, M0 and split identities, best selection, step/epoch, and Python,
PyTorch, shuffle, teacher-forcing, and available CUDA RNG states. Resume
rejects schema, data, config, or vocabulary changes. Checkpoints use Python
pickle: **load only trusted local checkpoints produced by this project**.
CPU continuation is tested for exact state equivalence. Different hardware,
CUDA kernels, and library versions can still produce different numerical
results; the stored environment identifies the run context.

For a later test prediction, pass `--split test --selection <selection.json>`.
The selection file must name the exact checkpoint SHA-256 before generation.
M1A does not run a real M0 test benchmark. See
[evaluation protocol](evaluation-protocol.md) for prediction and scoring rules.
There is no generated-code execution, pass@k, beam search, BLEU, or ROUGE.
