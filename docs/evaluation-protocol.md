# Prediction and evaluation protocol

Each prediction JSONL row has exactly `record_id`, `normalized_intent`, and
`prediction` string fields. The file has one row per M0 split record, sorted by
record ID. Its JSON sidecar records `prediction-v1`, model family, split, row
count, M0 manifest SHA-256, split-file SHA-256, config SHA-256, checkpoint
SHA-256, generation settings, and code revision. Validation rejects duplicate,
missing, or unexpected IDs, changed intents, data or metadata mismatches, and
different predictions for the same normalized intent.
All four identity digests use 64 lowercase hexadecimal SHA-256 characters.
This schema requires greedy generation with a positive integer `max_tokens`;
missing settings and unknown strategies are invalid.

Saved `evaluation-v1` results include the split, validated manifest and
canonical dataset identities, split and exact prediction-file SHA-256 hashes,
config and checkpoint identities, generation settings, prediction row count,
and a nested metrics payload. The prediction hash covers the exact JSONL bytes
read for scoring, including whitespace and line endings.

```bash
nl2code evaluate --data-dir artifacts/data/m0 --split validation --predictions artifacts/runs/seq2seq/validation.jsonl --metadata artifacts/runs/seq2seq/validation.meta.json --output artifacts/runs/seq2seq/validation-results.json
```

The primary task unit is `normalized_intent`. All unique snippet references
for each intent are collected. `intent_any_reference_exact_match` succeeds if
the one prediction equals any reference after CRLF/CR to LF normalization and
stripping only surrounding whitespace. Internal whitespace, case,
punctuation, formatting, and AST are untouched. The result gives matched
intents, total unique intents, and their ratio.

`python_syntax_validity` is measured once per unique intent: a non-whitespace
prediction must pass `ast.parse`. Code is **never executed**. The report gives
valid count, total unique intents, and ratio. `row_exact_match` compares each
prediction with that row's own reference using the same text normalization;
it is explicitly a secondary diagnostic. The report also counts records,
intent groups, groups with multiple references, maximum references in a group,
and empty predictions. No pass@k, BLEU, or ROUGE is computed.
