# Data provenance and M0 policy

## Tracked data and known history

`Data/mbpp_conala.csv` is the M0 input. `Data/mbpp(formatted).csv` and `Data/3.7K_python_dataset.csv` are also tracked university-era inputs; M0 does not prepare them. The first two have the four-column MBPP/CoNaLa-style schema; the 3.7K file has different `Problem` and `Python Code` columns.

The historical `Vineeth-branch` file `data_preproc.py` shows a CoNaLa load from `neulab/conala` (`curated`), an MBPP load from a local `data.json`, renaming MBPP fields to `question_id`, `intent`, `rewritten_intent`, and `snippet`, and concatenation into `mbpp_conala.csv`. This explains the apparent combination, but the exact source revisions, every transformation applied to the tracked bytes, redistribution permissions, and licensing of those upstream data are not established by M0. M0 does not invent missing provenance or add a license. The 600k retrieval corpus is present only on `Vineeth-branch` and is not moved to `main`.

## v1 contract and normalization

The CSV must have exactly `question_id`, `intent`, `rewritten_intent`, and `snippet` columns. IDs remain strings. Empty IDs, intents, and snippets are invalid. Empty `rewritten_intent` is permitted because the tracked dataset contains such rows.

- `question_id`: Unicode NFKC, then trim surrounding whitespace.
- `intent` and `rewritten_intent`: CRLF/CR to LF, NFKC, then trim surrounding whitespace; internal wording and spacing are preserved.
- `snippet`: CRLF/CR to LF only; code characters and indentation are preserved. Whitespace-only code is invalid.
- `normalized_intent`: from canonical intent, NFKC and newline normalization, `casefold()`, collapse every Unicode whitespace run to one ASCII space, then trim. Punctuation and semantics are untouched.

Each canonical four-field record receives a SHA-256 `record_id` from sorted-key UTF-8 JSON with compact separators. Exact canonical duplicates are removed; records with only a shared ID or intent remain distinct. The canonical dataset fingerprint hashes sorted canonical records serialized as UTF-8 JSONL with LF endings. The source fingerprint hashes raw CSV bytes, so changing row order or line endings can change it without changing canonical membership. The manifest also hashes each generated JSONL file. Artifacts contain no timestamp or absolute path.

## Leakage-safe split policy

`connected-components-hash-v1` connects rows sharing a canonical `question_id` or `normalized_intent`. Connectivity is transitive: if A shares an ID with B and B shares an intent with C, all three are one group. A group ID hashes a sorted, tagged list of its IDs and normalized intents. A SHA-256 bucket over policy name, seed (default 42), and group ID assigns the complete group to train, validation, or test using 80/10/10 ranges. Source row order cannot affect membership. Actual row and group counts may differ from target ratios because groups cannot be broken. Preparation checks zero cross-split overlap for both leakage keys before writing.

## Historical audit findings, not new M0 results

Historical row-random splitting allowed repeated intents across training and test, so those evaluations are contaminated. A prior audit reported that 111 of 381 historical test rows had intents represented in training, and that the historical 600k retrieval corpus contained the exact intent and gold snippet for 188 of those 381 test rows. These are historical audit findings supplied for project context, **not** benchmark results reproduced by M0. Old reported model and retrieval numbers are not currently suitable as validated benchmark or résumé claims. M0 does not add the 600k corpus to `main` or run model evaluations.
