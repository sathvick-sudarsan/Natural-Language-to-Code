"""Verified M0 split loading and small batches."""

import json
from pathlib import Path

from nl2code.data.contract import sha256
from nl2code.seq2seq.tokenization import encode_source, encode_target


def load_split(data_dir: Path, split: str):
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unknown split: {split}")
    manifest_bytes = (data_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("manifest_version") != "v1" or manifest.get("schema_version") != "v1":
        raise ValueError("incompatible M0 manifest")
    name = f"{split}.jsonl"
    split_bytes = (data_dir / name).read_bytes()
    split_hash = sha256(split_bytes)
    if split_hash != manifest["file_sha256"][name]:
        raise ValueError(f"M0 {name} SHA-256 mismatch")
    rows = [json.loads(line) for line in split_bytes.splitlines()]
    if len(rows) != manifest["row_counts"][split]:
        raise ValueError(f"M0 {name} row count mismatch")
    if [row["record_id"] for row in rows] != sorted(row["record_id"] for row in rows):
        raise ValueError(f"M0 {name} is not sorted")
    return manifest, sha256(manifest_bytes), rows, split_hash


def encode_rows(rows, vocabs, tokenization):
    encoded = []
    stats = {"source_truncated": 0, "source_oov": 0, "target_truncated": 0, "target_oov": 0}
    for row in rows:
        source, source_stats = encode_source(
            row["normalized_intent"], vocabs["source"], tokenization["max_source_tokens"]
        )
        target, target_stats = encode_target(
            row["snippet"], vocabs["target"], tokenization["max_target_tokens"]
        )
        for prefix, values in (("source", source_stats), ("target", target_stats)):
            for key, value in values.items():
                stats[f"{prefix}_{key}"] += value
        encoded.append((source, target))
    return encoded, stats


def collate(batch):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    sources = [torch.tensor(pair[0], dtype=torch.long) for pair in batch]
    targets = [torch.tensor(pair[1], dtype=torch.long) for pair in batch]
    return (
        pad_sequence(sources, batch_first=True),
        torch.tensor([len(source) for source in sources]),
        pad_sequence(targets, batch_first=True),
    )
