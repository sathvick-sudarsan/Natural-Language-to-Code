"""Deterministic greedy inference and M0 prediction files."""

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from nl2code.data.contract import sha256, stable_json
from nl2code.evaluation.predictions import validate_predictions, write_predictions
from nl2code.seq2seq.checkpoint import load_checkpoint
from nl2code.seq2seq.dataset import load_split
from nl2code.seq2seq.model import Seq2Seq
from nl2code.seq2seq.runtime import code_revision, sha_file, write_json
from nl2code.seq2seq.tokenization import decode_target, encode_source
from nl2code.seq2seq.vocabulary import vocab_sha256


def load_model(checkpoint_path: Path):
    checkpoint = load_checkpoint(checkpoint_path, {})
    config = checkpoint["config"]
    vocabs = checkpoint["vocabularies"]
    if (
        sha256(stable_json(config)) != checkpoint["config_sha256"]
        or vocab_sha256(vocabs) != checkpoint["vocab_sha256"]
    ):
        raise ValueError("checkpoint config or vocabulary corrupted")
    model = Seq2Seq(len(vocabs["source"]), len(vocabs["target"]), config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def infer(checkpoint_path: Path, text: str):
    model, checkpoint = load_model(checkpoint_path)
    config = checkpoint["config"]
    vocabs = checkpoint["vocabularies"]
    source, _ = encode_source(text, vocabs["source"], config["tokenization"]["max_source_tokens"])
    source = torch.tensor([source], dtype=torch.long)
    ids = model.generate(source, torch.tensor([source.size(1)]), config["generation"]["max_tokens"])
    return decode_target(ids[0], vocabs["target"])


def predict(
    data_dir: Path,
    checkpoint_path: Path,
    split: str,
    output: Path,
    metadata_output: Path,
    selection: Path | None = None,
):
    checkpoint_hash = sha_file(checkpoint_path)
    if split == "test":
        if selection is None:
            raise ValueError("test prediction requires --selection")
        selected = json.loads(selection.read_text(encoding="utf-8"))
        if (
            selected.get("checkpoint_sha256") != checkpoint_hash
            or selected.get("selection_metric") != "validation_token_nll"
        ):
            raise ValueError("test checkpoint does not match validation selection")
    manifest, manifest_hash, rows, split_hash = load_split(data_dir, split)
    model, checkpoint = load_model(checkpoint_path)
    if (
        manifest_hash != checkpoint["m0_manifest_sha256"]
        or manifest["canonical_sha256"] != checkpoint["canonical_dataset_sha256"]
    ):
        raise ValueError("checkpoint and M0 dataset mismatch")
    identity_key = {"train": "train_identity", "validation": "validation_identity"}.get(split)
    if identity_key and checkpoint[identity_key] != {"sha256": split_hash, "rows": len(rows)}:
        raise ValueError("checkpoint split identity mismatch")
    config = checkpoint["config"]
    vocabs = checkpoint["vocabularies"]
    predictions = []
    batch_size = config["training"]["batch_size"]
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        sources = [
            torch.tensor(
                encode_source(
                    row["normalized_intent"],
                    vocabs["source"],
                    config["tokenization"]["max_source_tokens"],
                )[0],
                dtype=torch.long,
            )
            for row in batch
        ]
        padded = pad_sequence(sources, batch_first=True)
        lengths = torch.tensor([len(source) for source in sources])
        generated = model.generate(padded, lengths, config["generation"]["max_tokens"])
        for row, ids in zip(batch, generated, strict=True):
            predictions.append(
                {
                    "record_id": row["record_id"],
                    "normalized_intent": row["normalized_intent"],
                    "prediction": decode_target(ids, vocabs["target"]),
                }
            )
    metadata = {
        "schema_version": "prediction-v1",
        "model_family": "seq2seq",
        "split": split,
        "record_count": len(rows),
        "data_manifest_sha256": manifest_hash,
        "split_file_sha256": split_hash,
        "config_sha256": checkpoint["config_sha256"],
        "checkpoint_sha256": checkpoint_hash,
        "generation_settings": config["generation"],
        "code_revision": code_revision(),
    }
    validate_predictions(rows, predictions, metadata, manifest_hash, split_hash, split)
    write_predictions(output, predictions)
    write_json(metadata_output, metadata)
    return metadata
