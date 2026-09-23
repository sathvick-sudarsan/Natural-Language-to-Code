import json
from pathlib import Path

import pytest
import torch

from nl2code.data.contract import sha256, stable_json
from nl2code.seq2seq.checkpoint import load_checkpoint, save_checkpoint
from nl2code.seq2seq.model import Seq2Seq
from nl2code.seq2seq.predict import infer, predict
from nl2code.seq2seq.runtime import sha_file
from nl2code.seq2seq.tokenization import decode_target, encode_target, source_tokens
from nl2code.seq2seq.train import train
from nl2code.seq2seq.vocabulary import build_vocabs

CONFIG = {
    "model": {
        "encoder_embedding_dim": 8,
        "decoder_embedding_dim": 8,
        "encoder_hidden_dim": 8,
        "decoder_hidden_dim": 8,
        "dropout": 0.0,
    },
    "tokenization": {"max_source_tokens": 8, "max_target_tokens": 8},
}


def test_deterministic_vocab_and_code_characters():
    rows = [
        {"normalized_intent": "Foo bar", "snippet": "A = 1\n"},
        {"normalized_intent": "bar Foo", "snippet": "a=2"},
    ]
    vocabs = build_vocabs(rows, 1)
    assert vocabs == build_vocabs(list(reversed(rows)), 1)
    assert vocabs["source"][:4] == ["<PAD>", "<UNK>", "bar", "foo"]
    assert source_tokens("Foo, bar!") == ["foo", ",", "bar", "!"]
    ids, stats = encode_target("A = 1\n", vocabs["target"], 8)
    assert decode_target(ids, vocabs["target"]) == "A = 1\n"
    assert stats["truncated"] == 0


def test_forward_attention_mask_and_greedy_path():
    model = Seq2Seq(7, 9, CONFIG["model"])
    src = torch.tensor([[2, 3, 0], [2, 3, 4]])
    lengths = torch.tensor([2, 3])
    target = torch.tensor([[2, 4, 3], [2, 5, 3]])
    scores = model(src, lengths, target, 1.0, torch.Generator().manual_seed(4))
    assert scores.shape == (2, 2, 9)
    outputs, hidden = model.encoder(src, lengths)
    weights = model.decoder.attention(hidden, outputs, src.ne(0))
    assert weights[0, 2].item() == 0
    assert len(model.generate(src, lengths, 3)) == 2


def test_checkpoint_rejects_identity_mismatch(tmp_path):
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        {
            "schema_version": "seq2seq-checkpoint-v1",
            "config_sha256": "a",
            "m0_manifest_sha256": "m",
            "vocab_sha256": "v",
        },
    )
    assert load_checkpoint(path, {"config_sha256": "a"})["config_sha256"] == "a"
    for key in ("config_sha256", "m0_manifest_sha256", "vocab_sha256"):
        with pytest.raises(ValueError):
            load_checkpoint(path, {key: "wrong"})


def _synthetic_data(path):
    path.mkdir()
    files = {}
    for split, labels in (("train", ("a", "b", "c", "d")), ("validation", ("e", "f"))):
        rows = [
            {
                "record_id": label,
                "normalized_intent": f"make {label}",
                "snippet": "x=1\n" if label in "ace" else "x=2\n",
            }
            for label in labels
        ]
        files[f"{split}.jsonl"] = b"".join(stable_json(row) + b"\n" for row in rows)
    for name, content in files.items():
        (path / name).write_bytes(content)
    (path / "manifest.json").write_bytes(
        stable_json(
            {
                "manifest_version": "v1",
                "schema_version": "v1",
                "canonical_sha256": "synthetic",
                "row_counts": {"train": 4, "validation": 2},
                "file_sha256": {name: sha256(content) for name, content in files.items()},
            }
        )
        + b"\n"
    )


def _same_state(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _same_state(a, b)
    else:
        assert left == right


def test_cpu_epoch_resume_equals_uninterrupted_and_needs_no_test_split(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _synthetic_data(data_dir)
    config = json.loads(Path("configs/seq2seq/m1-baseline.json").read_text())
    config["model"] = CONFIG["model"]
    config["training"]["batch_size"] = 2
    config["training"]["epochs"] = 2
    config["generation"]["max_tokens"] = 8
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    full = tmp_path / "full"
    train(data_dir, config_path, full, device="cpu")

    from nl2code.seq2seq import train as training_module

    interrupted = tmp_path / "interrupted"
    original_save = training_module.save_checkpoint

    def stop_after_first_epoch(path, state):
        if path.name == "last.pt" and state["completed_epoch"] == 2:
            raise RuntimeError("simulated interruption")
        original_save(path, state)

    with monkeypatch.context() as patch:
        patch.setattr(training_module, "save_checkpoint", stop_after_first_epoch)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train(data_dir, config_path, interrupted, device="cpu")
    train(data_dir, config_path, interrupted, interrupted / "last.pt", device="cpu")
    a = load_checkpoint(full / "last.pt", {})
    b = load_checkpoint(interrupted / "last.pt", {})
    for key in ("model_state_dict", "optimizer_state_dict", "completed_epoch", "global_step"):
        _same_state(a[key], b[key])
    assert not (data_dir / "test.jsonl").exists()
    assert isinstance(infer(full / "best.pt", "Make A"), str)
    metadata = predict(
        data_dir,
        full / "best.pt",
        "validation",
        tmp_path / "predictions.jsonl",
        tmp_path / "predictions.meta.json",
    )
    assert metadata["record_count"] == 2
    with pytest.raises(ValueError, match="selection"):
        predict(
            data_dir,
            full / "best.pt",
            "test",
            tmp_path / "test-predictions.jsonl",
            tmp_path / "test-predictions.meta.json",
        )


@pytest.mark.parametrize("boundary", ["before_last", "after_last"])
def test_epoch_commit_interruption_keeps_selected_checkpoint(tmp_path, monkeypatch, boundary):
    data_dir = tmp_path / "data"
    _synthetic_data(data_dir)
    config = json.loads(Path("configs/seq2seq/m1-baseline.json").read_text())
    config["model"] = CONFIG["model"]
    config["training"]["batch_size"] = 2
    config["training"]["epochs"] = 3
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    run_dir = tmp_path / "run"

    from nl2code.seq2seq import train as training_module

    original_save = training_module.save_checkpoint

    def interrupt(path, state):
        at_boundary = path.name == "last.pt" and state["completed_epoch"] == 2
        if at_boundary and boundary == "before_last":
            raise RuntimeError("simulated interruption")
        original_save(path, state)
        if at_boundary and boundary == "after_last":
            raise RuntimeError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(training_module, "save_checkpoint", interrupt)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train(data_dir, config_path, run_dir, device="cpu")
    committed = load_checkpoint(run_dir / "last.pt", {})
    assert committed["completed_epoch"] == (1 if boundary == "before_last" else 2)
    if boundary == "after_last":
        selected = json.loads((run_dir / "selection.json").read_text())
        assert selected["best_epoch"] == committed["best_epoch"]
        assert selected["checkpoint_sha256"] == sha_file(run_dir / "best.pt")
    train(data_dir, config_path, run_dir, run_dir / "last.pt", device="cpu")
    last = load_checkpoint(run_dir / "last.pt", {})
    selection = json.loads((run_dir / "selection.json").read_text())
    best = load_checkpoint(run_dir / "best.pt", {})
    assert last["completed_epoch"] == 3
    assert selection["best_epoch"] == last["best_epoch"] == best["completed_epoch"]
    assert selection["checkpoint_sha256"] == sha_file(run_dir / "best.pt")
    assert selection["best_validation_token_nll"] == last["best_validation_token_nll"]
