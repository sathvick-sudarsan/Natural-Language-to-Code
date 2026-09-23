"""Train and select by validation token NLL, without opening the test split."""

import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from nl2code.data.contract import stable_json
from nl2code.seq2seq.checkpoint import (
    SCHEMA_VERSION,
    capture_rng,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)
from nl2code.seq2seq.dataset import collate, encode_rows, load_split
from nl2code.seq2seq.model import Seq2Seq
from nl2code.seq2seq.runtime import (
    atomic_copy,
    atomic_write_bytes,
    code_revision,
    environment,
    read_config,
    sha_file,
    write_json,
)
from nl2code.seq2seq.vocabulary import build_vocabs, vocab_sha256


def run_epoch(model, loader, optimizer, teacher_generator, ratio, clip_norm, device):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    with torch.enable_grad() if training else torch.no_grad():
        for source, lengths, target in loader:
            source, lengths, target = source.to(device), lengths.to(device), target.to(device)
            scores = model(source, lengths, target, ratio if training else 0.0, teacher_generator)
            gold = target[:, 1:]
            token_count = gold.ne(0).sum().item()
            loss_sum = F.cross_entropy(
                scores.reshape(-1, scores.size(-1)),
                gold.reshape(-1),
                ignore_index=0,
                reduction="sum",
            )
            if training:
                optimizer.zero_grad()
                (loss_sum / token_count).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            total_loss += loss_sum.item()
            total_tokens += token_count
    if not total_tokens:
        raise ValueError("empty split or no target tokens")
    return total_loss / total_tokens, total_tokens


def _selection(state, best_hash):
    return {
        "selection_metric": "validation_token_nll",
        "selection_data": "validation only",
        "best_epoch": state["best_epoch"],
        "best_validation_token_nll": state["best_validation_token_nll"],
        "checkpoint": "best.pt",
        "checkpoint_sha256": best_hash,
    }


def _best_metadata(state, best_hash):
    return {
        "epoch": state["best_epoch"],
        "checkpoint": "best.pt",
        "checkpoint_sha256": best_hash,
    }


def _recover_run(run_dir, resume, checkpoint, identity):
    """Trust last.pt as the epoch commit marker; undo newer, unfinished sidecars."""
    completed = checkpoint["completed_epoch"]
    best_hash = checkpoint.get("best_checkpoint_sha256")
    best_path = run_dir / "best.pt"
    if not best_hash:
        if resume.resolve() == best_path.resolve() and checkpoint["best_epoch"] == completed:
            best_hash = sha_file(resume)
        else:
            selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
            if (
                selection.get("best_epoch") != checkpoint["best_epoch"]
                or selection.get("best_validation_token_nll")
                != checkpoint["best_validation_token_nll"]
                or not best_path.exists()
                or selection.get("checkpoint_sha256") != sha_file(best_path)
            ):
                raise ValueError("checkpoint has no consistent selected checkpoint identity")
            best_hash = selection["checkpoint_sha256"]
    if not best_path.exists() or sha_file(best_path) != best_hash:
        previous = run_dir / "best-previous.pt"
        if not previous.exists() or sha_file(previous) != best_hash:
            raise ValueError("selected checkpoint missing or SHA-256 mismatch")
        atomic_copy(previous, best_path)
    best = load_checkpoint(best_path, identity)
    if (
        best["completed_epoch"] != checkpoint["best_epoch"]
        or best["best_epoch"] != checkpoint["best_epoch"]
        or best["best_validation_token_nll"] != checkpoint["best_validation_token_nll"]
    ):
        raise ValueError("selected checkpoint epoch or score mismatch")
    for name, expected in (
        ("best-checkpoint.json", _best_metadata(checkpoint, best_hash)),
        ("selection.json", _selection(checkpoint, best_hash)),
    ):
        path = run_dir / name
        if not path.exists():
            raise ValueError(f"resume {name} missing")
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            epoch = actual.get("epoch", actual.get("best_epoch"))
            if type(epoch) is not int or epoch <= completed:
                raise ValueError(f"resume {name} inconsistent with checkpoint")
            write_json(path, expected)
    history_path = run_dir / "training-history.jsonl"
    lines = history_path.read_bytes().splitlines(keepends=True)
    if len(lines) < completed:
        raise ValueError("resume training history incomplete")
    history = [json.loads(line) for line in lines]
    if [row["epoch"] for row in history[:completed]] != list(range(1, completed + 1)) or history[
        completed - 1
    ]["global_step"] != checkpoint["global_step"]:
        raise ValueError("resume training history inconsistent with checkpoint")
    if len(lines) > completed:
        if any(row["epoch"] <= completed for row in history[completed:]):
            raise ValueError("resume training history has duplicate epochs")
        atomic_write_bytes(history_path, b"".join(lines[:completed]))
    return best_hash


def train(
    data_dir: Path, config_path: Path, run_dir: Path, resume: Path | None = None, device=None
):
    config, config_hash = read_config(config_path)
    train_manifest, manifest_hash, train_rows, train_hash = load_split(data_dir, "train")
    val_manifest, val_manifest_hash, val_rows, val_hash = load_split(data_dir, "validation")
    if manifest_hash != val_manifest_hash or train_manifest != val_manifest:
        raise ValueError("train and validation manifests differ")
    if not train_rows or not val_rows:
        raise ValueError("train and validation must be nonempty")
    vocabs = build_vocabs(train_rows, config["tokenization"]["source_min_frequency"])
    vocab_hash = vocab_sha256(vocabs)
    encoded_train, train_stats = encode_rows(train_rows, vocabs, config["tokenization"])
    encoded_val, val_stats = encode_rows(val_rows, vocabs, config["tokenization"])
    identity = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config_hash,
        "vocab_sha256": vocab_hash,
        "m0_manifest_sha256": manifest_hash,
        "canonical_dataset_sha256": train_manifest["canonical_sha256"],
        "train_identity": {"sha256": train_hash, "rows": len(train_rows)},
        "validation_identity": {"sha256": val_hash, "rows": len(val_rows)},
        "tokenizer_versions": {
            "source": config["tokenization"]["source"],
            "target": config["tokenization"]["target"],
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    config_output = run_dir / "config.json"
    vocab_output = run_dir / "vocab.json"
    if resume:
        if not config_output.exists() or not vocab_output.exists():
            raise ValueError("resume run metadata missing")
        if (
            json.loads(config_output.read_text(encoding="utf-8")) != config
            or json.loads(vocab_output.read_text(encoding="utf-8")) != vocabs
        ):
            raise ValueError("resume run config or vocabulary mismatch")
    elif config_output.exists() or (run_dir / "last.pt").exists():
        raise ValueError("run directory already contains training state; use --resume")
    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    shuffle_generator = torch.Generator().manual_seed(seed)
    teacher_generator = torch.Generator().manual_seed(seed + 1)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = Seq2Seq(len(vocabs["source"]), len(vocabs["target"]), config["model"]).to(
        selected_device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    first_epoch, global_step, best_nll, best_epoch = 1, 0, float("inf"), 0
    best_hash = None
    if resume:
        checkpoint = load_checkpoint(resume, identity)
        first_epoch = checkpoint["completed_epoch"] + 1
        if checkpoint["completed_epoch"] < 1 or checkpoint["epoch_boundary"] is not True:
            raise ValueError("resume requires a completed-epoch checkpoint")
        best_hash = _recover_run(run_dir, resume, checkpoint, identity)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(selected_device)
        global_step = checkpoint["global_step"]
        best_nll = checkpoint["best_validation_token_nll"]
        best_epoch = checkpoint["best_epoch"]
        restore_rng(checkpoint, shuffle_generator, teacher_generator)
    else:
        config_output.write_bytes(config_path.read_bytes())
        write_json(vocab_output, vocabs)
        write_json(
            run_dir / "environment.json",
            {
                **environment(),
                "preprocessing_stats": {"train": train_stats, "validation": val_stats},
            },
        )
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        encoded_train,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        encoded_val, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )
    for epoch in range(first_epoch, config["training"]["epochs"] + 1):
        train_nll, train_tokens = run_epoch(
            model,
            train_loader,
            optimizer,
            teacher_generator,
            config["training"]["teacher_forcing_ratio"],
            config["training"]["gradient_clip_norm"],
            selected_device,
        )
        global_step += len(train_loader)
        val_nll, val_tokens = run_epoch(
            model, val_loader, None, teacher_generator, 0, 0, selected_device
        )
        improved = val_nll < best_nll
        if improved:
            best_nll, best_epoch = val_nll, epoch
        state = {
            **identity,
            **capture_rng(shuffle_generator, teacher_generator),
            "epoch_boundary": True,
            "completed_epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "vocabularies": vocabs,
            "best_validation_token_nll": best_nll,
            "best_epoch": best_epoch,
            "best_checkpoint_sha256": best_hash,
            "code_revision": code_revision(),
        }
        if improved:
            if (run_dir / "best.pt").exists():
                atomic_copy(run_dir / "best.pt", run_dir / "best-previous.pt")
            state["best_checkpoint_sha256"] = None  # A checkpoint cannot contain its own SHA.
            save_checkpoint(run_dir / "best.pt", state)
            best_hash = sha_file(run_dir / "best.pt")
            write_json(
                run_dir / "best-checkpoint.json",
                _best_metadata(state, best_hash),
            )
        history_path = run_dir / "training-history.jsonl"
        history = history_path.read_bytes() if history_path.exists() else b""
        atomic_write_bytes(
            history_path,
            history
            + stable_json(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "training_token_nll": train_nll,
                    "validation_token_nll": val_nll,
                    "training_tokens": train_tokens,
                    "validation_tokens": val_tokens,
                }
            )
            + b"\n",
        )
        write_json(run_dir / "selection.json", _selection(state, best_hash))
        state["best_checkpoint_sha256"] = best_hash
        save_checkpoint(run_dir / "last.pt", state)
    return {"best_epoch": best_epoch, "best_validation_token_nll": best_nll}
