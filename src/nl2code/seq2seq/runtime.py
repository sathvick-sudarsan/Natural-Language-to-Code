"""Run environment and stable identity helpers."""

import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from nl2code.data.contract import sha256, stable_json


def code_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def sha_file(path: Path):
    return sha256(path.read_bytes())


def atomic_write_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value):
    atomic_write_bytes(path, stable_json(value) + b"\n")


def read_config(path: Path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "seq2seq-config-v1":
        raise ValueError("incompatible Seq2Seq config")
    if (
        config["tokenization"]["source"] != "source-word-regex-v1"
        or config["tokenization"]["target"] != "python-char-v1"
    ):
        raise ValueError("incompatible tokenizer version")
    if (
        config["training"]["optimizer"] != "adam"
        or config["training"]["selection_metric"] != "validation_token_nll"
    ):
        raise ValueError("unsupported training protocol")
    if config["training"]["num_workers"] != 0 or config["generation"]["strategy"] != "greedy":
        raise ValueError("unsupported worker or generation strategy")
    if (
        config["tokenization"]["max_source_tokens"] < 1
        or config["tokenization"]["max_target_tokens"] < 2
    ):
        raise ValueError("invalid token limit")
    return config, sha256(stable_json(config))


def environment():
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "code_revision": code_revision(),
    }
