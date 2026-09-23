"""Trusted local checkpoint persistence and strict resume identity checks."""

import os
import random
import tempfile
from pathlib import Path

import torch

SCHEMA_VERSION = "seq2seq-checkpoint-v1"


def capture_rng(shuffle_generator, teacher_generator):
    return {
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
        "shuffle_generator_state": shuffle_generator.get_state(),
        "teacher_generator_state": teacher_generator.get_state(),
    }


def restore_rng(checkpoint, shuffle_generator, teacher_generator):
    random.setstate(checkpoint["python_random_state"])
    torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
    if checkpoint["torch_cuda_rng_states"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_states"])
    shuffle_generator.set_state(checkpoint["shuffle_generator_state"])
    teacher_generator.set_state(checkpoint["teacher_generator_state"])


def save_checkpoint(path: Path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: Path, expected):
    # Only load trusted checkpoints created by this project: pickle can execute code.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"checkpoint {key} mismatch")
    return checkpoint
