"""Stable source word-regex and target Python-character tokenization."""

import re

from nl2code.data.normalize import normalize_intent

SOURCE_VERSION = "source-word-regex-v1"
TARGET_VERSION = "python-char-v1"
SOURCE_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def source_tokens(text):
    return SOURCE_PATTERN.findall(normalize_intent(text))


def encode_source(text, vocab, max_tokens):
    tokens = source_tokens(text)
    lookup = {token: index for index, token in enumerate(vocab)}
    ids = [lookup.get(token, 1) for token in tokens[:max_tokens]] or [1]
    return ids, {"truncated": max(0, len(tokens) - max_tokens), "oov": ids.count(1)}


def encode_target(code, vocab, max_tokens):
    if max_tokens < 2:
        raise ValueError("max target tokens must reserve SOS and EOS")
    lookup = {token: index for index, token in enumerate(vocab)}
    chars = list(code)
    ids = [2] + [lookup.get(char, 1) for char in chars[: max_tokens - 2]] + [3]
    return ids, {"truncated": max(0, len(chars) - (max_tokens - 2)), "oov": ids.count(1)}


def decode_target(ids, vocab):
    chars = []
    for index in ids:
        if index == 3:
            break
        if index in (0, 2):
            continue
        chars.append(vocab[index] if index != 1 else "\ufffd")
    return "".join(chars)
