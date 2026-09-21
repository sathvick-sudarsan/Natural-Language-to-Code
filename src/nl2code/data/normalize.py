"""Stable v1 text normalization rules."""

import unicodedata

NORMALIZATION_VERSION = "v1"


def normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def canonical_text(value: str) -> str:
    return unicodedata.normalize("NFKC", normalize_newlines(value)).strip()


def normalize_intent(value: str) -> str:
    return " ".join(canonical_text(value).casefold().split())
