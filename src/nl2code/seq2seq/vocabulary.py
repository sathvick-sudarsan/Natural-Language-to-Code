"""Train-only deterministic vocabularies."""

from collections import Counter

from nl2code.data.contract import sha256, stable_json
from nl2code.seq2seq.tokenization import source_tokens

SOURCE_SPECIALS = ["<PAD>", "<UNK>"]
TARGET_SPECIALS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]


def build_vocabs(rows, min_frequency):
    source = Counter(token for row in rows for token in source_tokens(row["normalized_intent"]))
    target = {char for row in rows for char in row["snippet"]}
    return {
        "source": SOURCE_SPECIALS
        + [
            token
            for token, count in sorted(source.items(), key=lambda item: (-item[1], item[0]))
            if count >= min_frequency and token not in SOURCE_SPECIALS
        ],
        "target": TARGET_SPECIALS + sorted(target - set(TARGET_SPECIALS)),
    }


def vocab_sha256(vocabs):
    return sha256(stable_json(vocabs))
