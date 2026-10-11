"""Stable character vocabulary shared by training and inference."""

from __future__ import annotations

import string


BLANK = "<blank>"
CHARACTERS = string.ascii_lowercase + " .,?!'-"
VOCAB = (BLANK,) + tuple(CHARACTERS)
CHAR_TO_ID = {character: index for index, character in enumerate(VOCAB)}
ID_TO_CHAR = dict(enumerate(VOCAB))


def encode_text(text: str) -> list[int]:
    normalized = " ".join(text.strip().split()).casefold()
    unknown = sorted(set(normalized) - set(CHAR_TO_ID))
    if unknown:
        raise ValueError(f"unsupported characters: {unknown}")
    return [CHAR_TO_ID[character] for character in normalized]


def decode_ids(ids: list[int]) -> str:
    output: list[str] = []
    previous = BLANK
    for token_id in ids:
        character = ID_TO_CHAR[token_id]
        if character != BLANK and character != previous:
            output.append(character)
        previous = character
    return "".join(output)
