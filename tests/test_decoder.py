from pathlib import Path

import numpy as np

from neural_cortex_decoder.data import collate_samples, split_paths
from neural_cortex_decoder.metrics import edit_distance, error_rate
from neural_cortex_decoder.vocab import CHAR_TO_ID, decode_ids, encode_text


def test_text_vocabulary_round_trip() -> None:
    text = "hello, world!"
    encoded = encode_text(text)
    ctc_path = []
    for token in encoded:
        if ctc_path and ctc_path[-1] == token:
            ctc_path.append(CHAR_TO_ID["<blank>"])
        ctc_path.append(token)
    assert decode_ids(ctc_path) == text


def test_edit_distance_and_error_rate() -> None:
    assert edit_distance("cat", "cut") == 1
    assert error_rate(["hello"], ["hello"]) == 0.0
    assert error_rate(["hello"], ["hallo"]) == 0.2


def test_split_is_disjoint_and_complete() -> None:
    paths = [Path(f"sample_{index:06d}.npz") for index in range(100)]
    train = set(split_paths(paths, "train"))
    validation = set(split_paths(paths, "validation"))
    test = set(split_paths(paths, "test"))
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert len(train | validation | test) == 100


def test_collate_pads_variable_length_signals() -> None:
    batch = collate_samples(
        [
            {"signal": np.zeros((4, 1024), dtype=np.float32), "target": np.array([1, 2]), "transcript": "ab", "path": "a"},
            {"signal": np.zeros((6, 1024), dtype=np.float32), "target": np.array([1]), "transcript": "a", "path": "b"},
        ]
    )
    assert tuple(batch["signals"].shape) == (2, 6, 1024)
    assert batch["input_lengths"].tolist() == [4, 6]
