from pathlib import Path

import numpy as np

from neural_cortex_encoder import NeuralSignalEncoder
from neural_cortex_encoder.corpus import default_sentences
from neural_cortex_encoder.encoder import load_sample, save_sample
from neural_cortex_encoder.label_db import create_label_database, read_database_labels


def test_same_text_and_seed_are_exactly_repeatable() -> None:
    encoder = NeuralSignalEncoder()
    first = encoder.encode("hello world", seed=17)
    second = encoder.encode("hello world", seed=17)
    assert np.array_equal(first.signal, second.signal)
    assert first.signal.dtype == np.float32
    assert first.signal.shape[1] == 1024


def test_seed_changes_nuisance_signal_but_not_shape() -> None:
    encoder = NeuralSignalEncoder()
    first = encoder.encode("hello world", seed=1)
    second = encoder.encode("hello world", seed=2)
    assert first.signal.shape == second.signal.shape
    assert not np.array_equal(first.signal, second.signal)


def test_character_code_is_compositional_and_case_insensitive() -> None:
    encoder = NeuralSignalEncoder()
    lower = encoder.encode("a", seed=0)
    upper = encoder.encode("A", seed=0)
    assert np.array_equal(lower.signal, upper.signal)


def test_npz_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    sample = NeuralSignalEncoder().encode("test", seed=4)
    save_sample(sample, str(path))
    loaded = load_sample(str(path))
    assert np.array_equal(loaded["signal"], sample.signal)
    assert loaded["transcript"] == "test"
    assert loaded["seed"] == 4


def test_builtin_label_factory_supports_thousands_of_labels() -> None:
    labels = default_sentences(1000)
    assert len(labels) == 1000
    assert len(set(labels)) == 1000


def test_label_database_is_deduplicated_and_queryable(tmp_path: Path) -> None:
    path = tmp_path / "labels.sqlite"
    count = create_label_database(path)
    labels = read_database_labels(path)
    assert count == len(labels)
    assert count > 30000
    assert len(labels) == len(set(labels))
    assert "hello." in labels
