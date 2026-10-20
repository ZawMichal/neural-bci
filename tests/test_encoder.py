from pathlib import Path

import numpy as np

from neural_cortex_encoder import EncoderConfig, NeuralSignalEncoder
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


def test_channel_noise_is_visible_in_channel_baselines() -> None:
    config = EncoderConfig(
        background_level=0.0,
        noise_std=0.0,
        channel_noise_std=0.05,
        drift_amplitude=0.0,
    )
    signal = NeuralSignalEncoder(config).encode("a", seed=1).signal
    channel_means = signal.mean(axis=0)
    assert channel_means.std() > 0.02


def test_character_code_is_compositional_and_case_insensitive() -> None:
    encoder = NeuralSignalEncoder()
    lower = encoder.encode("a", seed=0)
    upper = encoder.encode("A", seed=0)
    assert np.array_equal(lower.signal, upper.signal)


def test_linguistic_context_changes_signal_without_changing_channel_count() -> None:
    encoder = NeuralSignalEncoder()
    this = encoder.encode("this", seed=0)
    tyis = encoder.encode("tyis", seed=0)
    assert this.signal.shape[1] == tyis.signal.shape[1] == 1024
    assert this.signal.shape == tyis.signal.shape
    assert not np.array_equal(this.signal, tyis.signal)


def test_encoder_groups_channels_into_deterministic_thread_abstraction() -> None:
    config = EncoderConfig(channels=16, threads=4)
    first = NeuralSignalEncoder(config).encode("test", seed=4)
    second = NeuralSignalEncoder(config).encode("test", seed=4)
    assert first.signal.shape[1] == 16
    assert np.array_equal(first.signal, second.signal)


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
