"""Deterministic sentence-label generation for large synthetic corpora."""

from __future__ import annotations

from pathlib import Path

from .encoder import NeuralSignalEncoder, save_sample


SUBJECTS = (
    "the scientist",
    "the engineer",
    "the decoder",
    "the system",
    "the sensor",
    "the model",
    "the operator",
    "the recorder",
)
VERBS = (
    "analyzes",
    "decodes",
    "records",
    "measures",
    "tracks",
    "detects",
    "encodes",
    "classifies",
)
OBJECTS = (
    "a neural signal",
    "a quiet channel",
    "the cortical pattern",
    "a new sentence",
    "the activity map",
    "the rising activity",
    "a stable waveform",
    "the signal window",
)
ADVERBS = (
    "carefully",
    "in real time",
    "with low noise",
    "with stable timing",
    "reliably",
    "during calibration",
    "for the next sample",
    "without overflow",
)


def default_sentences(count: int) -> list[str]:
    """Return up to ``count`` deterministic, grammatical training labels."""
    if count < 1:
        raise ValueError("count must be positive")
    labels: list[str] = []
    for subject in SUBJECTS:
        for verb in VERBS:
            for object_ in OBJECTS:
                for adverb in ADVERBS:
                    labels.append(f"{subject} {verb} {object_} {adverb}.")
    return labels[:count]


def read_labels(path: str | Path) -> list[str]:
    labels = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label and not label.startswith("#")]
    if not labels:
        raise ValueError("label file contains no usable sentences")
    return labels


def generate_corpus(
    encoder: NeuralSignalEncoder,
    labels: list[str],
    output_dir: str | Path,
    *,
    seed: int = 0,
) -> int:
    """Generate one NPZ per label; seeds are stable across ordering changes."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for index, label in enumerate(labels):
        sample = encoder.encode(label, seed=seed + index)
        save_sample(sample, str(destination / f"sample_{index:06d}.npz"))
    return len(labels)
