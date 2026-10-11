"""Lazy NPZ dataset and deterministic train/validation/test splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from neural_cortex_encoder.encoder import load_sample
from .vocab import encode_text


def sample_paths(root: str | Path) -> list[Path]:
    paths = sorted(Path(root).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no NPZ samples found in {root}")
    return paths


def split_paths(paths: list[Path], split: str) -> list[Path]:
    if split not in {"train", "validation", "test", "all"}:
        raise ValueError("split must be train, validation, test, or all")
    if split == "all":
        return paths
    remainder = {"train": 0, "validation": 8, "test": 9}[split]
    return [path for index, path in enumerate(paths) if index % 10 == remainder
            or (split == "train" and index % 10 in range(8))]


class NPZSignalDataset:
    """Torch-compatible dataset without importing torch at module import time."""

    def __init__(self, paths: list[Path], max_samples: int | None = None) -> None:
        self.paths = paths[:max_samples] if max_samples else paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = load_sample(str(self.paths[index]))
        signal = np.asarray(sample["signal"], dtype=np.float32)
        target = np.asarray(encode_text(sample["transcript"]), dtype=np.int64)
        return {
            "signal": signal,
            "target": target,
            "transcript": sample["transcript"],
            "path": str(self.paths[index]),
        }


def collate_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    max_frames = max(item["signal"].shape[0] for item in batch)
    channels = batch[0]["signal"].shape[1]
    signals = torch.zeros((len(batch), max_frames, channels), dtype=torch.float32)
    input_lengths = []
    targets = []
    target_lengths = []
    for index, item in enumerate(batch):
        frames = item["signal"].shape[0]
        signals[index, :frames] = torch.from_numpy(item["signal"])
        input_lengths.append(frames)
        targets.append(torch.from_numpy(item["target"]))
        target_lengths.append(len(item["target"]))
    return {
        "signals": signals,
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        "targets": torch.cat(targets),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "transcripts": [item["transcript"] for item in batch],
        "paths": [item["path"] for item in batch],
    }
