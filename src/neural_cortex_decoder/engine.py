"""Training, evaluation, checkpointing, and manual inference."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from neural_cortex_encoder.encoder import NeuralSignalEncoder, save_sample
from .data import NPZSignalDataset, collate_samples, sample_paths, split_paths
from .metrics import error_rate, greedy_decode
from .model import DecoderConfig, NeuralSignalDecoder


def _torch_components():
    import torch
    from torch.utils.data import DataLoader

    return torch, DataLoader


def _loader(dataset: NPZSignalDataset, batch_size: int, shuffle: bool):
    _, DataLoader = _torch_components()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_samples)


def evaluate_model(model, loader, device) -> dict[str, float]:
    torch, _ = _torch_components()
    model.eval()
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    total_loss = 0.0
    predictions: list[str] = []
    targets: list[str] = []
    with torch.no_grad():
        for batch in loader:
            signals = batch["signals"].to(device)
            logits = model(signals)
            log_probs = logits.transpose(0, 1)
            input_lengths = batch["input_lengths"].to(device)
            loss = criterion(
                log_probs,
                batch["targets"].to(device),
                input_lengths,
                batch["target_lengths"].to(device),
            )
            total_loss += float(loss.item()) * len(batch["transcripts"])
            predictions.extend(greedy_decode(logits, batch["input_lengths"]))
            targets.extend(batch["transcripts"])
    count = len(targets)
    return {
        "loss": total_loss / max(1, count),
        "cer": error_rate(predictions, targets),
        "wer": error_rate(predictions, targets, words=True),
        "samples": float(count),
    }


def train(
    data_dir: str | Path,
    checkpoint: str | Path,
    *,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    max_samples: int | None = None,
    device_name: str = "auto",
    progress_every: int = 50,
) -> dict[str, Any]:
    torch, _ = _torch_components()
    all_paths = sample_paths(data_dir)
    train_data = NPZSignalDataset(split_paths(all_paths, "train"), max_samples)
    validation_data = NPZSignalDataset(split_paths(all_paths, "validation"), max_samples)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else device_name if device_name != "auto" else "cpu"
    )
    decoder = NeuralSignalDecoder()
    model = decoder.module().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    train_loader = _loader(train_data, batch_size, shuffle=True)
    validation_loader = _loader(validation_data, batch_size, shuffle=False)
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        epoch_started = time.perf_counter()
        total_batches = len(train_loader)
        print(f"[epoch {epoch}/{epochs}] starting: {total_batches} batches", flush=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["signals"].to(device))
            loss = criterion(
                logits.transpose(0, 1),
                batch["targets"].to(device),
                batch["input_lengths"].to(device),
                batch["target_lengths"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_size_actual = len(batch["transcripts"])
            total_loss += float(loss.item()) * batch_size_actual
            seen += batch_size_actual
            if batch_index == 1 or batch_index % progress_every == 0 or batch_index == total_batches:
                elapsed = time.perf_counter() - epoch_started
                percentage = 100.0 * batch_index / max(1, total_batches)
                running_loss = total_loss / max(1, seen)
                print(
                    f"[epoch {epoch}/{epochs}] "
                    f"batch {batch_index}/{total_batches} ({percentage:5.1f}%) "
                    f"loss={running_loss:.4f} elapsed={elapsed:.0f}s",
                    flush=True,
                )
        validation = evaluate_model(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, seen),
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record))
    destination = Path(checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(decoder.config),
            "history": history,
        },
        destination,
    )
    return history[-1]


def evaluate(
    data_dir: str | Path,
    checkpoint: str | Path,
    *,
    split: str = "test",
    batch_size: int = 8,
    max_samples: int | None = None,
    device_name: str = "auto",
) -> dict[str, float]:
    torch, _ = _torch_components()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    decoder = NeuralSignalDecoder(DecoderConfig(**payload["config"]))
    model = decoder.module()
    model.load_state_dict(payload["state_dict"])
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                           else device_name if device_name != "auto" else "cpu")
    model.to(device)
    paths = NPZSignalDataset(split_paths(sample_paths(data_dir), split), max_samples)
    metrics = evaluate_model(model, _loader(paths, batch_size, shuffle=False), device)
    print(json.dumps(metrics, indent=2))
    return metrics


def decode_file(checkpoint: str | Path, sample_path: str | Path, *, device_name: str = "auto") -> str:
    torch, _ = _torch_components()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    decoder = NeuralSignalDecoder(DecoderConfig(**payload["config"]))
    model = decoder.module()
    model.load_state_dict(payload["state_dict"])
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                           else device_name if device_name != "auto" else "cpu")
    model.to(device).eval()
    dataset = NPZSignalDataset([Path(sample_path)])
    batch = collate_samples([dataset[0]])
    with torch.no_grad():
        logits = model(batch["signals"].to(device))
    return greedy_decode(logits, batch["input_lengths"])[0]


def manual_decode(
    checkpoint: str | Path,
    text: str,
    *,
    seed: int = 0,
    device_name: str = "auto",
    waterfall_path: str | Path | None = None,
) -> str:
    """Encode text to a temporary NPZ and decode it through the trained model."""
    encoder = NeuralSignalEncoder()
    sample = encoder.encode(text, seed=seed)
    if waterfall_path is not None:
        from neural_cortex_encoder.visualizer import save_waterfall

        destination = Path(waterfall_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_waterfall(sample.signal, destination, title=f"Input: {sample.transcript}")
    with tempfile.TemporaryDirectory(prefix="neural-decode-") as directory:
        path = Path(directory) / "manual.npz"
        save_sample(sample, str(path))
        return decode_file(checkpoint, path, device_name=device_name)
