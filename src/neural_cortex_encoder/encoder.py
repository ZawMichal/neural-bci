"""Mathematically deterministic text-to-signal encoding.

This is a synthetic encoding scheme, not a claim about how human speech is
represented in cortex. It is designed to be reproducible, structured, and
useful for training a decoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EncoderConfig:
    """Signal and timing parameters shared with the future decoder."""

    channels: int = 1024
    sample_rate: int = 50
    frames_per_character: int = 8
    overlap_frames: int = 2
    background_level: float = 0.025
    noise_std: float = 0.012
    drift_amplitude: float = 0.018
    active_peak: float = 0.78
    spatial_sigma: float = 1.7
    dead_channel_fraction: float = 0.01

    def __post_init__(self) -> None:
        side = math.isqrt(self.channels)
        if side * side != self.channels:
            raise ValueError("channels must be a perfect square, such as 1024")
        if self.sample_rate <= 0 or self.frames_per_character < 3:
            raise ValueError("sample_rate must be positive and timing must have >= 3 frames")
        if not 0 <= self.overlap_frames < self.frames_per_character:
            raise ValueError("overlap_frames must be in [0, frames_per_character)")
        if not 0 <= self.dead_channel_fraction < 1:
            raise ValueError("dead_channel_fraction must be in [0, 1)")


@dataclass(frozen=True)
class EncodedSample:
    """A signal and the metadata required to reproduce or decode it."""

    signal: np.ndarray
    transcript: str
    seed: int
    sample_rate: int
    channels: int
    config_json: str


def _seed_from(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


class NeuralSignalEncoder:
    """Encode text using deterministic spatial fingerprints and temporal envelopes."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        self.config = config or EncoderConfig()
        side = math.isqrt(self.config.channels)
        coordinates = np.arange(side, dtype=np.float32)
        yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
        self._x = xx.reshape(-1)
        self._y = yy.reshape(-1)

    def encode(self, text: str, *, seed: int = 0) -> EncodedSample:
        """Return a reproducible float32 signal with shape ``(T, channels)``."""
        transcript = " ".join(text.strip().split()).casefold()
        if not transcript:
            raise ValueError("text must contain at least one non-whitespace character")

        characters = list(transcript)
        step = self.config.frames_per_character - self.config.overlap_frames
        frame_count = step * (len(characters) - 1) + self.config.frames_per_character
        signal = np.full((frame_count, self.config.channels), self.config.background_level, dtype=np.float64)

        for index, character in enumerate(characters):
            start = index * step
            fingerprint = self._fingerprint(character)
            envelope = self._temporal_envelope(self.config.frames_per_character)
            amplitude = 0.94 + 0.06 * self._unit_value(seed, "amplitude", index)
            for local_frame, value in enumerate(envelope):
                signal[start + local_frame] += fingerprint * (value * amplitude)

        time = np.arange(frame_count, dtype=np.float64) / self.config.sample_rate
        drift_phase = 2.0 * math.pi * self._unit_value(seed, "drift-phase")
        drift = self.config.drift_amplitude * np.sin(
            2.0 * math.pi * 0.35 * time + drift_phase
        )[:, None]
        signal += drift

        rng = np.random.Generator(np.random.PCG64(_seed_from("noise", seed, transcript)))
        signal += rng.normal(0.0, self.config.noise_std, signal.shape)
        dead_count = int(round(self.config.channels * self.config.dead_channel_fraction))
        if dead_count:
            dead_rng = np.random.Generator(np.random.PCG64(_seed_from("dead", seed, transcript)))
            dead_channels = dead_rng.choice(self.config.channels, dead_count, replace=False)
            signal[:, dead_channels] *= 0.08

        signal = np.clip(signal, 0.0, 1.0).astype(np.float32)
        return EncodedSample(
            signal=signal,
            transcript=transcript,
            seed=seed,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            config_json=json.dumps(asdict(self.config), sort_keys=True),
        )

    def _fingerprint(self, character: str) -> np.ndarray:
        """Create a smooth, repeatable 2-D spatial code for one character."""
        normalized = character.casefold()
        points = []
        for bump in range(3):
            phase = 2.0 * math.pi * self._unit_value(normalized, "x", bump)
            radius = 2.0 + 9.0 * self._unit_value(normalized, "radius", bump)
            center_x = (self.config.channels ** 0.5 - 1.0) * (
                0.5 + 0.45 * math.sin(phase + bump)
            )
            center_y = (self.config.channels ** 0.5 - 1.0) * (
                0.5 + 0.45 * math.cos(phase + radius)
            )
            points.append((center_x, center_y, 1.0 + self._unit_value(normalized, "weight", bump)))

        fingerprint = np.zeros(self.config.channels, dtype=np.float64)
        for center_x, center_y, weight in points:
            distance_squared = (self._x - center_x) ** 2 + (self._y - center_y) ** 2
            fingerprint += weight * np.exp(-distance_squared / (2.0 * self.config.spatial_sigma**2))
        peak = float(fingerprint.max())
        if peak == 0.0:
            raise RuntimeError("spatial fingerprint unexpectedly contains no energy")
        return fingerprint * (self.config.active_peak / peak)

    @staticmethod
    def _temporal_envelope(length: int) -> np.ndarray:
        phase = np.linspace(0.0, math.pi, length, dtype=np.float64)
        return np.sin(phase) ** 1.35

    @staticmethod
    def _unit_value(*parts: object) -> float:
        return int.from_bytes(
            hashlib.blake2b(
                "\x1f".join(str(part) for part in parts).encode("utf-8"),
                digest_size=8,
            ).digest(),
            "little",
        ) / float(2**64)


def save_sample(sample: EncodedSample, path: str) -> None:
    """Save one sample in the decoder-facing NPZ exchange format."""
    np.savez_compressed(
        path,
        signal=sample.signal,
        sample_rate=np.int32(sample.sample_rate),
        channels=np.int32(sample.channels),
        transcript=np.asarray(sample.transcript),
        seed=np.uint64(sample.seed),
        config_json=np.asarray(sample.config_json),
    )


def load_sample(path: str) -> dict[str, Any]:
    """Load a sample while preserving scalar metadata as Python values."""
    with np.load(path, allow_pickle=False) as data:
        return {
            "signal": data["signal"],
            "sample_rate": int(data["sample_rate"]),
            "channels": int(data["channels"]),
            "transcript": str(data["transcript"]),
            "seed": int(data["seed"]),
            "config_json": str(data["config_json"]),
        }
