"""Compact character-level CTC decoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from .vocab import VOCAB


@dataclass(frozen=True)
class DecoderConfig:
    channels: int = 1024
    hidden_size: int = 128
    spatial_size: int = 128
    layers: int = 2
    vocabulary_size: int = len(VOCAB)


class NeuralSignalDecoder:
    """Wrapper around a PyTorch spatial encoder, BiGRU, and CTC head."""

    def __init__(self, config: DecoderConfig | None = None) -> None:
        try:
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("decoder requires PyTorch; install the [decoder] extra") from exc

        self.config = config or DecoderConfig()
        self.network = nn.Sequential(
            nn.Linear(self.config.channels, self.config.spatial_size),
            nn.LayerNorm(self.config.spatial_size),
            nn.GELU(),
        )
        self.recurrent = nn.GRU(
            input_size=self.config.spatial_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if self.config.layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(self.config.hidden_size * 2, self.config.vocabulary_size)

    def module(self):
        import torch.nn as nn

        class Network(nn.Module):
            def __init__(self, owner: NeuralSignalDecoder) -> None:
                super().__init__()
                self.spatial = owner.network
                self.recurrent = owner.recurrent
                self.classifier = owner.classifier

            def forward(self, signals):
                hidden = self.spatial(signals)
                sequence, _ = self.recurrent(hidden)
                return self.classifier(sequence).log_softmax(dim=-1)

        return Network(self)

    def config_json(self) -> str:
        return json.dumps(asdict(self.config), sort_keys=True)
