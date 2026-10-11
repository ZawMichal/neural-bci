# Neural BCI Suite: Project Specification

## Scope

Neural BCI Suite is a reproducible software prototype for experimenting with
text-conditioned synthetic neural signals and neural decoding. It has two
cooperating components in one repository:

1. **Encoder** — converts text into deterministic, multi-channel synthetic
   signals and renders waterfall visualizations.
2. **Decoder** — trains a character-level CTC model to convert those signals
   back into text.

This project does **not** use human neural recordings and does not claim to
simulate the biological activity of a real brain, cortical implant, or
Neuralink device. Its signal design is an engineering abstraction intended
for reproducible software experiments.

## Signal contract

Each encoded sample is stored as a compressed NumPy `.npz` file containing:

```text
signal      float32 array with shape (T, 1024), values in [0, 1]
sample_rate 50
channels    1024
transcript  normalized source sentence
seed        integer provenance value
config_json encoder configuration
```

The 1,024 channels are arranged conceptually as a 32 × 32 grid. The encoder
assigns stable spatial fingerprints to characters, applies smooth temporal
activation envelopes, and adds deterministic nuisance variation:

- Gaussian noise
- low-frequency baseline drift
- small amplitude variation
- a deterministic set of inactive channels

The contract is reproducible:

```text
signal = encode(text, configuration, seed)
```

The same normalized text, configuration, and seed produce the same signal.
Changing the seed changes nuisance variation while preserving the character
encoding.

## Encoder

The encoder package is located in
[`src/neural_cortex_encoder`](./src/neural_cortex_encoder). Its command-line
interface supports:

- encoding one sentence
- creating the SQLite label catalog
- exporting one signal per catalog sentence
- encoding custom UTF-8 label files
- generating waterfall PNG visualizations

The included catalog contains 30,000 unique synthetic sentences across
communication-oriented categories. The templates are a controlled benchmark,
not a complete or naturally distributed English corpus.

## Decoder

The decoder package is located in
[`src/neural_cortex_decoder`](./src/neural_cortex_decoder). It uses:

- lazy loading of `.npz` samples
- a learned spatial projection
- a bidirectional GRU
- character-level CTC loss
- greedy CTC decoding
- CER and WER evaluation
- checkpoint save/load
- manual text-to-signal-to-text inference

The decoder uses deterministic train, validation, and test partitions. Because
the current catalog contains related template variants, results on those
partitions can be optimistic. They are a baseline for this synthetic
benchmark, not a measure of real-world speech-decoding performance.

Text-to-speech is not currently part of the implemented pipeline. It can be
added after text decoding is evaluated independently.

## Data and repository policy

The small label database at [`data/labels.sqlite`](./data/labels.sqlite) is
included because it is useful metadata and is only a few megabytes.

The generated signal corpus can occupy approximately 33 GB and is ignored by
Git under `data/corpus/`. Trained checkpoints, logs, caches, and generated
visual assets are also ignored. These artifacts should be stored locally or
in dedicated dataset/model storage rather than in the source repository.

## Quick start

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
python -m pytest -q
```

Encode a sample:

```bash
neural-encode encode \
  "the decoder analyzes a neural signal" \
  --output data/example.npz \
  --plot assets/example.png
```

Train a small decoder smoke test after generating or obtaining
`data/corpus/`:

```bash
neural-decode train \
  --data-dir data/corpus \
  --checkpoint checkpoints/decoder.pt \
  --epochs 1 \
  --max-samples 256
```

See [`README.md`](./README.md) for an overview and
[`COMMANDS.md`](./COMMANDS.md) for the complete command reference.
