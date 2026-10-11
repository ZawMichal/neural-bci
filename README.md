# Neural BCI Suite

This repository contains a reproducible synthetic brain-computer-interface
prototype with an encoder and a character-level neural decoder.

> **Important:** This is a software simulation and machine-learning
> experiment. It does not use human neural recordings, model a real brain,
> reproduce Neuralink hardware, or provide a medical device or clinical
> result.

## Current status

- Deterministic text-to-signal encoding is implemented.
- Waterfall visualization is implemented.
- A 30,000-row SQLite sentence catalog is included.
- Character-level CTC decoding, training, evaluation, and manual inference are
  implemented.
- The generated signal corpus and trained checkpoints are intentionally kept
  out of Git because they are large derived artifacts. See
  [`.gitignore`](./.gitignore).
- Results on this synthetic dataset should not be interpreted as evidence of
  generalization to biological neural data.

This repository is the deterministic signal-generation half of the BCI suite.
It creates **synthetic training data**, not medically valid recordings or a
claim about the true cortical representation of speech.

## Reproducibility contract

The encoder is fully deterministic:

```text
signal = encode(text, config, seed)
```

The same normalized text, configuration, and seed produce exactly the same
`float32` array. The seed affects controlled nuisance variation (noise, drift
phase, amplitude, and dead channels); the character fingerprint itself is
derived from the character and remains stable.

Each character is encoded as three smooth Gaussian activity regions on a
32 x 32 electrode grid. Characters overlap in time using a raised-sine
activation envelope. The output then receives deterministic Gaussian noise,
low-frequency baseline drift, and a small deterministic set of dead channels.
This gives the future decoder structure to learn while retaining variation.

## Install and use

```bash
python -m pip install -e ".[all]"
neural-encode encode "the decoder analyzes a neural signal" \
  --output data/example.npz \
  --plot assets/example.png
```

The project uses `pyproject.toml` as its dependency manifest; a separate
`requirements.txt` is not required. The `all` extra installs the development
test runner and the optional Matplotlib visualizer. For a headless encoder
without plotting, install only `python -m pip install -e .`.

For a persistent command reference, see [COMMANDS.md](./COMMANDS.md).

Render waterfall previews from existing encoded samples:

```bash
neural-encode visualize \
  --input-dir data/corpus \
  --output-dir assets/waterfalls \
  --count 5
```

Generate the built-in 1,000-sentence starter corpus:

```bash
neural-encode labels --database data/labels.sqlite --count 1000
```

The repository includes a starter [SQLite label database](./data/labels.sqlite)
with 30,000 unique sentences across 31 communication-oriented categories. It
stores the sentence, category, character count, and token count. Categories
include greetings, requests, questions, status, commands, confirmations,
corrections, preferences, emotions, time, location, technical messages,
planning, workflow, feedback, and safety messages.

The templates are deliberately compositional rather than an attempt to
enumerate every possible English sentence. English has no finite list of all
possible sentences. The catalog instead covers communication intents and
reusable sentence structures; additional human-written labels can be imported
with the `corpus --labels` command.

To create the labels and immediately encode every row:

```bash
neural-encode labels \
  --database data/labels.sqlite \
  --count 30000 \
  --export-dir data/corpus \
  --seed 1000
```

The database can also be queried directly:

```bash
sqlite3 data/labels.sqlite \
  "SELECT category, COUNT(*) FROM labels GROUP BY category ORDER BY category;"
```

The current corpus export creates one signal sample per label. For example,
with `--seed 1000`, the first samples use seeds 1000, 1001, 1002, and so on.
This is one deterministic seed per sentence, not 1,000 recordings per
sentence. Re-running an export into the same directory overwrites the same
`sample_*.npz` paths rather than appending duplicates. Repeated recordings per
sentence will be added as a separate corpus-generation option.

For additional hand-written labels, put one sentence per line in a UTF-8 text
file and run:

```bash
neural-encode corpus --labels labels.txt --output-dir data/corpus --seed 1000
```

Every `.npz` sample contains:

* `signal`: `float32`, shape `(T, 1024)`, range `[0, 1]`
* `sample_rate`: `50`
* `channels`: `1024`
* `transcript`: normalized source sentence
* `seed` and `config_json`: provenance for exact reproduction

The format is intentionally simple so the decoder can load it without
depending on the encoder package.

## Next decoder contract

The decoder should initially predict characters, not whole words. This matches
the compositional encoder and allows it to decode sentences not present in the
training label list, provided their characters are represented in training.
CER and WER should be reported separately. TTS belongs after text decoding and
must never be treated as part of the signal model.

## Decoder

The decoder is a separate package in this workspace. It uses a lazy NPZ
loader, a spatial projection, bidirectional GRU, and character-level CTC
decoding. It does not load the approximately 33 GB generated corpus into RAM.

Install the decoder dependencies:

```bash
source .venv/bin/activate
python -m pip install -e ".[all]"
```

Run a small smoke-test training job first:

```bash
neural-decode train \
  --data-dir data/corpus \
  --checkpoint checkpoints/decoder.pt \
  --epochs 1 \
  --max-samples 256
```

Evaluate a checkpoint:

```bash
neural-decode evaluate \
  --data-dir data/corpus \
  --checkpoint checkpoints/decoder.pt \
  --split test \
  --max-samples 256
```

Manual end-to-end inference encodes the typed text into a temporary NPZ,
passes it through the model, prints the decoded text, and deletes the
temporary file:

```bash
neural-decode manual \
  --checkpoint checkpoints/decoder.pt \
  --seed 40000 \
  "the decoder analyzes a neural signal"
```

Omit the quoted sentence to enter it interactively at the terminal.

Manual inference overwrites [data/manual_waterfall.png](./data/manual_waterfall.png)
with a waterfall visualization of the newly encoded input sentence. Use
`--waterfall path/to/image.png` to choose another location.

Training reports the current epoch, batch progress, percentage, running loss,
and elapsed time. By default it prints at the first batch, every 50 batches,
and the final batch of each epoch. Use `--progress-every 10` for more frequent
updates.
