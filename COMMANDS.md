# Encoder Commands

Run these commands from:

```bash
cd /home/michalzawitaj/bci-project
source .venv/bin/activate
```

This repository intentionally excludes the generated signal corpus,
checkpoints, caches, and generated images from Git. The commands below assume
that `data/corpus` has already been generated locally. The corpus is a derived
artifact and is not required to clone the source repository.

If the environment has not been installed yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
```

## Show help

```bash
neural-encode --help
```

## Encode one sentence

Save a signal as an NPZ file:

```bash
neural-encode encode \
  "the decoder analyzes a neural signal" \
  --output data/example.npz
```

Save a signal and a waterfall image:

```bash
neural-encode encode \
  "the decoder analyzes a neural signal" \
  --output data/example.npz \
  --plot assets/example.png
```

## Visualize existing encoded data

Render five representative samples from an existing corpus without
regenerating the signals:

```bash
neural-encode visualize \
  --input-dir data/corpus \
  --output-dir assets/waterfalls \
  --count 5
```

The resulting PNGs show channels as horizontal rows over time, using a
black-to-navy-to-blue-to-cyan-to-yellow activity palette.

Use a specific seed:

```bash
neural-encode encode \
  "hello world" \
  --seed 42 \
  --output data/hello-seed-42.npz
```

The same sentence, configuration, and seed produce the same signal exactly.

## Create or recreate the label database

Create the 30,000-label catalog:

```bash
neural-encode labels \
  --database data/labels.sqlite \
  --count 30000
```

The command recreates the `labels` table. It does not append duplicate rows.

## Generate one signal per label

```bash
neural-encode labels \
  --database data/labels.sqlite \
  --count 30000 \
  --export-dir data/corpus \
  --seed 1000
```

The current implementation generates one sample per label. The seed is
calculated as:

```text
sample 000000 → seed 1000
sample 000001 → seed 1001
sample 000002 → seed 1002
...
```

Therefore, `--seed 1000` means “start the deterministic seed sequence at
1000”; it does not mean “generate 1,000 recordings for every sentence.”

The output filenames are stable:

```text
data/corpus/sample_000000.npz
data/corpus/sample_000001.npz
...
```

Running the same export again with the same output directory overwrites those
same files. It does not append another copy.

## Inspect the label database

Count labels:

```bash
sqlite3 data/labels.sqlite \
  "SELECT COUNT(*) AS total_sentences FROM labels;"
```

Count unique labels:

```bash
sqlite3 data/labels.sqlite \
  "SELECT COUNT(*) AS total, COUNT(DISTINCT text) AS unique_text FROM labels;"
```

Count labels in each category:

```bash
sqlite3 -header -column data/labels.sqlite \
  "SELECT category, COUNT(*) AS sentence_count
   FROM labels
   GROUP BY category
   ORDER BY sentence_count DESC;"
```

Print the first 20 labels:

```bash
sqlite3 -header -column data/labels.sqlite \
  "SELECT id, category, text
   FROM labels
   ORDER BY id
   LIMIT 20;"
```

Search labels:

```bash
sqlite3 -header -column data/labels.sqlite \
  "SELECT id, category, text
   FROM labels
   WHERE text LIKE '%signal%'
   LIMIT 50;"
```

## Count generated signal files

```bash
find data/corpus -maxdepth 1 -type f -name '*.npz' | wc -l
```

## Run tests

```bash
python -m pytest -q
```

## Important dataset note

The current `labels` export creates one signal sample for each sentence. To
create repeated recordings of every sentence, the corpus generator will need a
separate repeated-seed option. That should produce, for example:

```text
30,000 labels × 5 recordings = 150,000 samples
```

Until that option is implemented, do not assume that a single export creates
multiple recordings per sentence.

## Train and test the decoder

Install PyTorch and the decoder package:

```bash
python -m pip install -e ".[all]"
```

Run a safe smoke test:

```bash
neural-decode train \
  --data-dir data/corpus \
  --checkpoint checkpoints/decoder.pt \
  --epochs 1 \
  --max-samples 256
```

Train on the full lazy-loaded corpus:

```bash
neural-decode train \
  --data-dir data/corpus \
  --checkpoint checkpoints/decoder.pt \
  --epochs 3 \
  --batch-size 8
```

Training prints progress at the first batch, every 50 batches, and the final
batch of each epoch. Each progress line includes the epoch, batch count,
percentage, running loss, and elapsed time. Change the interval with
`--progress-every`, for example `--progress-every 10`.

Evaluate the test split:

```bash
neural-decode evaluate \
  --data-dir data/corpus \
  --checkpoint checkpoints/decoder.pt \
  --split test
```

Decode one existing NPZ:

```bash
neural-decode decode-file \
  --checkpoint checkpoints/decoder.pt \
  --sample data/corpus/sample_000000.npz
```

Manual end-to-end test:

```bash
neural-decode manual \
  --checkpoint checkpoints/decoder.pt \
  --seed 40000 \
  "the decoder analyzes a neural signal"
```

```bash
neural-decode manual \
  --checkpoint checkpoints/decoder.pt \
  --waterfall assets/my-test.png \
  --seed 40001 \
  --device cpu \
  "example manual test"
```

Manual mode creates the signal, writes it to a temporary NPZ for the same
loading path used by real data, runs inference, prints the prediction, and
deletes the temporary file.

Manual mode also overwrites this waterfall image by default:

```text
data/manual_waterfall.png
```

It shows the exact signal generated from the current input sentence. Choose a
different output path with `--waterfall`, or disable the image with
`--no-waterfall`.

To type the sentence interactively instead:

```bash
neural-decode manual \
  --checkpoint checkpoints/decoder.pt \
  --seed 40000
```
