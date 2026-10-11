"""Command-line interface for encoding and corpus generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .corpus import default_sentences, generate_corpus, read_labels
from .encoder import NeuralSignalEncoder, load_sample, save_sample
from .label_db import create_label_database, read_database_labels
from .visualizer import save_waterfall


def main() -> None:
    parser = argparse.ArgumentParser(prog="neural-encode")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode = subparsers.add_parser("encode", help="encode one sentence")
    encode.add_argument("text")
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--seed", type=int, default=0)
    encode.add_argument("--plot", type=Path)

    corpus = subparsers.add_parser("corpus", help="generate a labeled NPZ corpus")
    corpus.add_argument("--output-dir", type=Path, required=True)
    corpus.add_argument("--labels", type=Path)
    corpus.add_argument("--count", type=int, default=1000)
    corpus.add_argument("--seed", type=int, default=0)

    labels = subparsers.add_parser("labels", help="create or use a SQLite label catalog")
    labels.add_argument("--database", type=Path, required=True)
    labels.add_argument("--count", type=int)
    labels.add_argument("--export-dir", type=Path)
    labels.add_argument("--seed", type=int, default=0)

    visualize = subparsers.add_parser(
        "visualize",
        help="render selected NPZ samples as waterfall images",
    )
    visualize.add_argument("--input-dir", type=Path, required=True)
    visualize.add_argument("--output-dir", type=Path, required=True)
    visualize.add_argument("--count", type=int, default=5)

    args = parser.parse_args()
    encoder = NeuralSignalEncoder()
    if args.command == "encode":
        sample = encoder.encode(args.text, seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_sample(sample, str(args.output))
        if args.plot:
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            save_waterfall(sample.signal, args.plot, title=sample.transcript)
        print(f"saved {args.output} shape={sample.signal.shape} seed={sample.seed}")
    elif args.command == "corpus":
        labels = read_labels(args.labels) if args.labels else default_sentences(args.count)
        written = generate_corpus(encoder, labels, args.output_dir, seed=args.seed)
        print(f"saved {written} samples to {args.output_dir}")
    elif args.command == "labels":
        count = create_label_database(args.database, limit=args.count)
        print(f"saved {count} labels to {args.database}")
        if args.export_dir:
            written = generate_corpus(
                encoder,
                read_database_labels(args.database),
                args.export_dir,
                seed=args.seed,
            )
            print(f"saved {written} samples to {args.export_dir}")
    else:
        if args.count < 1:
            raise ValueError("--count must be positive")
        sample_paths = sorted(args.input_dir.glob("*.npz"))
        if not sample_paths:
            raise FileNotFoundError(f"no NPZ samples found in {args.input_dir}")
        selected = sample_paths[:: max(1, len(sample_paths) // args.count)][: args.count]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for index, sample_path in enumerate(selected):
            sample = load_sample(str(sample_path))
            output_path = args.output_dir / f"waterfall_{index:02d}.png"
            save_waterfall(sample["signal"], output_path, title=sample["transcript"])
            print(f"saved {output_path} from {sample_path.name}: {sample['transcript']}")
