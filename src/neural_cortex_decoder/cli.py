"""CLI for training, evaluation, and manual signal-to-text decoding."""

from __future__ import annotations

import argparse

from .engine import decode_file, evaluate, manual_decode, train


def main() -> None:
    parser = argparse.ArgumentParser(prog="neural-decode")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-dir", required=True)
    train_parser.add_argument("--checkpoint", required=True)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--max-samples", type=int)
    train_parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    train_parser.add_argument("--progress-every", type=int, default=50)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--data-dir", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--split", default="test", choices=("train", "validation", "test", "all"))
    evaluate_parser.add_argument("--batch-size", type=int, default=8)
    evaluate_parser.add_argument("--max-samples", type=int)
    evaluate_parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))

    decode_parser = subparsers.add_parser("decode-file")
    decode_parser.add_argument("--checkpoint", required=True)
    decode_parser.add_argument("--sample", required=True)
    decode_parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))

    manual_parser = subparsers.add_parser("manual")
    manual_parser.add_argument("--checkpoint", required=True)
    manual_parser.add_argument("--seed", type=int, default=40000)
    manual_parser.add_argument("--waterfall", default="data/manual_waterfall.png")
    manual_parser.add_argument("--no-waterfall", action="store_true")
    manual_parser.add_argument("text", nargs="?")
    manual_parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))

    args = parser.parse_args()
    if args.command == "train":
        train(
            args.data_dir,
            args.checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples,
            device_name=args.device,
            progress_every=args.progress_every,
        )
    elif args.command == "evaluate":
        evaluate(
            args.data_dir,
            args.checkpoint,
            split=args.split,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device_name=args.device,
        )
    elif args.command == "decode-file":
        print(decode_file(args.checkpoint, args.sample, device_name=args.device))
    else:
        text = args.text if args.text is not None else input("Text to encode and decode: ")
        print(manual_decode(
            args.checkpoint,
            text,
            seed=args.seed,
            device_name=args.device,
            waterfall_path=None if args.no_waterfall else args.waterfall,
        ))
