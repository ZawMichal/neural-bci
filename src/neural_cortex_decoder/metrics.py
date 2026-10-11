"""Text decoding and error metrics."""

from __future__ import annotations

from .vocab import decode_ids


def greedy_decode(logits, input_lengths) -> list[str]:
    predictions = logits.argmax(dim=-1).detach().cpu().tolist()
    lengths = input_lengths.detach().cpu().tolist()
    return [decode_ids(row[:length]) for row, length in zip(predictions, lengths)]


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def error_rate(predictions: list[str], targets: list[str], *, words: bool = False) -> float:
    if len(predictions) != len(targets):
        raise ValueError("prediction and target counts must match")
    total_errors = sum(
        edit_distance(prediction.split() if words else prediction,
                      target.split() if words else target)
        for prediction, target in zip(predictions, targets)
    )
    total_units = sum(len(target.split() if words else target) for target in targets)
    return total_errors / max(1, total_units)
