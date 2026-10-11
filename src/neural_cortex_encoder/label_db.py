"""SQLite label catalog for reproducible communication-focused corpora."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sqlite3
from typing import Iterable


@dataclass(frozen=True)
class LabelRecord:
    text: str
    category: str


FIXED_LABELS: tuple[tuple[str, str], ...] = tuple(
    ("greeting", phrase)
    for phrase in (
        "hello.",
        "hi.",
        "good morning.",
        "good afternoon.",
        "good evening.",
        "welcome.",
        "nice to meet you.",
        "how are you?",
        "i am glad to see you.",
        "thank you.",
        "thanks for your help.",
        "please continue.",
        "goodbye.",
        "see you later.",
        "have a good day.",
    )
)

TEMPLATE_FAMILIES: tuple[tuple[str, str, tuple[tuple[str, ...], ...]], ...] = (
    (
        "identity",
        "i am {identity}.",
        (
            ("a person", "a researcher", "an engineer", "a student", "a user"),
        ),
    ),
    (
        "identity",
        "my name is {name}.",
        (("alex", "sam", "jordan", "taylor", "morgan"),),
    ),
    (
        "request",
        "please {action} {object}.",
        (
            ("repeat", "explain", "describe", "check", "save", "open", "close"),
            ("the signal", "the result", "the message", "the file", "the report"),
        ),
    ),
    (
        "request",
        "could you {action}?",
        (("help me", "repeat that", "slow down", "wait", "start the analysis"),),
    ),
    (
        "question",
        "what is {subject}?",
        (
            (
                "the answer",
                "the current status",
                "the signal quality",
                "the next step",
                "the time",
                "the result",
            ),
        ),
    ),
    (
        "question",
        "where is {subject}?",
        (("the file", "the device", "the meeting", "the output", "the signal"),),
    ),
    (
        "question",
        "when will {event}?",
        (
            (
                "the analysis finish",
                "the recording start",
                "the model respond",
                "the meeting begin",
            ),
        ),
    ),
    (
        "question",
        "can you tell me whether {condition}?",
        (
            (
                "the signal is valid",
                "the file was saved",
                "the model is ready",
                "the connection is stable",
            ),
        ),
    ),
    (
        "statement",
        "i {verb} {object}.",
        (
            (
                "understand",
                "do not understand",
                "remember",
                "do not remember",
                "agree with",
                "disagree with",
            ),
            ("the message", "the result", "the instruction", "the request"),
        ),
    ),
    (
        "status",
        "the {system} is {state}.",
        (
            ("system", "decoder", "encoder", "recording", "connection", "dataset"),
            ("ready", "running", "paused", "complete", "unavailable", "calibrated"),
        ),
    ),
    (
        "command",
        "{command} the {target}.",
        (
            ("start", "stop", "pause", "resume", "reset", "save", "load", "delete"),
            ("recording", "decoder", "encoder", "session", "experiment", "export"),
        ),
    ),
    (
        "confirmation",
        "{answer}, {statement}.",
        (
            ("yes", "no", "correct", "that is right", "that is not right"),
            ("the signal is clean", "the file is ready", "the test is complete"),
        ),
    ),
    (
        "correction",
        "please correct {item}.",
        (("the sentence", "the label", "the timestamp", "the channel map", "the result"),),
    ),
    (
        "preference",
        "i prefer {preference}.",
        (
            (
                "a shorter answer",
                "a detailed explanation",
                "written text",
                "spoken audio",
                "a visual preview",
            ),
        ),
    ),
    (
        "emotion",
        "i feel {feeling}.",
        (("calm", "confident", "confused", "tired", "ready", "concerned"),),
    ),
    (
        "time",
        "the event happened {time}.",
        (
            ("today", "yesterday", "this morning", "recently", "a few minutes ago"),
        ),
    ),
    (
        "location",
        "the {thing} is {location}.",
        (
            ("device", "file", "recording", "sample", "report"),
            ("here", "there", "on the desk", "in the output folder", "in the database"),
        ),
    ),
    (
        "technical",
        "the {component} uses {setting}.",
        (
            ("encoder", "decoder", "visualizer", "dataset", "training process"),
            ("a fixed seed", "one thousand channels", "a low noise level", "real time inference"),
        ),
    ),
    (
        "safety",
        "stop because {reason}.",
        (
            (
                "the signal is unstable",
                "the file is corrupted",
                "the temperature is too high",
                "the connection was lost",
            ),
        ),
    ),
    (
        "social",
        "{person} {action}.",
        (
            ("i", "we", "you", "the user", "the operator"),
            (
                "need help",
                "can wait",
                "want to continue",
                "understand the result",
                "accept the answer",
            ),
        ),
    ),
    (
        "explanation",
        "the {subject} means {meaning}.",
        (
            ("word", "signal", "label", "pattern", "result", "message"),
            (
                "the input is valid",
                "the test is complete",
                "the system is ready",
                "the activity is increasing",
                "the activity is decreasing",
                "the answer is uncertain",
            ),
        ),
    ),
    (
        "observation",
        "i can see {observation}.",
        (
            (
                "a clear pattern",
                "a bright signal",
                "a change in activity",
                "a new label",
                "the latest result",
                "the current status",
            ),
        ),
    ),
    (
        "planning",
        "we will {action} {when}.",
        (
            (
                "run the experiment",
                "review the result",
                "record another sample",
                "train the model",
                "export the dataset",
                "compare the signals",
            ),
            ("now", "later", "tomorrow", "after calibration", "before the meeting"),
        ),
    ),
    (
        "comparison",
        "the {first} is {relation} than the {second}.",
        (
            ("new signal", "current result", "decoded sentence", "latest sample"),
            ("better", "worse", "clearer", "more stable"),
            ("old signal", "previous result", "reference sentence", "first sample"),
        ),
    ),
    (
        "access",
        "i need access to {resource}.",
        (
            (
                "the database",
                "the recording",
                "the output folder",
                "the model",
                "the configuration",
                "the test results",
            ),
        ),
    ),
    (
        "feedback",
        "the answer is {quality}.",
        (
            (
                "clear",
                "useful",
                "correct",
                "incomplete",
                "too long",
                "too short",
                "hard to understand",
            ),
        ),
    ),
    (
        "measurement",
        "the {measurement} is {value}.",
        (
            ("signal quality", "latency", "channel count", "sample rate", "confidence"),
            (
                "high",
                "low",
                "stable",
                "within the expected range",
                "outside the expected range",
            ),
        ),
    ),
    (
        "data",
        "save {data} as {format}.",
        (
            ("the signal", "the labels", "the report", "the training data"),
            ("a compressed file", "a text file", "an npz file", "a database record"),
        ),
    ),
    (
        "meeting",
        "the meeting is {meeting_state}.",
        (
            (
                "scheduled",
                "starting soon",
                "in progress",
                "finished",
                "cancelled",
                "rescheduled",
            ),
        ),
    ),
    (
        "travel",
        "i am going to {destination} {time}.",
        (
            ("the laboratory", "the office", "the meeting", "the next session"),
            ("now", "soon", "today", "tomorrow"),
        ),
    ),
    (
        "learning",
        "teach me about {topic}.",
        (
            (
                "the signal",
                "the encoder",
                "the decoder",
                "the training process",
                "the evaluation metric",
                "the visualization",
            ),
        ),
    ),
    (
        "permission",
        "may i {action}?",
        (
            (
                "start the recording",
                "use this sample",
                "change the settings",
                "run the decoder",
                "share the result",
            ),
        ),
    ),
    (
        "workflow",
        "{actor} will {action} the {object}.",
        (
            ("i", "we", "you", "the user", "the operator", "the model", "the encoder", "the decoder"),
            ("start", "stop", "review", "save", "load", "check", "share", "repeat"),
            ("recording", "signal", "sentence", "result", "dataset", "report", "experiment", "session"),
        ),
    ),
    (
        "communication",
        "{speaker} {verb} that the {subject} is {state}.",
        (
            ("i", "we", "you", "the user", "the operator", "the scientist", "the engineer"),
            ("say", "confirm", "report", "explain", "believe", "remember", "notice"),
            ("signal", "model", "recording", "connection", "decoder", "dataset", "experiment"),
            ("ready", "stable", "complete", "active", "valid", "available"),
        ),
    ),
    (
        "analysis",
        "the {actor} {verb} the {object} {context}.",
        (
            (
                "scientist",
                "engineer",
                "operator",
                "researcher",
                "model",
                "decoder",
                "encoder",
                "system",
                "assistant",
                "student",
            ),
            (
                "analyzes",
                "decodes",
                "records",
                "measures",
                "tracks",
                "detects",
                "encodes",
                "classifies",
                "compares",
                "validates",
                "reviews",
                "summarizes",
            ),
            (
                "the neural signal",
                "the channel activity",
                "the cortical pattern",
                "the current sentence",
                "the signal window",
                "the training sample",
                "the database record",
                "the decoded result",
                "the noisy waveform",
                "the model output",
                "the timing profile",
                "the activity map",
            ),
            (
                "carefully",
                "in real time",
                "during calibration",
                "with low noise",
                "with stable timing",
                "before the next test",
                "after the recording",
                "for quality control",
                "without changing the input",
                "using the reference data",
            ),
        ),
    ),
    (
        "request",
        "{speaker} asks you to {action} the {object} {when}.",
        (
            ("i", "we", "you", "the user", "the operator", "the scientist", "the engineer", "the assistant"),
            (
                "start",
                "stop",
                "pause",
                "resume",
                "review",
                "save",
                "load",
                "check",
                "share",
                "repeat",
            ),
            (
                "recording",
                "signal",
                "sentence",
                "result",
                "dataset",
                "report",
                "experiment",
                "session",
                "configuration",
                "channel map",
                "model",
                "output file",
            ),
            ("now", "soon", "today", "tomorrow", "after calibration", "before the meeting", "when ready", "one more time"),
        ),
    ),
    (
        "question",
        "can {speaker} {action} the {object} {when}?",
        (
            ("i", "we", "you", "the user", "the operator", "the scientist", "the engineer", "the assistant"),
            (
                "start",
                "stop",
                "pause",
                "resume",
                "review",
                "save",
                "load",
                "check",
                "share",
                "repeat",
            ),
            (
                "recording",
                "signal",
                "sentence",
                "result",
                "dataset",
                "report",
                "experiment",
                "session",
                "configuration",
                "channel map",
                "model",
                "output file",
            ),
            ("now", "soon", "today", "tomorrow", "after calibration", "before the meeting", "when ready", "one more time"),
        ),
    ),
)


def iter_label_records(limit: int | None = None) -> Iterable[LabelRecord]:
    """Yield deterministic, deduplicated labels in communication categories."""
    seen: set[str] = set()
    produced = 0

    for category, text in FIXED_LABELS:
        if text not in seen:
            seen.add(text)
            yield LabelRecord(text, category)
            produced += 1
            if limit is not None and produced >= limit:
                return

    for category, template, fields in TEMPLATE_FAMILIES:
        for values in product(*fields):
            text = template.format(**dict(zip(_field_names(template), values)))
            if text in seen:
                continue
            seen.add(text)
            yield LabelRecord(text, category)
            produced += 1
            if limit is not None and produced >= limit:
                return


def _field_names(template: str) -> list[str]:
    names: list[str] = []
    start = 0
    while True:
        start = template.find("{", start)
        if start < 0:
            return names
        end = template.index("}", start)
        names.append(template[start + 1 : end])
        start = end + 1


def create_label_database(path: str | Path, *, limit: int | None = None) -> int:
    """Create or replace a SQLite catalog and return the inserted row count."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = list(iter_label_records(limit))
    with sqlite3.connect(destination) as connection:
        connection.executescript(
            """
            DROP TABLE IF EXISTS labels;
            CREATE TABLE labels (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                token_count INTEGER NOT NULL
            );
            CREATE INDEX idx_labels_category ON labels(category);
            """
        )
        connection.executemany(
            """
            INSERT INTO labels(text, category, char_count, token_count)
            VALUES (?, ?, ?, ?)
            """,
            (
                (record.text, record.category, len(record.text), len(record.text.split()))
                for record in records
            ),
        )
    return len(records)


def read_database_labels(path: str | Path) -> list[str]:
    """Read labels in stable id order for the NPZ corpus generator."""
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT text FROM labels ORDER BY id").fetchall()
    if not rows:
        raise ValueError("label database contains no labels")
    return [str(row[0]) for row in rows]
