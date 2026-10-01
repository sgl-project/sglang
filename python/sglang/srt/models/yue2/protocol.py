# SPDX-License-Identifier: Apache-2.0
"""Versioned, checkpoint-native prompting and generation defaults."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import math
import re

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
LATENT_START, LATENT_END, LATENT_PAD = 184621, 184622, 184623
VOCAB_SIZE, CONTEXT = 184704, 24576
PROTOCOL_VERSION = "yue2-native-v1"
INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
    "full": "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
}


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 100
    repetition_penalty: float = 1.2
    penalty_window: int = 50
    min_tokens: int = 200
    max_tokens: int = 9000

    def __post_init__(self):
        if any(type(x) is not int for x in (self.top_k, self.penalty_window, self.min_tokens, self.max_tokens)):
            raise ValueError("Sampling counts must be integers")
        if not all(math.isfinite(x) for x in (self.temperature, self.top_p, self.repetition_penalty)):
            raise ValueError("Sampling numbers must be finite")
        if not 0 <= self.temperature <= 5 or not 0 < self.top_p <= 1 or self.top_k < 1:
            raise ValueError("Invalid sampling temperature/top_p/top_k")
        if self.repetition_penalty <= 0 or not 1 <= self.penalty_window <= 100:
            raise ValueError("Invalid repetition penalty/window")
        if not 0 <= self.min_tokens <= self.max_tokens or self.max_tokens < 1:
            raise ValueError("Require 0 <= min_tokens <= max_tokens")


@dataclass(frozen=True)
class GenerationConfig:
    abc: Sampling = field(default_factory=lambda: Sampling(.7, .9, 30, 1.005, 100, 32, 4096))
    semantic: Sampling = field(default_factory=Sampling)
    ode_steps: int = 32
    ode_method: str = "midpoint"
    context: int = CONTEXT
    version: str = PROTOCOL_VERSION

    def __post_init__(self):
        if self.context != CONTEXT or self.ode_method != "midpoint" or type(self.ode_steps) is not int or self.ode_steps < 1:
            raise ValueError("Require context=24576 and midpoint with positive integer steps")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        value = dict(data)
        defaults = cls()
        for key in ("abc", "semantic"):
            if key in value and isinstance(value[key], dict):
                value[key] = Sampling(**{**asdict(getattr(defaults, key)), **value[key]})
        return cls(**value)


def resolve_sampling(value, default):
    if value is None:
        return default
    if isinstance(value, Sampling):
        return value
    if isinstance(value, dict):
        return Sampling(**{**asdict(default), **value})
    raise TypeError("Sampling must be a Sampling object or a dictionary of overrides")


@dataclass(frozen=True)
class SongRequest:
    style: str
    lyrics: str
    cot: str = "full"
    seed: int = 831001
    abc: str | None = None
    cfg_scale: float | None = None
    id: str = "song"

    def __post_init__(self):
        if self.cot not in INSTRUCTIONS:
            raise ValueError("cot must be off, melody or full")
        if not isinstance(self.style, str) or not isinstance(self.lyrics, str):
            raise TypeError("style and lyrics must be strings")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", self.id) or self.id in {".", ".."}:
            raise ValueError("id must be a filename-safe identifier")
        if self.abc is not None and (self.cot == "off" or not isinstance(self.abc, str) or not self.abc.strip()):
            raise ValueError("External ABC requires nonempty text and cot=melody/full")
        if self.cfg_scale is not None and (not math.isfinite(self.cfg_scale) or not 0 <= self.cfg_scale <= 20):
            raise ValueError("cfg_scale must be finite and in [0,20]")

    @property
    def guidance(self):
        return (1.01 if self.cot == "off" else 1.0) if self.cfg_scale is None else self.cfg_scale

    def text(self):
        return f"{INSTRUCTIONS[self.cot]}\n[Tags]\n{self.style}\n[Lyrics]\n{self.lyrics}\n"

    def to_dict(self):
        return asdict(self)


def token_prefixes(request, tokenizer, abc_ids=None):
    base = [EOD] + tokenizer.encode(request.text())
    if request.cot == "off":
        return base + [ABC_START, ABC_END, MUSIC_START]
    if abc_ids is None:
        if request.abc is None:
            return base + [ABC_START]
        abc_ids = tokenizer.encode(request.abc)
    abc_ids = list(abc_ids)
    if any(type(token) is not int or not 0 <= token < EOD for token in abc_ids):
        raise ValueError("ABC IDs must remain inside the ordinary text vocabulary")
    return base + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]


def negative_prefix(request, tokenizer, abc_ids=None):
    base = [EOD] + tokenizer.encode(INSTRUCTIONS[request.cot])
    if request.cot == "off":
        return base + [MUSIC_START]
    if abc_ids is None:
        raise ValueError("Symbolic CFG must retain the exact positive-branch ABC IDs")
    abc_ids = list(abc_ids)
    if any(type(token) is not int or not 0 <= token < EOD for token in abc_ids):
        raise ValueError("ABC IDs must remain inside the ordinary text vocabulary")
    return base + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]


def chunk_ranges(frames, prefix_tokens, context=CONTEXT):
    size = min((context - prefix_tokens - 3) // 2, CONTEXT)
    if frames < 1 or size < 1:
        raise ValueError("Empty codec or prefix leaves no acoustic context")
    return [(a, min(a + size, frames)) for a in range(0, frames, size)]
