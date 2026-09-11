"""Energy-aware audio chunking for ASR models with a bounded input window.

Whisper-style encoders ingest a fixed-length window (30 s of audio = 3000
mel frames); the feature extractor silently truncates anything longer, so
long audio must be split into independent chunks before prompting. Cutting
blindly at the window boundary can land mid-word and corrupt the
transcription on both sides of the seam, so the splitter searches the tail
of each window for the quietest stretch (lowest RMS energy) and cuts there.

This matches the chunking behavior of vLLM's
``OpenAISpeechToText._split_audio`` / ``_find_split_point``: chunks are
contiguous and non-overlapping — the "search window" is only the region in
which the cut is allowed to land, and stitching the transcripts back is a
plain in-order concatenation.
"""

from __future__ import annotations

import io
import math
from typing import List, Tuple

import numpy as np
import soundfile as sf

from sglang.srt.utils import load_audio

# Region at the tail of each max-length window in which to search for a
# low-energy split point, in seconds.
SPLIT_SEARCH_WINDOW_S = 1.0

# RMS energy is evaluated over strides of this many samples (100 ms at
# 16 kHz); the quietest stride in the search region wins.
MIN_ENERGY_WINDOW_SIZE = 1600


def find_split_point(wav: np.ndarray, start_idx: int, end_idx: int) -> int:
    """Return the start index of the quietest energy window in
    ``wav[start_idx:end_idx]``.

    Energy is RMS over consecutive ``MIN_ENERGY_WINDOW_SIZE``-sample
    strides; the returned index is absolute (into ``wav``). The loop bound
    intentionally leaves the final stride of the search region unevaluated:
    it mirrors vLLM's ``_find_split_point`` verbatim so split points stay
    bit-identical with the old vLLM transcription endpoint.
    """
    segment = wav[start_idx:end_idx]
    min_energy = math.inf
    quietest_idx = start_idx
    for i in range(0, len(segment) - MIN_ENERGY_WINDOW_SIZE, MIN_ENERGY_WINDOW_SIZE):
        window = segment[i : i + MIN_ENERGY_WINDOW_SIZE]
        energy = (window**2).mean() ** 0.5
        if energy < min_energy:
            quietest_idx = i + start_idx
            min_energy = energy
    return quietest_idx


def split_audio_energy_aware(
    audio_data: bytes,
    max_clip_s: float,
    sample_rate: int = 16000,
) -> Tuple[List[bytes], List[float]]:
    """Split audio into WAV chunks no longer than ``max_clip_s`` seconds.

    Decodes (and resamples to ``sample_rate``) the input, walks it in
    ``max_clip_s`` strides, and cuts each chunk at the lowest-energy point
    within the final ``SPLIT_SEARCH_WINDOW_S`` of the stride so cuts land
    on pauses instead of mid-word. Chunks are contiguous and
    non-overlapping; concatenated they reproduce the full waveform.

    Returns ``(chunk_wav_bytes, chunk_start_offsets_s)`` where
    ``chunk_start_offsets_s[i]`` is the start time of chunk ``i`` in the
    original audio.
    """
    if not audio_data:
        raise ValueError("audio_data is empty")
    audio = load_audio(audio_data, sr=sample_rate, mono=True)
    chunk_size = int(sample_rate * max_clip_s)
    search_size = int(sample_rate * SPLIT_SEARCH_WINDOW_S)
    total = audio.shape[-1]

    raw_chunks: List[np.ndarray] = []
    offsets_s: List[float] = []
    i = 0
    while i < total:
        offsets_s.append(i / sample_rate)
        if i + chunk_size >= total:
            raw_chunks.append(audio[i:])
            break
        search_start = i + chunk_size - search_size
        search_end = min(i + chunk_size, total)
        split_point = find_split_point(audio, search_start, search_end)
        raw_chunks.append(audio[i:split_point])
        i = split_point

    chunks: List[bytes] = []
    for chunk in raw_chunks:
        buf = io.BytesIO()
        sf.write(buf, chunk, sample_rate, format="WAV")
        chunks.append(buf.getvalue())
    return chunks, offsets_s
