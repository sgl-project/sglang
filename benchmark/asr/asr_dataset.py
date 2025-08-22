from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ASRSample:
    # Raw waveform and sample rate
    audio: np.ndarray
    sr: int
    # Optional ground-truth text if provided by the dataset
    text: Optional[str]


def _get_duration_seconds(y: np.ndarray, sr: int) -> float:
    if sr <= 0:
        return 0.0
    return float(y.shape[0]) / float(sr)


class ASRDataset:
    """Simple loader for HF ASR-style datasets.

    Expects an audio column named `audio` with fields `array` and `sampling_rate`.
    Optionally reads `text` or `transcription` as reference text if present.
    """

    def __init__(
        self,
        path: str = "openslr/librispeech_asr",
        split: str = "test",
        subset: Optional[str] = None,
        skip_long: bool = True,
        max_duration_s: float = 30.0,
    ) -> None:
        self.path = path
        self.split = split
        self.subset = subset
        self.skip_long = skip_long
        self.max_duration_s = max_duration_s

        # Lazy-loaded dataset
        self._data = None

    def _ensure_loaded(self) -> None:
        if self._data is not None:
            return
        from datasets import load_dataset  # Lazy import to keep startup fast

        if self.subset:
            self._data = load_dataset(self.path, self.subset, split=self.split)
        else:
            self._data = load_dataset(self.path, split=self.split)

    def iter_samples(self, limit: Optional[int] = None) -> List[ASRSample]:
        self._ensure_loaded()
        results: List[ASRSample] = []
        skipped = 0
        for item in self._data:
            audio = item.get("audio")
            if not audio:
                continue
            y = audio["array"]
            sr = int(audio["sampling_rate"]) if "sampling_rate" in audio else 0

            if isinstance(y, list):
                y = np.asarray(y)

            # Convert stereo to mono if needed
            if y.ndim > 1:
                y = np.mean(y, axis=1)

            dur = _get_duration_seconds(y, sr)
            if self.skip_long and dur > self.max_duration_s:
                skipped += 1
                continue

            text = None
            if "text" in item and isinstance(item["text"], str):
                text = item["text"]
            elif "transcription" in item and isinstance(item["transcription"], str):
                text = item["transcription"]

            results.append(ASRSample(audio=y, sr=sr, text=text))
            if limit is not None and len(results) >= limit:
                break

        return results

