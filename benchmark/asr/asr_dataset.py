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

            # Handle multiple HF audio representations robustly
            y = None
            sr = 0
            try:
                # Case 1: Standard dict with "array" and "sampling_rate"
                if isinstance(audio, dict) and "array" in audio:
                    y = audio["array"]
                    sr = int(audio.get("sampling_rate", 0))

                # Case 2: TorchCodec decoder object (streaming or backend-enabled)
                # Prefer decoding raw bytes to avoid get_all_samples lifecycle issues
                elif hasattr(audio, "metadata") and hasattr(audio, "get_all_samples"):
                    sr = int(getattr(audio.metadata, "sample_rate", 0))
                    hf_encoded = getattr(audio, "_hf_encoded", None)
                    if isinstance(hf_encoded, dict) and isinstance(hf_encoded.get("bytes"), (bytes, bytearray)):
                        try:
                            import soundfile as sf
                            from io import BytesIO

                            bio = BytesIO(hf_encoded["bytes"])
                            y, decoded_sr = sf.read(bio)
                            if sr <= 0:
                                sr = int(decoded_sr)
                        except Exception:
                            # Fallback to decoder API if bytes decode fails
                            samples = audio.get_all_samples()
                            y = np.asarray(samples.numpy() if hasattr(samples, "numpy") else samples)
                    else:
                        # No bytes available; use decoder API
                        samples = audio.get_all_samples()
                        y = np.asarray(samples.numpy() if hasattr(samples, "numpy") else samples)

                # Case 3: Other unexpected types (attempt minimal conversion)
                else:
                    if hasattr(audio, "numpy"):
                        y = np.asarray(audio.numpy())
                    elif hasattr(audio, "__array__"):
                        y = np.asarray(audio)

            except Exception:
                # Skip problematic sample
                continue

            if y is None or sr <= 0:
                continue

            if isinstance(y, list):
                y = np.asarray(y)

            # Convert stereo to mono if needed
            if hasattr(y, "ndim") and y.ndim > 1:
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
