"""Sampling parameters for text generation."""

from typing import List, Optional, Union

_SAMPLING_EPS = 1e-6


class SamplingParams:
    def __init__(
        self,
        max_new_tokens: int = 16,
        stop: Optional[Union[str, List[str]]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        ignore_eos: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        dtype: Optional[str] = None,
        regex: Optional[str] = None,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop_strs = stop
        self.max_new_tokens = max_new_tokens
        self.ignore_eos = ignore_eos
        self.skip_special_tokens = skip_special_tokens
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self.dtype = dtype
        self.regex = regex

        # Process some special cases
        if self.temperature < _SAMPLING_EPS:
            self.temperature = 1.0
            self.top_k = 1
        if self.top_k == -1:
            self.top_k = 1 << 30  # whole vocabulary
        if self.dtype == "int":
            self.stop_strs = [" ", "\n"]

    def verify(self):
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(
                f"top_k must be -1 (disable), or at least 1, " f"got {self.top_k}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                "frequency_penalty must be in [-2, 2], got "
                f"{self.frequency_penalty}."
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                "presence_penalty must be in [-2, 2], got " f"{self.presence_penalty}."
            )
        if self.max_new_tokens < 0:
            raise ValueError(
                f"max_new_tokens must be at least 0, got {self.max_new_tokens}."
            )

    def normalize(self, tokenizer):
        # Process stop strings
        if self.stop_strs is None:
            self.stop_strs = []
            self.stop_str_max_len = 0
        else:
            if isinstance(self.stop_strs, str):
                self.stop_strs = [self.stop_strs]

            stop_str_max_len = 0
            for stop_str in self.stop_strs:
                stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                stop_str_max_len = max(stop_str_max_len, len(stop_str_ids))
            self.stop_str_max_len = stop_str_max_len
