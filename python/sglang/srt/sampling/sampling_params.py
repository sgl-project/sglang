# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Sampling parameters for text generation."""

from typing import Any, Dict, List, Optional, Union

_SAMPLING_EPS = 1e-6


class SamplingParams:
    """
    The sampling parameters.

    See docs/backend/sampling_params.md or
    https://docs.sglang.ai/backend/sampling_params.html
    for the documentation.
    """

    def __init__(
        self,
        max_new_tokens: int = 128,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        n: int = 1,
        json_schema: Optional[str] = None,
        regex: Optional[str] = None,
        ebnf: Optional[str] = None,
        structural_tag: Optional[str] = None,
        ignore_eos: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        no_stop_trim: bool = False,
        min_reasoning_penalty: float = 0.0,
        max_reasoning_penalty: float = 0.0,
        num_reasoning_penalty_steps: int = 0,
        stop_reasoning: Optional[Union[str, List[str]]] = None,
        stop_reasoning_token_ids: Optional[List[int]] = None,
        ngram_penalty: float = 0.0,
        ngram_n: int = 32,
        ngram_lookback_window: int = 512,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.stop_strs = stop
        if stop_token_ids:
            self.stop_token_ids = set(stop_token_ids)
        else:
            self.stop_token_ids = None
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.min_new_tokens = min_new_tokens
        self.regex = regex
        self.n = n
        self.json_schema = json_schema
        self.ebnf = ebnf
        self.structural_tag = structural_tag
        self.ignore_eos = ignore_eos
        self.skip_special_tokens = skip_special_tokens
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self.no_stop_trim = no_stop_trim
        self.min_reasoning_penalty = min_reasoning_penalty
        self.max_reasoning_penalty = max_reasoning_penalty
        self.num_reasoning_penalty_steps = num_reasoning_penalty_steps
        self.stop_reasoning_strs = stop_reasoning
        self.stop_reasoning_token_ids = stop_reasoning_token_ids
        self.custom_params = custom_params
        self.ngram_penalty = ngram_penalty
        self.ngram_n = ngram_n
        self.ngram_lookback_window = ngram_lookback_window

        print('sampling params init')
        print("temperature: ", self.temperature)

        # Process some special cases
        if self.temperature < _SAMPLING_EPS:
            # top_k = 1 means greedy sampling
            self.temperature = 1.0
            self.top_k = 1
        if self.top_k == -1:
            self.top_k = 1 << 30  # whole vocabulary

        if self.stop_reasoning_token_ids is None:
            self.stop_reasoning_token_ids = set()
        else:
            self.stop_reasoning_token_ids = set(self.stop_reasoning_token_ids)
        
        if self.stop_reasoning_strs is None:
            self.stop_reasoning_strs = set()
        else:
            self.stop_reasoning_strs = set(self.stop_reasoning_strs)

    def verify(self):
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
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
        if not 0.0 <= self.repetition_penalty <= 2.0:
            raise ValueError(
                "repetition_penalty must be in (0, 2], got "
                f"{self.repetition_penalty}."
            )
        if not 0 <= self.min_new_tokens:
            raise ValueError(
                f"min_new_tokens must be in (0, max_new_tokens], got "
                f"{self.min_new_tokens}."
            )
        if self.num_reasoning_penalty_steps is not None and self.num_reasoning_penalty_steps < 0:
            raise ValueError(
                f"num_reasoning_penalty_steps must be non-negative, got {self.num_reasoning_penalty_steps}."
            )
        if self.max_new_tokens is not None:
            if self.max_new_tokens < 0:
                raise ValueError(
                    f"max_new_tokens must be at least 0, got {self.max_new_tokens}."
                )
            if not self.min_new_tokens <= self.max_new_tokens:
                raise ValueError(
                    f"min_new_tokens must be in (0, max_new_tokens({self.max_new_tokens})], got "
                    f"{self.min_new_tokens}."
                )
        grammars = [
            self.json_schema,
            self.regex,
            self.ebnf,
        ]  # since mutually exclusive, only one can be set
        if sum(x is not None for x in grammars) > 1:
            raise ValueError("Only one of regex, json_schema, or ebnf can be set.")

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
                if tokenizer is not None:
                    stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                    stop_str_max_len = max(stop_str_max_len, len(stop_str_ids))
                else:
                    stop_str_max_len = max(stop_str_max_len, len(stop_str))
            self.stop_str_max_len = stop_str_max_len

        # Process stop reasoning
        #print('stop_reasoning_strs: ', self.stop_reasoning_strs)
        #print('stop_reasoning_token_ids: ', self.stop_reasoning_token_ids)
        #assert False, 'test'
        
        if len(self.stop_reasoning_strs) > 0:
            for stop_str in self.stop_reasoning_strs:
                if tokenizer is not None:
                    stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)[0]
                    print('stop_str_ids: ', stop_str_ids)
                    self.stop_reasoning_token_ids.add(stop_str_ids)
                else:
                    raise ValueError("tokenizer is required for stop_reasoning_strs")
        
