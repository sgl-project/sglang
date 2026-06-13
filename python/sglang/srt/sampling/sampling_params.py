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

import logging
from typing import Any, Dict, List, Optional, Union

# sre_parse is deprecated in Python 3.11+, use re._parser instead
try:
    import re._parser as sre_parse
except ImportError:
    import sre_parse  # Python < 3.11

_SAMPLING_EPS = 1e-6
TOP_K_ALL = 1 << 30

logger = logging.getLogger(__name__)


class SamplingParams:
    """
    The sampling parameters.

    See docs/backend/sampling_params.md or
    https://docs.sglang.io/backend/sampling_params.html
    for the documentation.
    """

    def __init__(
        self,
        max_new_tokens: int = 128,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        stop_regex: Optional[Union[str, List[str]]] = None,
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
        custom_params: Optional[Dict[str, Any]] = None,
        stream_interval: Optional[int] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        sampling_seed: Optional[int] = None,
    ) -> None:
        # For non-optional params, treat None as "use default" so that callers
        # (e.g. /generate) can pass null without crashing verify().
        self.max_new_tokens = max_new_tokens
        self.stop_strs = stop
        if stop_token_ids:
            filtered = {int(t) for t in stop_token_ids if t is not None}
            self.stop_token_ids = filtered or None
        else:
            self.stop_token_ids = None
        self.stop_regex_strs = stop_regex
        self.temperature = temperature if temperature is not None else 1.0
        self.top_p = top_p if top_p is not None else 1.0
        self.top_k = top_k if top_k is not None else -1
        self.min_p = min_p if min_p is not None else 0.0
        self.frequency_penalty = (
            frequency_penalty if frequency_penalty is not None else 0.0
        )
        self.presence_penalty = (
            presence_penalty if presence_penalty is not None else 0.0
        )
        self.repetition_penalty = (
            repetition_penalty if repetition_penalty is not None else 1.0
        )
        self.min_new_tokens = min_new_tokens if min_new_tokens is not None else 0
        self.regex = regex
        self.n = n if n is not None else 1
        self.json_schema = json_schema
        self.ebnf = ebnf
        self.structural_tag = structural_tag
        self.ignore_eos = ignore_eos if ignore_eos is not None else False
        self.skip_special_tokens = (
            skip_special_tokens if skip_special_tokens is not None else True
        )
        self.spaces_between_special_tokens = (
            spaces_between_special_tokens
            if spaces_between_special_tokens is not None
            else True
        )
        self.no_stop_trim = no_stop_trim if no_stop_trim is not None else False
        self.custom_params = custom_params
        self.stream_interval = stream_interval
        self.logit_bias = logit_bias
        self.sampling_seed = sampling_seed

        # Process some special cases
        if 0 <= self.temperature < _SAMPLING_EPS:
            # top_k = 1 means greedy sampling
            self.temperature = 1.0
            self.top_k = 1
        if self.top_k == -1:
            self.top_k = TOP_K_ALL  # whole vocabulary

    def verify(self, vocab_size):
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.top_k < 1 or self.top_k == -1:
            raise ValueError(
                f"top_k must be -1 (disable) or at least 1, got {self.top_k}."
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
        if not 0.0 < self.repetition_penalty <= 2.0:
            raise ValueError(
                "repetition_penalty must be in (0, 2] (1.0 = no penalty), "
                f"got {self.repetition_penalty}."
            )
        if not 0 <= self.min_new_tokens:
            raise ValueError(
                f"min_new_tokens must be in [0, max_new_tokens], got "
                f"{self.min_new_tokens}."
            )
        if self.max_new_tokens is not None:
            if self.max_new_tokens < 0:
                raise ValueError(
                    f"max_new_tokens must be at least 0, got {self.max_new_tokens}."
                )
            if not self.min_new_tokens <= self.max_new_tokens:
                raise ValueError(
                    f"min_new_tokens must be in [0, max_new_tokens({self.max_new_tokens})], got "
                    f"{self.min_new_tokens}."
                )
        if self.logit_bias is not None:
            for token_id in self.logit_bias:
                if not 0 <= int(token_id) < vocab_size:
                    raise ValueError(
                        f"logit_bias must has keys in [0, {vocab_size - 1}], got "
                        f"{token_id}."
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

        # Process stop regex strings
        if self.stop_regex_strs is None:
            self.stop_regex_strs = []
            self.stop_regex_max_len = 0
        else:
            if isinstance(self.stop_regex_strs, str):
                self.stop_regex_strs = [self.stop_regex_strs]

            stop_regex_max_len = 0
            for stop_regex in self.stop_regex_strs:
                stop_regex_max_len = max(
                    stop_regex_max_len, get_max_seq_length(stop_regex)
                )

            self.stop_regex_max_len = stop_regex_max_len


# This function gets a strict upperbound on the maximum number of tokens that would need
# to be buffered to match the input regex string
# NOTE: in the worst case, one character that needs to be buffered corresponds to one
# token
def get_max_seq_length(regex_str: str):
    return _max_length_from_subpattern(sre_parse.parse(regex_str))


MAX_LEN = 2**30

# Hoist token-kind sets and other ``sre_parse`` constants out of the inner loop:
# the original ``token in {sre_parse.LITERAL, ...}`` literal rebuilt the set on
# every iteration because its members are attribute lookups, not compile-time
# constants. Hoisting them makes the membership test a single C-level hash
# lookup against a pre-built ``frozenset``.
#
# ``LITERAL``    -- ``value`` is any one character
# ``IN``         -- Any character within ``value``
# ``ANY``        -- "."
_SINGLE_CHAR_TOKENS = frozenset({sre_parse.LITERAL, sre_parse.IN, sre_parse.ANY})
_REPEAT_TOKENS = frozenset({sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT})
_SUBPATTERN_TOKEN = sre_parse.SUBPATTERN
_BRANCH_TOKEN = sre_parse.BRANCH
_AT_TOKEN = sre_parse.AT
_MAXREPEAT = sre_parse.MAXREPEAT


def _max_length_from_subpattern(subpattern: sre_parse.SubPattern):
    total = 0
    for token, value in subpattern:
        if token in _SINGLE_CHAR_TOKENS:
            total += 1
        elif token == _SUBPATTERN_TOKEN:
            # EG: (a\d+) ->
            # [(SUBPATTERN,
            #   (1, 0, 0, [(LITERAL, 97),
            #              (MAX_REPEAT, (1, MAXREPEAT, [(IN, [(CATEGORY, CATEGORY_DIGIT)])]))]))]
            _, _, _, inner_subpattern = value
            total += _max_length_from_subpattern(inner_subpattern)
        elif token == _BRANCH_TOKEN:
            _, branches = value
            total += max(_max_length_from_subpattern(branch) for branch in branches)
        elif token in _REPEAT_TOKENS:
            _, max_num_repeat, inner_subpattern = value
            if max_num_repeat == _MAXREPEAT:
                # Unbounded repeat saturates the upper bound; nothing else in
                # the pattern can lower it below ``MAX_LEN``.
                return MAX_LEN
            total += max_num_repeat * _max_length_from_subpattern(inner_subpattern)
        elif token == _AT_TOKEN:
            # Zero-width assertions like ^, $, and \b that don't add to the max
            # length.
            continue
        else:
            logger.warning(f"Got unhandled regex token: {token}")
            return MAX_LEN

        # Once the bound has saturated, further additions can't change the
        # downstream behaviour (callers compare against ``MAX_LEN`` and use
        # the result as a buffer size capped by ``len(output_ids)``).
        if total >= MAX_LEN:
            return MAX_LEN

    return total
