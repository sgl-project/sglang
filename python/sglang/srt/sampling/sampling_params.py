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
import math
from typing import Dict, List, Optional, Set, Union

import msgspec

# sre_parse is deprecated in Python 3.11+, use re._parser instead
try:
    import re._parser as sre_parse
except ImportError:
    import sre_parse  # Python < 3.11

# JSON-safe value types for custom_params.  Must survive msgpack IPC
# without PickleWrapper.  After deserialization on the scheduler side,
# Req.__init__ injects "__req__" (a Req object) into the dict in-process;
# that augmented dict is never re-serialized.
_JsonScalar = Union[None, bool, int, float, str]
CustomParamValue = Union[
    _JsonScalar,
    List[_JsonScalar],
    Dict[str, _JsonScalar],
]

_SAMPLING_EPS = 1e-6
TOP_K_ALL = 1 << 30

logger = logging.getLogger(__name__)


def raise_if_tokenizer_required(
    tokenizer, stop_strs, stop_regex_strs, min_new_tokens=0
):
    """Raise ValueError if tokenizer-dependent features are used without a tokenizer.

    String-based stop conditions (stop_strs, stop_regex_strs) require tokenizer.decode()
    to convert output token IDs to text for matching. min_new_tokens requires the
    tokenizer's eos_token_id to penalize. When skip_tokenizer_init=True, these cannot
    be used.
    """
    if tokenizer is not None:
        return

    if stop_strs:
        raise ValueError(
            f"stop={stop_strs!r} is unavailable when skip_tokenizer_init=True "
            "(requires tokenizer to decode tokens to text for matching)."
        )
    if stop_regex_strs:
        raise ValueError(
            f"stop_regex={stop_regex_strs!r} is unavailable when skip_tokenizer_init=True "
            "(requires tokenizer to decode tokens to text for matching)."
        )
    if min_new_tokens > 0:
        raise ValueError(
            f"min_new_tokens={min_new_tokens} is unavailable when skip_tokenizer_init=True "
            "(requires tokenizer for eos_token_id)."
        )


class SamplingParams(msgspec.Struct, kw_only=True, omit_defaults=True):
    """
    The sampling parameters.

    See docs_new/docs/basic_usage/sampling_params.mdx
    for the documentation.
    """

    # --- API parameters (set by callers) ---
    max_new_tokens: Optional[int] = 128
    stop: Optional[Union[str, List[str]]] = (
        None  # API input alias, copied to stop_strs then cleared in normalize()
    )
    stop_token_ids: Optional[Set[int]] = None
    stop_regex: Optional[Union[str, List[str]]] = (
        None  # API input alias, copied to stop_regex_strs then cleared in normalize()
    )
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = TOP_K_ALL
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_new_tokens: int = 0
    n: int = 1
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    structural_tag: Optional[str] = None
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    no_stop_trim: bool = False
    custom_params: Optional[Dict[str, CustomParamValue]] = None
    stream_interval: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    sampling_seed: Optional[int] = None

    # --- Internal fields (populated by __post_init__ or normalize(), not API-facing) ---
    stop_strs: Optional[Union[str, List[str]]] = None  # from stop
    stop_regex_strs: Optional[Union[str, List[str]]] = None  # from stop_regex
    stop_str_max_len: int = 0  # set by normalize()
    stop_regex_max_len: int = 0  # set by normalize()
    is_normalized: bool = False  # set by normalize()

    def __post_init__(self):
        # For non-optional params, treat None as "use default" so that callers
        # (e.g. /generate) can pass null without crashing verify().

        # msgspec calls __post_init__ after deserialization. Once normalize()
        # has populated tokenizer-derived fields, avoid resetting them.
        if self.is_normalized:
            return

        self.stop_strs = self.stop
        if self.stop_token_ids:
            filtered = {int(t) for t in self.stop_token_ids if t is not None}
            self.stop_token_ids = filtered or None
        else:
            self.stop_token_ids = None
        self.stop_regex_strs = self.stop_regex
        self.temperature = self.temperature if self.temperature is not None else 1.0
        self.top_p = self.top_p if self.top_p is not None else 1.0
        self.top_k = self.top_k if self.top_k is not None else -1
        self.min_p = self.min_p if self.min_p is not None else 0.0
        self.frequency_penalty = (
            self.frequency_penalty if self.frequency_penalty is not None else 0.0
        )
        self.presence_penalty = (
            self.presence_penalty if self.presence_penalty is not None else 0.0
        )
        self.repetition_penalty = (
            self.repetition_penalty if self.repetition_penalty is not None else 1.0
        )
        self.min_new_tokens = (
            self.min_new_tokens if self.min_new_tokens is not None else 0
        )
        self.n = self.n if self.n is not None else 1
        self.ignore_eos = self.ignore_eos if self.ignore_eos is not None else False
        self.skip_special_tokens = (
            self.skip_special_tokens if self.skip_special_tokens is not None else True
        )
        self.spaces_between_special_tokens = (
            self.spaces_between_special_tokens
            if self.spaces_between_special_tokens is not None
            else True
        )
        self.no_stop_trim = (
            self.no_stop_trim if self.no_stop_trim is not None else False
        )

        # Process some special cases
        if 0 <= self.temperature < _SAMPLING_EPS:
            # top_k = 1 means greedy sampling
            self.temperature = 1.0
            self.top_k = 1
        if self.top_k == -1:
            self.top_k = TOP_K_ALL  # whole vocabulary

    def verify(self, vocab_size):
        if not math.isfinite(self.temperature) or self.temperature < 0.0:
            raise ValueError(
                f"temperature must be a non-negative finite number, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.top_k < 1:
            raise ValueError(
                f"top_k must be at least 1, got {self.top_k}. "
                f"Note: top_k=-1 is also accepted as input, which disables top_k filtering "
                f"(equivalent to top_k={TOP_K_ALL} internally)."
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

        # Validate tokenizer is available for tokenizer-dependent features
        raise_if_tokenizer_required(
            tokenizer, self.stop_strs, self.stop_regex_strs, self.min_new_tokens
        )

        # Clear API input aliases so omit_defaults=True drops them from the wire.
        self.stop = None
        self.stop_regex = None
        self.is_normalized = True


# This function gets a strict upperbound on the maximum number of tokens that would need
# to be buffered to match the input regex string
# NOTE: in the worst case, one character that needs to be buffered corresponds to one
# token
def get_max_seq_length(regex_str: str):
    return _max_length_from_subpattern(sre_parse.parse(regex_str))


MAX_LEN = 2**30


def _max_length_from_subpattern(subpattern: sre_parse.SubPattern):
    total = 0
    for token, value in subpattern:
        if token in {
            sre_parse.LITERAL,  # `value` is any one character
            sre_parse.IN,  # Any character within `value`
            sre_parse.ANY,  # "."
        }:
            total += 1
        elif token == sre_parse.SUBPATTERN:
            # EG: (a\d+) ->
            # [(SUBPATTERN,
            #   (1, 0, 0, [(LITERAL, 97),
            #              (MAX_REPEAT, (1, MAXREPEAT, [(IN, [(CATEGORY, CATEGORY_DIGIT)])]))]))]
            _, _, _, inner_subpattern = value
            total += _max_length_from_subpattern(inner_subpattern)
        elif token == sre_parse.BRANCH:
            _, branches = value
            total += max(_max_length_from_subpattern(branch) for branch in branches)
        elif token in {sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT}:
            _, max_num_repeat, inner_subpattern = value
            if max_num_repeat == sre_parse.MAXREPEAT:
                total += MAX_LEN
            else:
                total += max_num_repeat * _max_length_from_subpattern(inner_subpattern)
        elif token == sre_parse.AT:
            # These are zero-width assertions like ^, $, and \b that don't add to the max
            # length
            total += 0
        else:
            logger.warning(f"Got unhandled regex token: {token}")

            total += MAX_LEN

    return total
