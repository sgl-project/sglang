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
"""Constrained decoding with llguidance backend."""

import json
import os
from typing import List, Optional, Tuple

import llguidance
import llguidance.hf
import llguidance.torch
import torch
from llguidance.gbnf_to_lark import any_to_lark

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
)


class GuidanceGrammar(BaseGrammarObject):
    def __init__(
        self, llguidance_tokenizer: llguidance.LLTokenizer, serialized_grammar: str
    ):
        super().__init__()
        self.llguidance_tokenizer = llguidance_tokenizer
        self.serialized_grammar = serialized_grammar

        # TODO: add support for fast-forward tokens in the future
        self.ll_interpreter = llguidance.LLInterpreter(
            self.llguidance_tokenizer,
            self.serialized_grammar,
            enable_backtrack=False,
            enable_ff_tokens=False,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )
        self.pending_ff_tokens: list[int] = []
        self.finished = False
        self.bitmask = None

    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        if len(self.pending_ff_tokens) > 0:
            s = self.llguidance_tokenizer.decode_str(self.pending_ff_tokens)
            ff_tokens = self.pending_ff_tokens
            self.pending_ff_tokens = []
            return (ff_tokens, s)

        return None

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        return "", -1

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        pass

    def accept_token(self, token: int):
        backtrack, ff_tokens = self.ll_interpreter.commit_token(token)
        if len(ff_tokens) > 0 and backtrack == 0:
            # first token is last generated token
            ff_tokens = ff_tokens[1:]
            self.pending_ff_tokens.extend(ff_tokens)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        if len(self.pending_ff_tokens) > 0:
            # if we have pending fast-forward tokens,
            # just return them immediately
            ff_token = self.pending_ff_tokens.pop(0)
            vocab_mask[idx, :] = 0
            vocab_mask[idx, ff_token // 32] = 1 << (ff_token % 32)
            return

        if self.ll_interpreter.has_pending_stop():
            self.finished = True

        llguidance.torch.fill_next_token_bitmask(self.ll_interpreter, vocab_mask, idx)

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        if self.bitmask is None or self.bitmask.shape[0] < batch_size:
            # only create bitmask when batch gets larger
            self.bitmask = llguidance.torch.allocate_token_bitmask(
                batch_size, self.llguidance_tokenizer.vocab_size
            )
            bitmask = self.bitmask
        else:
            bitmask = self.bitmask[:batch_size]

        return bitmask

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        llguidance.torch.apply_token_bitmask_inplace(logits, vocab_mask)

    def copy(self):
        return GuidanceGrammar(
            llguidance_tokenizer=self.llguidance_tokenizer,
            serialized_grammar=self.serialized_grammar,
        )


class GuidanceBackend(BaseGrammarBackend):
    def __init__(self, tokenizer, whitespace_pattern: Optional[str] = None):
        super().__init__()

        self.tokenizer = tokenizer
        self.whitespace_flexible = (
            True if whitespace_pattern == "whitespace_flexible" else False
        )
        self.llguidance_tokenizer = llguidance.hf.from_tokenizer(self.tokenizer, None)

    def _from_serialized(self, serialized_grammar) -> GuidanceGrammar:
        return GuidanceGrammar(
            llguidance_tokenizer=self.llguidance_tokenizer,
            serialized_grammar=serialized_grammar,
        )

    def dispatch_json(self, key_string: str) -> GuidanceGrammar:
        json_schema = key_string
        compiler = llguidance.JsonCompiler(whitespace_flexible=self.whitespace_flexible)
        serialized_grammar = compiler.compile(json_schema)
        return self._from_serialized(serialized_grammar)

    def dispatch_regex(self, key_string: str) -> GuidanceGrammar:
        compiler = llguidance.RegexCompiler()
        serialized_grammar = compiler.compile(regex=key_string)
        return self._from_serialized(serialized_grammar)

    def dispatch_ebnf(self, key_string: str) -> GuidanceGrammar:
        compiler = llguidance.LarkCompiler()
        serialized_grammar = compiler.compile(any_to_lark(key_string))
        return self._from_serialized(serialized_grammar)

    def dispatch_structural_tag(self, key_string: str):
        return super().dispatch_structural_tag(key_string)
