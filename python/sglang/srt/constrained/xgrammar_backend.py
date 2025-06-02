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
"""Constrained decoding with xgrammar backend."""

import json
import logging
from typing import List, Optional, Tuple, Union

import torch
from xgrammar import (
    CompiledGrammar,
    GrammarCompiler,
    GrammarMatcher,
    StructuralTagItem,
    TokenizerInfo,
    allocate_token_bitmask,
)

from sglang.srt.constrained.base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
    BaseGrammarBackend,
    BaseGrammarObject,
)
from sglang.srt.constrained.triton_ops.bitmask_ops import (
    apply_token_bitmask_inplace_triton,
)

logger = logging.getLogger(__name__)


MAX_ROLLBACK_TOKENS = 200


class XGrammarGrammar(BaseGrammarObject):

    def __init__(
        self,
        matcher: GrammarMatcher,
        vocab_size: int,
        ctx: CompiledGrammar,
        override_stop_tokens: Optional[Union[List[int], int]],
        key_string: Optional[str] = None,  # TODO (sk): for debugging, remove later
    ) -> None:
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx
        self.override_stop_tokens = override_stop_tokens
        self.finished = False
        self.accepted_tokens = []
        self.key_string = key_string

    def accept_token(self, token: int):
        if not self.is_terminated():
            accepted = self.matcher.accept_token(token)
            if not accepted:
                # log for debugging
                raise ValueError(
                    f"Tokens not accepted: {token}\n"
                    f"Accepted tokens: {self.accepted_tokens}\n"
                    f"Key string: {self.key_string}"
                )
            else:
                self.accepted_tokens.append(token)

    def rollback(self, k: int):
        self.matcher.rollback(k)
        self.accepted_tokens = self.accepted_tokens[:-k]

    def is_terminated(self):
        return self.matcher.is_terminated()

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return allocate_token_bitmask(batch_size, vocab_size)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(vocab_mask, idx)

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    def apply_vocab_mask(self, logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        if logits.device.type == "cuda":
            apply_token_bitmask_inplace_triton(logits, vocab_mask)
        elif logits.device.type == "cpu" and self.apply_vocab_mask_cpu:
            self.apply_vocab_mask_cpu(logits, vocab_mask)
        else:
            raise RuntimeError(f"Unsupported device: {logits.device.type}")

    def copy(self):
        matcher = GrammarMatcher(
            self.ctx,
            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
            override_stop_tokens=self.override_stop_tokens,
        )
        return XGrammarGrammar(
            matcher,
            self.vocab_size,
            self.ctx,
            self.override_stop_tokens,
            self.key_string,
        )

    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        s = self.matcher.find_jump_forward_string()
        if s:
            return [], s
        return None

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        _, data = helper
        return data, -1

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        k = 0
        for i, old_id in enumerate(old_output_ids):
            if old_id == new_output_ids[i]:
                k = i + 1
            else:
                break

        # rollback to the last token that is the same
        if k < len(old_output_ids):
            self.matcher.rollback(len(old_output_ids) - k)

        for i in range(k, len(new_output_ids)):
            assert self.matcher.accept_token(new_output_ids[i])

    def __repr__(self):
        return f"XGrammarGrammar({self.key_string=}, {self.accepted_tokens=})"


class XGrammarGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
    ):
        super().__init__()

        if True:
            tokenizer_info = TokenizerInfo.from_huggingface(
                tokenizer, vocab_size=vocab_size
            )
            override_stop_tokens = None

        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)
        self.vocab_size = vocab_size
        self.override_stop_tokens = override_stop_tokens

    def _from_context(self, ctx: CompiledGrammar, key_string: str) -> XGrammarGrammar:
        matcher = GrammarMatcher(
            ctx,
            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
            override_stop_tokens=self.override_stop_tokens,
        )
        return XGrammarGrammar(
            matcher, self.vocab_size, ctx, self.override_stop_tokens, key_string
        )

    def dispatch_json(self, key_string: str) -> Optional[XGrammarGrammar]:
        try:
            if key_string == "$$ANY$$":
                # Note: This builtin JSON grammar includes *all* valid JSON (including, for example, arrays at the root)
                ctx = self.grammar_compiler.compile_builtin_json_grammar()
            else:
                ctx = self.grammar_compiler.compile_json_schema(schema=key_string)

        except (RuntimeError, json.decoder.JSONDecodeError) as e:
            logging.error(f"Hit invalid json_schema: {key_string=}, {e=}")
            return INVALID_GRAMMAR_OBJ
        return self._from_context(ctx, key_string)

    def dispatch_ebnf(self, key_string: str) -> Optional[XGrammarGrammar]:
        try:
            ctx = self.grammar_compiler.compile_grammar(key_string)
        except RuntimeError as e:
            logging.error(f"Hit invalid ebnf: {key_string=}, {e=}")
            return INVALID_GRAMMAR_OBJ
        return self._from_context(ctx, key_string)

    def dispatch_regex(self, key_string: str) -> Optional[XGrammarGrammar]:
        try:
            ctx = self.grammar_compiler.compile_regex(key_string)
        except RuntimeError as e:
            logging.error(f"Hit invalid regex: {key_string=}, {e=}")
            return INVALID_GRAMMAR_OBJ
        return self._from_context(ctx, key_string)

    def dispatch_structural_tag(self, key_string: str) -> Optional[XGrammarGrammar]:
        try:
            structural_tag = json.loads(key_string)
            tags = [
                StructuralTagItem(
                    begin=structure["begin"],
                    schema=json.dumps(structure["schema"]),
                    end=structure["end"],
                )
                for structure in structural_tag["structures"]
            ]
            ctx = self.grammar_compiler.compile_structural_tag(
                tags, structural_tag["triggers"]
            )
        except (RuntimeError, json.decoder.JSONDecodeError) as e:
            logging.error(f"Hit invalid structural_tag: {key_string=}, {e=}")
            return INVALID_GRAMMAR_OBJ
        return self._from_context(ctx, key_string)

    def reset(self):
        self.grammar_compiler.clear_cache()
