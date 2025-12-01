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

import dataclasses
import json
import logging
from typing import Dict, List, Optional, Tuple, Union

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
    GrammarStats,
)
from sglang.srt.constrained.utils import is_legacy_structural_tag
from sglang.srt.utils import is_hip

_is_hip = is_hip()
if _is_hip:
    from sgl_kernel import apply_token_bitmask_inplace_cuda
else:
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
        grammar_stats: Optional[GrammarStats] = GrammarStats(),
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx
        self.override_stop_tokens = override_stop_tokens
        self.accepted_tokens = []
        self.key_string = key_string
        self.grammar_stats = grammar_stats

    def accept_token(self, token: int):
        if not self.is_terminated():
            self.current_token = token
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
        if (
            logits.device.type == "cuda"
            or logits.device.type == "npu"
            or logits.device.type == "xpu"
        ):
            if _is_hip:
                apply_token_bitmask_inplace_cuda(logits, vocab_mask)
            else:
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
            dataclasses.replace(
                self.grammar_stats, is_cache_hit=True, tree_traversal_time=[]
            ),
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
        return f"XGrammarGrammar({self.key_string=}, {self.accepted_tokens=}, {self.current_token=})"


class XGrammarGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
        model_eos_token_ids: Optional[List[int]] = None,
        any_whitespace: bool = True,
    ):
        super().__init__()

        if hasattr(tokenizer, "init_xgrammar"):
            # For special tokenizer
            tokenizer_info, override_stop_tokens = tokenizer.init_xgrammar()

            if tokenizer_info is None:
                # Not supported tokenizer
                return
        else:
            # Create TokenizerInfo with model's EOS tokens as the authoritative stop tokens
            # This ensures consistency between what the model considers EOS and what XGrammar uses
            tokenizer_info = TokenizerInfo.from_huggingface(
                tokenizer, vocab_size=vocab_size, stop_token_ids=model_eos_token_ids
            )
            override_stop_tokens = None

        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)
        self.vocab_size = vocab_size
        self.override_stop_tokens = override_stop_tokens
        self.any_whitespace = any_whitespace

    @staticmethod
    def _sanitize_structural_format(structural_format):
        """Recursively replace missing json_schema fields with an empty schema."""
        if not isinstance(structural_format, dict):
            return

        fmt_type = structural_format.get("type")
        if fmt_type in {"json_schema", "qwen_xml_parameter"}:
            if structural_format.get("json_schema") is None:
                structural_format["json_schema"] = {}

        if fmt_type == "tag":
            XGrammarGrammarBackend._sanitize_structural_format(
                structural_format.get("content")
            )
        elif fmt_type in {"sequence", "or"}:
            for element in structural_format.get("elements", []):
                XGrammarGrammarBackend._sanitize_structural_format(element)
        elif fmt_type in {"triggered_tags", "tags_with_separator"}:
            for tag in structural_format.get("tags", []):
                XGrammarGrammarBackend._sanitize_structural_format(tag)

    @staticmethod
    def _sanitize_structural_tag_structures(structural_tag: Dict) -> None:
        for structure in structural_tag.get("structures", []):
            if structure.get("schema") is None:
                structure["schema"] = {}

    def _from_context(
        self, ctx: CompiledGrammar, key_string: str, grammar_stats: GrammarStats
    ) -> XGrammarGrammar:
        matcher = GrammarMatcher(
            ctx,
            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
            override_stop_tokens=self.override_stop_tokens,
        )
        return XGrammarGrammar(
            matcher,
            self.vocab_size,
            ctx,
            self.override_stop_tokens,
            key_string,
            grammar_stats,
        )

    def dispatch_json(self, key_string: str) -> Optional[XGrammarGrammar]:
        try:
            if key_string == "$$ANY$$":
                # Note: This builtin JSON grammar includes *all* valid JSON (including, for example, arrays at the root)
                ctx = self.grammar_compiler.compile_builtin_json_grammar()
            else:
                ctx = self.grammar_compiler.compile_json_schema(
                    schema=key_string, any_whitespace=self.any_whitespace
                )

        except (RuntimeError, json.decoder.JSONDecodeError) as e:
            logging.error(f"Hit invalid json_schema: {key_string=}, {e=}")
            return INVALID_GRAMMAR_OBJ
        return self._from_context(ctx, key_string, GrammarStats(dispatch_type="json"))

    def dispatch_ebnf(self, key_string: str) -> Optional[XGrammarGrammar]:
        try:
            ctx = self.grammar_compiler.compile_grammar(key_string)
        except RuntimeError as e:
            logging.error(f"Hit invalid ebnf: {key_string=}, {e=}")
            return INVALID_GRAMMAR_OBJ
        return self._from_context(ctx, key_string, GrammarStats(dispatch_type="ebnf"))

    def dispatch_regex(self, key_string: str) -> Optional[XGrammarGrammar]:
        try:
            ctx = self.grammar_compiler.compile_regex(key_string)
        except RuntimeError as e:
            logging.error(f"Hit invalid regex: {key_string=}, {e=}")
            return INVALID_GRAMMAR_OBJ
        return self._from_context(ctx, key_string, GrammarStats(dispatch_type="regex"))

    def dispatch_structural_tag(self, key_string: str) -> Optional[XGrammarGrammar]:
        try:
            # TODO(dark): it's REALLY stupid to construct object from string and decode it again
            structural_tag = json.loads(key_string)
            if is_legacy_structural_tag(structural_tag):
                self._sanitize_structural_tag_structures(structural_tag)
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
            else:
                format_dict = structural_tag.get("format")
                if isinstance(format_dict, dict):
                    self._sanitize_structural_format(format_dict)
                    structural_tag["format"] = format_dict
                    key_string = json.dumps(structural_tag)
                ctx = self.grammar_compiler.compile_structural_tag(key_string)
        except (RuntimeError, json.decoder.JSONDecodeError) as e:
            logging.error(f"Hit invalid structural_tag: {key_string=}, {e=}")
            return INVALID_GRAMMAR_OBJ
        return self._from_context(
            ctx, key_string, GrammarStats(dispatch_type="structural_tag")
        )

    def reset(self):
        self.grammar_compiler.clear_cache()


def demo_test():
    from transformers import AutoConfig, AutoTokenizer

    from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME_FOR_TEST)
    hf_config = AutoConfig.from_pretrained(DEFAULT_MODEL_NAME_FOR_TEST)

    # Should use vocab size from model config
    vocab_size = hf_config.vocab_size
    eos_token_id = tokenizer.eos_token_id

    backend = XGrammarGrammarBackend(
        tokenizer, vocab_size=vocab_size, model_eos_token_ids=[eos_token_id]
    )
    regex = r"hello (world|there)"
    grammar = backend.dispatch_regex(regex)
    tokens = [
        tokenizer.encode(t, add_special_tokens=False)[0] for t in ["hello", " world"]
    ]

    # Test termination
    grammar.accept_token(tokens[0])  # accept "hello"
    grammar.accept_token(tokens[1])  # accept " world"
    grammar.accept_token(eos_token_id)  # accept EOS
    assert grammar.is_terminated()

    # Test rollback the terminated state
    grammar.rollback(1)
    assert not grammar.is_terminated()


if __name__ == "__main__":
    demo_test()
