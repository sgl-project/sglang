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
from threading import Lock
from typing import Dict, List, Optional, Tuple, Union

import msgspec
import torch
from xgrammar import (
    BatchGrammarMatcher,
    CompiledGrammar,
    GrammarCompiler,
    GrammarMatcher,
    StructuralTag,
    StructuralTagItem,
    TokenizerInfo,
    bitmask_dtype,
    get_bitmask_shape,
)

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    GrammarRow,
    GrammarStats,
    InvalidGrammarObject,
)
from sglang.srt.constrained.json_schema_validation import (
    UnsupportedJSONSchemaFeature,
    validate_xgrammar_json_schema,
)
from sglang.srt.constrained.utils import is_legacy_structural_tag
from sglang.srt.function_call.glm4_moe_detector import glm47_thinking_grammar_prefix
from sglang.srt.utils import is_hip
from sglang.srt.utils.common import is_pin_memory_available

_is_hip = is_hip()

if _is_hip:
    from sgl_kernel import apply_token_bitmask_inplace_cuda
else:
    from sglang.kernels.ops.grammar.bitmask_ops import (
        apply_token_bitmask_inplace_triton,
    )

from sglang.kernels.ops.grammar.token_filter_ops import set_token_filter_triton
from sglang.srt.constrained.torch_ops.token_filter_torch_ops import (
    set_token_filter_torch,
)

logger = logging.getLogger(__name__)
MAX_ROLLBACK_TOKENS = 200
XGRAMMAR_BATCH_MIN_SIZE = 16
XGRAMMAR_BATCH_MAX_THREADS = 4


def _allocate_token_bitmask(vocab_size: int, batch_size: int) -> torch.Tensor:
    # Pin where pinning exists, so the later H2D can be a genuine non_blocking
    # copy (a pageable source silently downgrades it).  MPS torch has no
    # pin-memory kernel and asserts on pin_memory=True.
    return torch.full(
        get_bitmask_shape(batch_size, vocab_size),
        -1,
        dtype=bitmask_dtype,
        pin_memory=is_pin_memory_available(),
    )


class XGrammarGrammar(BaseGrammarObject):
    def __init__(
        self,
        matcher: GrammarMatcher,
        vocab_size: int,
        ctx: CompiledGrammar,
        override_stop_tokens: Optional[Union[List[int], int]],
        key_string: Optional[str] = None,
        grammar_stats: Optional[GrammarStats] = GrammarStats(),
        batch_matcher: Optional[BatchGrammarMatcher] = None,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx
        self.override_stop_tokens = override_stop_tokens
        self.accepted_tokens = []
        self.key_string = key_string
        self.grammar_stats = grammar_stats
        self.batch_matcher = batch_matcher

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
        return _allocate_token_bitmask(vocab_size, batch_size)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(vocab_mask, idx)

    def fill_vocab_mask_batched(
        self, entries: List[GrammarRow], vocab_mask: torch.Tensor
    ) -> None:
        if self.batch_matcher is None or len(entries) < XGRAMMAR_BATCH_MIN_SIZE:
            return super().fill_vocab_mask_batched(entries, vocab_mask)
        matchers = []
        indices = []
        for entry in entries:
            if isinstance(entry.grammar, XGrammarGrammar):
                matchers.append(entry.grammar.matcher)
                indices.append(entry.row)
            else:
                entry.grammar.fill_vocab_mask(vocab_mask, entry.row)
        if matchers:
            self.batch_matcher.batch_fill_next_token_bitmask(
                matchers, vocab_mask, indices
            )

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    def apply_vocab_mask(self, logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        if logits.device.type in {"cuda", "xpu", "musa"}:
            if _is_hip:
                apply_token_bitmask_inplace_cuda(logits, vocab_mask)
            else:
                apply_token_bitmask_inplace_triton(logits, vocab_mask)
        elif logits.device.type == "npu":
            import sgl_kernel_npu  # noqa: F401

            torch.ops.npu.apply_token_bitmask(logits, vocab_mask)
        elif logits.device.type == "cpu":
            # Used by the MLX backend, which builds its additive mask rows
            # on the CPU before inserting them into the lazy graph.
            from xgrammar import apply_token_bitmask_inplace

            apply_token_bitmask_inplace(logits, vocab_mask, backend="cpu")
        else:
            raise RuntimeError(f"Unsupported device: {logits.device.type}")

    def copy(self):
        matcher = GrammarMatcher(
            self.ctx,
            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
            override_stop_tokens=self.override_stop_tokens,
        )
        if grammar_stats := self.grammar_stats:
            grammar_stats = dataclasses.replace(
                grammar_stats, is_cache_hit=True, tree_traversal_time=[]
            )
        return XGrammarGrammar(
            matcher,
            self.vocab_size,
            self.ctx,
            self.override_stop_tokens,
            self.key_string,
            grammar_stats,
            self.batch_matcher,
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
            if not self.matcher.accept_token(new_output_ids[i]):
                raise ValueError(
                    f"Token not accepted during retokenization: {new_output_ids[i]} "
                    f"at position {i}\n"
                    f"Old output IDs: {old_output_ids}\n"
                    f"New output IDs: {new_output_ids}\n"
                    f"Key string: {self.key_string}"
                )

    def __repr__(self):
        return f"XGrammarGrammar({self.key_string=}, {self.accepted_tokens=}, {self.current_token=})"


class TokenizerNotSupportedError(Exception):
    """Raised when tokenizer is not supported by XGrammar backend."""

    pass


class ThinkingMetadata(msgspec.Struct, frozen=True):
    mask: torch.Tensor
    safe_tokens: bytes
    think_end_ids: frozenset[int]
    generation_mask_words: tuple[int, ...]


class XGrammarThinkingGrammar(XGrammarGrammar):
    def __init__(
        self,
        full: XGrammarGrammar,
        generation: XGrammarGrammar,
        thinking_mask: torch.Tensor,
        safe_tokens: bytes,
        think_end_ids: frozenset[int],
        thinking_mask_overrides: tuple[tuple[int, int], ...] = (),
    ):
        super().__init__(
            full.matcher,
            full.vocab_size,
            full.ctx,
            full.override_stop_tokens,
            full.key_string,
            full.grammar_stats,
            full.batch_matcher,
        )
        self.full_matcher = full.matcher
        self.generation = generation
        self.thinking_mask = thinking_mask
        self.safe_tokens = safe_tokens
        self.think_end_ids = think_end_ids
        self.thinking_mask_overrides = thinking_mask_overrides
        self.phase = "thinking"
        self.accept_modes: List[str] = []

    def is_terminated(self):
        return self.phase != "thinking" and super().is_terminated()

    def accept_token(self, token: int):
        if self.is_terminated():
            return
        if self.phase == "thinking":
            if token in self.think_end_ids:
                self.matcher = self.generation.matcher
                self.phase = "generation"
                mode = "boundary"
            elif 0 <= token < len(self.safe_tokens) and self.safe_tokens[token]:
                mode = "thinking"
            else:
                super().accept_token(token)
                self.phase = "fallback"
                self.accept_modes.append("fallback")
                return
            self.current_token = token
            self.accepted_tokens.append(token)
            self.accept_modes.append(mode)
        else:
            super().accept_token(token)
            self.accept_modes.append(self.phase)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        if self.phase == "thinking":
            vocab_mask[idx].copy_(self.thinking_mask[0])
            for word, value in self.thinking_mask_overrides:
                vocab_mask[idx, word] = value
        else:
            super().fill_vocab_mask(vocab_mask, idx)

    def fill_vocab_mask_batched(
        self, entries: List[GrammarRow], vocab_mask: torch.Tensor
    ) -> None:
        remaining = []
        for entry in entries:
            grammar = entry.grammar
            if (
                isinstance(grammar, XGrammarThinkingGrammar)
                and grammar.phase == "thinking"
            ):
                grammar.fill_vocab_mask(vocab_mask, entry.row)
            else:
                remaining.append(entry)
        super().fill_vocab_mask_batched(remaining, vocab_mask)

    def rollback(self, k: int):
        if k == 0:
            return
        if not 0 <= k <= len(self.accept_modes):
            raise ValueError(f"Cannot roll back {k} of {len(self.accept_modes)} tokens")
        modes = self.accept_modes[-k:]
        for matcher, mode in (
            (self.full_matcher, "fallback"),
            (self.generation.matcher, "generation"),
        ):
            count = modes.count(mode)
            if count:
                matcher.rollback(count)
        del self.accept_modes[-k:]
        del self.accepted_tokens[-k:]
        last = self.accept_modes[-1] if self.accept_modes else "thinking"
        self.phase = "generation" if last == "boundary" else last
        self.matcher = (
            self.generation.matcher if self.phase == "generation" else self.full_matcher
        )

    def copy(self):
        return XGrammarThinkingGrammar(
            super().copy(),
            self.generation.copy(),
            self.thinking_mask,
            self.safe_tokens,
            self.think_end_ids,
            self.thinking_mask_overrides,
        )

    def try_jump_forward(self, tokenizer):
        if self.phase == "thinking":
            return None
        return super().try_jump_forward(tokenizer)

    def jump_and_retokenize(self, old_output_ids, new_output_ids, next_state):
        common = 0
        for old, new in zip(old_output_ids, new_output_ids):
            if old != new:
                break
            common += 1
        self.rollback(len(old_output_ids) - common)
        for token in new_output_ids[common:]:
            self.accept_token(token)


class XGrammarGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
        model_eos_token_ids: Optional[List[int]] = None,
        any_whitespace: bool = True,
        max_whitespace_cnt: Optional[int] = None,
    ):
        super().__init__()

        if hasattr(tokenizer, "init_xgrammar"):
            # For special tokenizer
            tokenizer_info, override_stop_tokens = tokenizer.init_xgrammar()

            if tokenizer_info is None:
                # Not supported tokenizer
                raise TokenizerNotSupportedError(
                    f"Tokenizer type {type(tokenizer).__name__} is not supported by XGrammar"
                )
        else:
            # Create TokenizerInfo with model's EOS tokens as the authoritative stop tokens
            # This ensures consistency between what the model considers EOS and what XGrammar uses
            try:
                tokenizer_info = TokenizerInfo.from_huggingface(
                    tokenizer, vocab_size=vocab_size, stop_token_ids=model_eos_token_ids
                )
                override_stop_tokens = None
            except Exception as e:
                raise TokenizerNotSupportedError(
                    f"Failed to create XGrammar TokenizerInfo from tokenizer: {e}"
                )

        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)
        self.tokenizer_info = tokenizer_info
        # The automatic pool scales with host CPUs, even for small decode batches.
        self.batch_matcher = BatchGrammarMatcher(max_threads=XGRAMMAR_BATCH_MAX_THREADS)
        self.vocab_size = vocab_size
        self.override_stop_tokens = override_stop_tokens
        self.any_whitespace = any_whitespace
        self.max_whitespace_cnt = max_whitespace_cnt
        self._thinking_metadata: Optional[ThinkingMetadata] = None
        self._thinking_metadata_lock = Lock()

    @property
    def is_support_token_filter(self):
        return True

    @staticmethod
    def allocate_vocab_mask(vocab_size: int, batch_size: int, device) -> torch.Tensor:
        return _allocate_token_bitmask(vocab_size, batch_size)

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        if logits.device.type in {"cuda", "xpu", "musa"}:
            if _is_hip:
                apply_token_bitmask_inplace_cuda(logits, vocab_mask)
            else:
                apply_token_bitmask_inplace_triton(logits, vocab_mask)
        elif logits.device.type == "npu":
            import sgl_kernel_npu  # noqa: F401

            torch.ops.npu.apply_token_bitmask(logits, vocab_mask)
        else:
            raise RuntimeError(f"Unsupported device: {logits.device.type}")

    @staticmethod
    def set_token_filter(
        vocab_mask: torch.Tensor,
        token_ids: List[int],
        batch_idx: int,
        is_allowed: bool = True,
        reset_vocab_mask: bool = True,
    ):
        if _is_hip or (vocab_mask.device.type != "cuda"):
            set_token_filter_torch(
                vocab_mask,
                token_ids,
                batch_idx,
                is_allowed=is_allowed,
                reset_vocab_mask=reset_vocab_mask,
            )
        else:
            set_token_filter_triton(
                vocab_mask,
                token_ids,
                batch_idx,
                is_allowed=is_allowed,
                reset_vocab_mask=reset_vocab_mask,
            )

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
            self.batch_matcher,
        )

    def dispatch_json(self, key_string: str) -> BaseGrammarObject:
        try:
            if key_string == "$$ANY$$":
                # Note: This builtin JSON grammar includes *all* valid JSON (including, for example, arrays at the root)
                ctx = self.grammar_compiler.compile_builtin_json_grammar()
            else:
                # Inspect the decoded schema for semantic loss, then give the
                # original string to XGrammar to preserve its parser behavior.
                schema = json.loads(key_string)
                validate_xgrammar_json_schema(schema)
                ctx = self.grammar_compiler.compile_json_schema(
                    schema=key_string,
                    any_whitespace=self.any_whitespace,
                    max_whitespace_cnt=self.max_whitespace_cnt,
                )

        except (
            RuntimeError,
            UnsupportedJSONSchemaFeature,
            json.decoder.JSONDecodeError,
            UnicodeDecodeError,
        ) as e:
            logger.error(f"Hit invalid json_schema: {key_string=}, {e=}")
            return InvalidGrammarObject(str(e))
        return self._from_context(ctx, key_string, GrammarStats(dispatch_type="json"))

    def dispatch_ebnf(self, key_string: str) -> BaseGrammarObject:
        try:
            ctx = self.grammar_compiler.compile_grammar(key_string)
        except RuntimeError as e:
            logger.error(f"Hit invalid ebnf: {key_string=}, {e=}")
            return InvalidGrammarObject(str(e))
        return self._from_context(ctx, key_string, GrammarStats(dispatch_type="ebnf"))

    def wrap_full_assistant_grammar(
        self, grammar: BaseGrammarObject, key_string: str
    ) -> BaseGrammarObject:
        if self.tokenizer_info.add_prefix_space or not key_string.startswith(
            glm47_thinking_grammar_prefix()
        ):
            return grammar
        with self._thinking_metadata_lock:
            if self._thinking_metadata is None:
                self._thinking_metadata = self._build_thinking_metadata(grammar)
            metadata = self._thinking_metadata
        if not metadata.think_end_ids:
            return grammar
        overrides = ()
        if metadata.generation_mask_words:
            mask = grammar.allocate_vocab_mask(self.vocab_size, 1, "cpu")
            grammar.fill_vocab_mask(mask, 0)
            overrides = tuple(
                (word, int(mask[0, word]))
                for word in metadata.generation_mask_words
                if mask[0, word] != metadata.mask[0, word]
            )
        # Only thinking metadata is shared; generation still encodes the tool schema.
        ctx = self.grammar_compiler.compile_grammar(
            key_string, root_rule_name="generation"
        )
        generation = self._from_context(
            ctx, key_string, GrammarStats(dispatch_type="ebnf")
        )
        return XGrammarThinkingGrammar(
            grammar,
            generation,
            metadata.mask,
            metadata.safe_tokens,
            metadata.think_end_ids,
            overrides,
        )

    def _build_thinking_metadata(self, grammar: BaseGrammarObject) -> ThinkingMetadata:
        mask = grammar.allocate_vocab_mask(self.vocab_size, 1, "cpu")
        grammar.fill_vocab_mask(mask, 0)
        words = mask[0].tolist()
        safe_tokens = bytearray(self.vocab_size)
        think_end_ids = set()
        generation_mask_words = set()
        for token_id, value in enumerate(self.tokenizer_info.decoded_vocab):
            if token_id >= self.vocab_size:
                continue
            # Tokens spanning the boundary can depend on the generation suffix.
            if value.partition(b"</think>")[2]:
                generation_mask_words.add(token_id // 32)
            if not (words[token_id // 32] >> (token_id % 32)) & 1:
                continue
            if value == b"</think>":
                think_end_ids.add(token_id)
            if not value or b"<" in value:
                continue
            try:
                value.decode("utf-8")
            except UnicodeDecodeError:
                continue
            # Complete UTF-8 without '<' returns this exclusion DFA to its root.
            # Delimiter prefixes and partial codepoints retain the full matcher.
            safe_tokens[token_id] = 1
        return ThinkingMetadata(
            mask,
            bytes(safe_tokens),
            frozenset(think_end_ids),
            tuple(sorted(generation_mask_words)),
        )

    def dispatch_regex(self, key_string: str) -> BaseGrammarObject:
        try:
            ctx = self.grammar_compiler.compile_regex(key_string)
        except RuntimeError as e:
            logger.error(f"Hit invalid regex: {key_string=}, {e=}")
            return InvalidGrammarObject(str(e))
        return self._from_context(ctx, key_string, GrammarStats(dispatch_type="regex"))

    def dispatch_structural_tag(self, key_string: str) -> BaseGrammarObject:
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
                new_tag = StructuralTag.from_legacy_structural_tag(
                    tags, structural_tag["triggers"]
                )
                new_tag.format.at_least_one = structural_tag.get("at_least_one", False)
                ctx = self.grammar_compiler.compile_structural_tag(new_tag)
            else:
                format_dict = structural_tag.get("format")
                if isinstance(format_dict, dict):
                    self._sanitize_structural_format(format_dict)
                    structural_tag["format"] = format_dict
                    key_string = json.dumps(structural_tag)
                ctx = self.grammar_compiler.compile_structural_tag(key_string)
        except (RuntimeError, json.decoder.JSONDecodeError) as e:
            logger.error(f"Hit invalid structural_tag: {key_string=}, {e=}")
            return InvalidGrammarObject(str(e))
        return self._from_context(
            ctx, key_string, GrammarStats(dispatch_type="structural_tag")
        )

    def reset(self):
        super().reset()
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
