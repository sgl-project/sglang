# Copyright 2026 SGLang Team
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
"""SynthID Text hashing, configuration, and request state.

The hash and tournament transform are adapted from the Apache-2.0 reference
implementation at google-deepmind/synthid-text commit
addb4a158143c7c6851a1308f78b89fceed59683.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Iterable, Optional, Sequence

import torch

from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

_HASH_MULTIPLIER = 6364136223846793005
_HASH_INCREMENT = 1
_HASH_ROUNDS = 12
_HASH_SHIFT = 5
_G_VALUE_SHIFT = 30
_INT64_MASK = (1 << 64) - 1
_INT64_SIGN_BIT = 1 << 63

# Public reference keys from google-deepmind/synthid-text. These are suitable
# for compatibility tests and benchmarks, not for a production watermark.
REFERENCE_SYNTHID_KEYS = (
    654,
    400,
    836,
    123,
    340,
    443,
    597,
    160,
    57,
    29,
    590,
    639,
    13,
    715,
    468,
    990,
    966,
    226,
    324,
    585,
    118,
    504,
    421,
    521,
    129,
    669,
    732,
    225,
    90,
    960,
)


def _wrap_int64(value: int) -> int:
    value &= _INT64_MASK
    return value - (1 << 64) if value >= _INT64_SIGN_BIT else value


def _accumulate_hash_python(current_hash: int, value: int) -> int:
    return _wrap_int64(
        _wrap_int64(current_hash + int(value)) * _HASH_MULTIPLIER + _HASH_INCREMENT
    )


def _hash_context_python(context: Sequence[int], hash_iv: int) -> int:
    result = hash_iv
    for token_id in context:
        result = _accumulate_hash_python(result, token_id)
    return result


def compute_synthid_key_iv(keys: Sequence[int]) -> int:
    """Compute the reference SHA-256 initialization vector for integer keys."""
    if not keys:
        raise ValueError("SynthID requires at least one key.")
    if len(set(keys)) != len(keys):
        raise ValueError("SynthID layer keys must be distinct.")
    try:
        key_bytes = struct.pack(f"<{len(keys)}q", *(int(key) for key in keys))
    except (OverflowError, struct.error) as exc:
        raise ValueError("SynthID keys must be signed 64-bit integers.") from exc
    digest = hashlib.sha256(key_bytes).digest()
    return int.from_bytes(digest, byteorder="big") % ((1 << 63) - 1)


@lru_cache(maxsize=None)
def load_synthid_keys(path: str) -> tuple[int, ...]:
    """Load a JSON key list without placing secret key material in CLI args."""
    key_path = Path(path).expanduser()
    try:
        payload = json.loads(key_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read SynthID key file {key_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"SynthID key file {key_path} must contain valid JSON: {exc}"
        ) from exc

    if isinstance(payload, dict):
        payload = payload.get("keys")
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            f"SynthID key file {key_path} must contain a non-empty JSON list "
            'or an object with a non-empty "keys" list.'
        )
    if any(isinstance(key, bool) or not isinstance(key, int) for key in payload):
        raise ValueError(f"SynthID keys in {key_path} must all be integers.")

    keys = tuple(int(key) for key in payload)
    compute_synthid_key_iv(keys)
    return keys


@dataclass(frozen=True)
class SynthIDConfig:
    keys: tuple[int, ...] = field(repr=False)
    ngram_len: int = 5
    top_k: int = 1024
    context_history_size: int = 1024
    skip_first_ngram_calls: bool = False
    hash_iv: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hash_iv", compute_synthid_key_iv(self.keys))

    @property
    def context_width(self) -> int:
        return self.ngram_len - 1


def get_synthid_config(exec_config=None) -> Optional[SynthIDConfig]:
    exec_config = get_exec() if exec_config is None else exec_config
    kernel = exec_config.kernel
    if kernel.synthid_watermark_key_file is None:
        return None
    keys = load_synthid_keys(kernel.synthid_watermark_key_file)
    return SynthIDConfig(
        keys=keys,
        ngram_len=kernel.synthid_watermark_ngram_len,
        top_k=kernel.synthid_watermark_top_k,
        context_history_size=kernel.synthid_watermark_context_history_size,
        skip_first_ngram_calls=kernel.synthid_watermark_skip_first_ngram_calls,
    )


def accumulate_hash_torch(
    current_hash: torch.Tensor, data: torch.Tensor | int
) -> torch.Tensor:
    """Apply the reference signed-int64 hash step with two's-complement wrap."""
    return (current_hash + data) * _HASH_MULTIPLIER + _HASH_INCREMENT


def compute_context_hash_torch(contexts: torch.Tensor, hash_iv: int) -> torch.Tensor:
    """Hash the final dimension of an int64 context tensor."""
    result = torch.full(
        contexts.shape[:-1],
        hash_iv,
        dtype=torch.int64,
        device=contexts.device,
    )
    for index in range(contexts.shape[-1]):
        result = accumulate_hash_torch(result, contexts[..., index])
    return result


def compute_g_values_torch(
    context_hashes: torch.Tensor,
    token_ids: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """Return SynthID binary g-values with shape ``[rows, candidates, depth]``."""
    ngram_keys = accumulate_hash_torch(
        context_hashes[:, None, None],
        token_ids.to(torch.int64)[:, :, None],
    )
    ngram_keys = accumulate_hash_torch(ngram_keys, keys[None, None, :])
    for _ in range(_HASH_ROUNDS):
        ngram_keys = torch.bitwise_right_shift(
            accumulate_hash_torch(ngram_keys, 1), _HASH_SHIFT
        )
    return torch.remainder(torch.bitwise_right_shift(ngram_keys, _G_VALUE_SHIFT), 2)


def watermark_topk_scores_torch(
    scores: torch.Tensor,
    token_ids: torch.Tensor,
    context_hashes: torch.Tensor,
    keys: torch.Tensor,
    apply_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the reference two-leaf SynthID tournament to sparse top-k scores."""
    probs = torch.softmax(scores, dim=-1)
    g_values = compute_g_values_torch(context_hashes, token_ids, keys)
    for depth in range(keys.numel()):
        g_values_at_depth = g_values[:, :, depth]
        g_mass = (g_values_at_depth * probs).sum(dim=-1, keepdim=True)
        probs = probs * (1 + g_values_at_depth - g_mass)

    watermarked_scores = torch.log(probs)
    watermarked_scores = torch.where(
        torch.isfinite(watermarked_scores),
        watermarked_scores,
        torch.full_like(watermarked_scores, -1e12),
    )
    return torch.where(apply_mask[:, None], watermarked_scores, scores)


@dataclass
class SynthIDState:
    """Batch metadata used to initialize stable request-slot state."""

    _BATCH_FIELDS: ClassVar[tuple[str, ...]] = (
        "slot_indices",
        "generations",
        "initial_context",
        "initial_context_history",
        "initial_history_cursor",
        "initial_num_calls",
        "commit_mask",
    )

    slot_indices: torch.Tensor
    generations: torch.Tensor
    initial_context: torch.Tensor
    initial_context_history: torch.Tensor
    initial_history_cursor: torch.Tensor
    initial_num_calls: torch.Tensor
    commit_mask: torch.Tensor
    pending_init: Optional[torch.Tensor]
    config: SynthIDConfig
    request_enabled: Optional[torch.Tensor] = None
    request_enabled_host: Optional[list[bool]] = None

    @classmethod
    def from_requests(
        cls,
        *,
        reqs: Sequence[Req],
        config: SynthIDConfig,
        device: str | torch.device,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        req_generation: torch.Tensor,
        chunked_req: Optional[Req],
        pin_memory: bool,
    ) -> SynthIDState:
        generations = req_generation[req_pool_indices_cpu.to(dtype=torch.int64)].to(
            device, non_blocking=True
        )
        return cls.from_output_ids(
            (req.output_ids for req in reqs),
            config=config,
            device=device,
            slot_indices=req_pool_indices.to(dtype=torch.int64),
            generations=generations,
            commit_mask=torch.tensor(
                [req is not chunked_req for req in reqs],
                dtype=torch.bool,
                pin_memory=pin_memory,
            ).to(device, non_blocking=True),
        )

    @classmethod
    def for_initialized_slots(
        cls,
        slot_indices: torch.Tensor,
        *,
        config: SynthIDConfig,
    ) -> SynthIDState:
        """Create a graph-capture view over state already initialized in the pool."""
        batch_size = slot_indices.shape[0]
        device = slot_indices.device

        def empty(*shape):
            return torch.empty(shape, dtype=torch.int64, device=device)

        return cls(
            slot_indices=slot_indices,
            generations=empty(batch_size),
            initial_context=empty(batch_size, config.context_width),
            initial_context_history=empty(batch_size, config.context_history_size),
            initial_history_cursor=empty(batch_size),
            initial_num_calls=empty(batch_size),
            commit_mask=torch.ones(batch_size, dtype=torch.bool, device=device),
            pending_init=None,
            config=config,
        )

    @classmethod
    def from_output_ids(
        cls,
        output_ids: Iterable[Sequence[int]],
        *,
        config: SynthIDConfig,
        device: str | torch.device,
        slot_indices: torch.Tensor,
        generations: torch.Tensor,
        commit_mask: Optional[torch.Tensor] = None,
    ) -> SynthIDState:
        output_rows = [list(row) for row in output_ids]
        batch_size = len(output_rows)
        context_width = config.context_width
        history_size = config.context_history_size

        contexts: list[list[int]] = []
        histories: list[list[int]] = []
        cursors: list[int] = []
        call_counts: list[int] = []
        for tokens in output_rows:
            context = [0] * context_width
            history = [0] * history_size
            cursor = 0
            num_calls = 0
            for token_id in tokens:
                call_number = num_calls + 1
                if not config.skip_first_ngram_calls or call_number >= config.ngram_len:
                    history[cursor] = _hash_context_python(context, config.hash_iv)
                    cursor = (cursor + 1) % history_size
                context = context[1:] + [int(token_id)]
                num_calls = call_number
            contexts.append(context)
            histories.append(history)
            cursors.append(cursor)
            call_counts.append(num_calls)

        def as_tensor(values, *shape):
            return torch.tensor(values, dtype=torch.int64, device=device).reshape(
                batch_size, *shape
            )

        if commit_mask is not None and commit_mask.shape != (batch_size,):
            raise ValueError("commit_mask must have one value per request.")

        return cls(
            slot_indices=slot_indices.to(device=device, dtype=torch.int64),
            generations=generations.to(device=device, dtype=torch.int64),
            initial_context=as_tensor(contexts, context_width),
            initial_context_history=as_tensor(histories, history_size),
            initial_history_cursor=as_tensor(cursors),
            initial_num_calls=as_tensor(call_counts),
            commit_mask=(
                torch.ones(batch_size, dtype=torch.bool, device=device)
                if commit_mask is None
                else commit_mask.to(device=device, dtype=torch.bool)
            ),
            pending_init=torch.ones(batch_size, dtype=torch.bool, device=device),
            config=config,
        )

    def __len__(self) -> int:
        return self.slot_indices.shape[0]

    def filter_batch(
        self,
        keep_indices_device: torch.Tensor,
        keep_indices: Optional[list[int]] = None,
    ) -> None:
        for field_name in self._BATCH_FIELDS:
            setattr(self, field_name, getattr(self, field_name)[keep_indices_device])
        if self.pending_init is not None:
            self.pending_init = self.pending_init[keep_indices_device]
        if self.request_enabled is not None:
            self.request_enabled = self.request_enabled[keep_indices_device]
            indices = (
                keep_indices
                if keep_indices is not None
                else keep_indices_device.cpu().tolist()
            )
            self.request_enabled_host = [self.request_enabled_host[i] for i in indices]

    def merge_batch(self, other: SynthIDState) -> None:
        if self.config != other.config:
            raise ValueError("Cannot merge incompatible SynthID watermark states.")
        left_len = len(self)
        if self.request_enabled is not None or other.request_enabled is not None:
            self.request_enabled_host = (
                self.request_enabled_host
                if self.request_enabled_host is not None
                else [True] * left_len
            ) + (
                other.request_enabled_host
                if other.request_enabled_host is not None
                else [True] * len(other)
            )
            self.request_enabled = torch.cat(
                (
                    (
                        self.request_enabled
                        if self.request_enabled is not None
                        else torch.ones(
                            left_len, dtype=torch.bool, device=self.slot_indices.device
                        )
                    ),
                    (
                        other.request_enabled
                        if other.request_enabled is not None
                        else torch.ones(
                            len(other),
                            dtype=torch.bool,
                            device=other.slot_indices.device,
                        )
                    ),
                )
            )
        for field_name in self._BATCH_FIELDS:
            setattr(
                self,
                field_name,
                torch.cat((getattr(self, field_name), getattr(other, field_name))),
            )
        if self.pending_init is None and other.pending_init is None:
            return

        def pending(state, length):
            if state.pending_init is not None:
                return state.pending_init
            return torch.zeros(
                length, dtype=torch.bool, device=state.slot_indices.device
            )

        self.pending_init = torch.cat(
            (pending(self, left_len), pending(other, len(other)))
        )


class SynthIDStatePool:
    """Stable GPU state indexed by request-pool slot and slot generation."""

    def __init__(
        self,
        config: SynthIDConfig,
        num_slots: int,
        device: str | torch.device,
    ):
        self.config = config
        self.context = torch.zeros(
            (num_slots, config.context_width), dtype=torch.int64, device=device
        )
        self.context_history = torch.zeros(
            (num_slots, config.context_history_size),
            dtype=torch.int64,
            device=device,
        )
        self.history_cursor = torch.zeros(num_slots, dtype=torch.int64, device=device)
        self.num_calls = torch.zeros(num_slots, dtype=torch.int64, device=device)
        self.generations = torch.full(
            (num_slots,), -1, dtype=torch.int64, device=device
        )

    def initialize_pending(self, state: SynthIDState) -> None:
        pending = state.pending_init
        if pending is None:
            return
        needs_init = pending & (
            self.generations[state.slot_indices] != state.generations
        )
        slots = state.slot_indices[needs_init]
        self.context[slots] = state.initial_context[needs_init]
        self.context_history[slots] = state.initial_context_history[needs_init]
        self.history_cursor[slots] = state.initial_history_cursor[needs_init]
        self.num_calls[slots] = state.initial_num_calls[needs_init]
        self.generations[slots] = state.generations[needs_init]
        state.pending_init = None

    def _active(
        self, state: SynthIDState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.initialize_pending(state)
        slots = state.slot_indices
        return (
            self.context[slots],
            self.context_history[slots],
            self.history_cursor[slots],
            self.num_calls[slots],
        )

    def _eligible(self, num_calls: torch.Tensor) -> torch.Tensor:
        if self.config.skip_first_ngram_calls:
            return num_calls + 1 >= self.config.ngram_len
        return torch.ones_like(num_calls, dtype=torch.bool)

    def _hashes_and_mask(self, context, history, num_calls):
        context_hashes = compute_context_hash_torch(context, self.config.hash_iv)
        repeated = torch.any(history == context_hashes[:, None], dim=1)
        return context_hashes, self._eligible(num_calls) & ~repeated

    def current_hashes_and_mask(
        self,
        state: SynthIDState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context, history, _, num_calls = self._active(state)
        return self._hashes_and_mask(context, history, num_calls)

    def project_linear_speculative(
        self,
        state: SynthIDState,
        verify_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project exact bounded-history state for a linear verify block."""
        if verify_tokens.ndim != 2 or verify_tokens.shape[0] != len(state):
            raise ValueError(
                "verify_tokens must have shape [batch, verify_width] matching "
                f"SynthID state batch size {len(state)}, got "
                f"{tuple(verify_tokens.shape)}."
            )

        context, history, history_cursor, num_calls = self._active(state)
        context = context.clone()
        history = history.clone()
        history_cursor = history_cursor.clone()
        num_calls = num_calls.clone()
        hashes_per_step = []
        masks_per_step = []

        for step in range(verify_tokens.shape[1]):
            context_hashes, apply_mask = self._hashes_and_mask(
                context, history, num_calls
            )
            eligible = self._eligible(num_calls)
            hashes_per_step.append(context_hashes)
            masks_per_step.append(apply_mask)

            write_cols = history_cursor[:, None]
            previous_hashes = history.gather(1, write_cols).view(-1)
            history.scatter_(
                1,
                write_cols,
                torch.where(eligible, context_hashes, previous_hashes)[:, None],
            )
            history_cursor = torch.where(
                eligible,
                (history_cursor + 1) % self.config.context_history_size,
                history_cursor,
            )

            if step + 1 < verify_tokens.shape[1]:
                context = torch.cat(
                    (context[:, 1:], verify_tokens[:, step + 1 : step + 2]), dim=1
                )
                num_calls = num_calls + 1

        return (
            torch.stack(hashes_per_step, dim=1).reshape(-1),
            torch.stack(masks_per_step, dim=1).reshape(-1),
        )

    def commit(
        self,
        state: SynthIDState,
        token_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> None:
        """Commit generated tokens to stable request slots."""
        self.initialize_pending(state)
        if token_ids.ndim == 1:
            token_ids = token_ids[:, None]
        if token_ids.ndim != 2 or token_ids.shape[0] != len(state):
            raise ValueError(
                "token_ids must have shape [batch] or [batch, width] matching "
                f"SynthID state batch size {len(state)}, got {tuple(token_ids.shape)}."
            )
        token_ids = token_ids.to(
            device=self.context.device, dtype=torch.int64, non_blocking=True
        )
        if lengths is None:
            lengths = torch.full(
                (token_ids.shape[0],),
                token_ids.shape[1],
                dtype=torch.int64,
                device=self.context.device,
            )
        else:
            lengths = lengths.to(
                device=self.context.device, dtype=torch.int64, non_blocking=True
            )
        lengths = torch.where(state.commit_mask, lengths, 0)

        slots = state.slot_indices
        context = self.context[slots]
        history_cursor = self.history_cursor[slots]
        num_calls = self.num_calls[slots]

        for step in range(token_ids.shape[1]):
            committed = step < lengths
            context_hashes = compute_context_hash_torch(context, self.config.hash_iv)
            record = committed & self._eligible(num_calls)

            write_slots = slots[record]
            write_cols = history_cursor[record]
            self.context_history[write_slots, write_cols] = context_hashes[record]
            history_cursor[record] = (write_cols + 1) % self.config.context_history_size

            shifted = torch.cat((context[:, 1:], token_ids[:, step : step + 1]), dim=1)
            context = torch.where(committed[:, None], shifted, context)
            num_calls = num_calls + committed.to(torch.int64)

        self.context[slots] = context
        self.history_cursor[slots] = history_cursor
        self.num_calls[slots] = num_calls
