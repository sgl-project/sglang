# Copyright 2023-2026 SGLang Team
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

"""Breakable CUDA graph helpers for context-parallel prefill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from sglang.srt.arg_groups.overrides import (
    resolved_view,
    resolving_view,
)
from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.layers.cp.padding import get_cp_padding_align_size
from sglang.srt.layers.cp.utils import (
    cp_gather_after_forward,
    cp_split_before_forward,
    enable_cp_v2,
    prepare_cp_forward,
)
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )
    from sglang.srt.server_args import ServerArgs


def supports_prefill_cp_bcg(server_args: ServerArgs) -> bool:
    """Return whether the selected prefill-CP configuration supports BCG."""
    from sglang.srt.arg_groups.overrides import attention_backends_of

    cfg = resolving_view(server_args)
    resolved = resolved_view(server_args)
    prefill_attention_backend, _ = attention_backends_of(resolved_view(server_args))
    return (
        cfg.enable_prefill_cp
        and resolved.attn_cp_size == cfg.tp_size
        and cfg.cp_strategy == "zigzag"
        and prefill_attention_backend == "trtllm_mha"
    )


def enable_cp_v2_bcg_capture(server_args: ServerArgs) -> bool:
    """Return whether CP-v2 breakable prefill capture is enabled."""
    return enable_cp_v2() and supports_prefill_cp_bcg(server_args)


def filter_prefill_cp_bcg_capture_num_tokens(
    capture_num_tokens: list[int], server_args: ServerArgs
) -> list[int]:
    """Keep only token buckets where the zigzag CP strategy can run."""
    min_num_tokens = resolved_view(server_args).attn_cp_size * 2
    filtered = [size for size in capture_num_tokens if size >= min_num_tokens]
    if not filtered:
        raise ValueError(
            "Prefill CP breakable CUDA graph requires at least one token bucket "
            f">= {min_num_tokens}, but got {capture_num_tokens}."
        )
    return filtered


def _slice_output_rows(output: Any, num_tokens: int) -> Any:
    if output is None:
        return None
    if torch.is_tensor(output) or isinstance(output, PPProxyTensors):
        return output[:num_tokens]
    if isinstance(output, tuple):
        return tuple(_slice_output_rows(item, num_tokens) for item in output)
    if isinstance(output, list):
        return [_slice_output_rows(item, num_tokens) for item in output]
    raise TypeError(f"Unsupported prefill CP BCG output: {type(output)}")


@dataclass
class PrefillCPBCGInput:
    """Fixed-address CP-local inputs and per-bucket replay state."""

    input_embeds: torch.Tensor
    positions: torch.Tensor
    bucket_local_tokens: Dict[int, int] = field(default_factory=dict)
    live_local_tokens: int = 0

    @classmethod
    def create(cls, runner: PrefillCudaGraphRunner) -> PrefillCPBCGInput:
        with torch.device(runner.device):
            return cls(
                input_embeds=torch.zeros(
                    (
                        runner.max_num_tokens,
                        runner.model_runner.model_config.hidden_size,
                    ),
                    dtype=runner.model_runner.dtype,
                ),
                positions=torch.zeros(
                    (runner.max_num_tokens,),
                    dtype=torch.int64,
                ),
            )

    def required_local_tokens(self, extend_seq_lens: Any) -> Optional[int]:
        """Return the aligned CP-local rows required by a live zigzag layout."""
        strategy = get_cp_strategy()
        if not isinstance(strategy, ZigzagCPStrategy) or extend_seq_lens is None:
            return None

        cp_segment_num = strategy.cp_size * 2
        per_rank_logical_tokens = [0] * strategy.cp_size
        for raw_length in extend_seq_lens:
            base, remainder = divmod(int(raw_length), cp_segment_num)
            for rank in range(strategy.cp_size):
                opposite_rank = cp_segment_num - 1 - rank
                per_rank_logical_tokens[rank] += (
                    base * 2 + int(rank < remainder) + int(opposite_rank < remainder)
                )
        align_size = get_cp_padding_align_size()
        return (
            (max(per_rank_logical_tokens) + align_size - 1) // align_size * align_size
        )

    def select_replay_bucket(
        self,
        *,
        num_tokens: int,
        required_local_tokens: int,
        capture_num_tokens: list[int],
        max_padding_factor: int,
    ) -> Optional[int]:
        """Return the smallest global capture whose CP-local rows fit."""
        max_num_tokens = num_tokens * max_padding_factor
        for bucket in capture_num_tokens:
            if bucket < num_tokens:
                continue
            if bucket > max_num_tokens:
                break
            captured_local_tokens = self.bucket_local_tokens.get(bucket)
            if (
                captured_local_tokens is not None
                and required_local_tokens <= captured_local_tokens
            ):
                return bucket
        return None

    def select_replay_bucket_for_batch(
        self,
        *,
        num_tokens: int,
        extend_seq_lens: Any,
        capture_num_tokens: list[int],
        max_padding_factor: int,
    ) -> Optional[int]:
        required_local_tokens = self.required_local_tokens(extend_seq_lens)
        if required_local_tokens is None:
            return None
        return self.select_replay_bucket(
            num_tokens=num_tokens,
            required_local_tokens=required_local_tokens,
            capture_num_tokens=capture_num_tokens,
            max_padding_factor=max_padding_factor,
        )

    def prepare(
        self,
        runner: PrefillCudaGraphRunner,
        forward_batch: ForwardBatch,
        *,
        static_num_tokens: int,
        capture: bool,
    ) -> None:
        """Shard global prefill inputs into fixed-address CP-local buffers."""
        # Replay batches may reuse a ForwardBatch object whose metadata was
        # built for a different request layout. Always rebuild before sharding.
        forward_batch.attn_cp_metadata = None
        prepare_cp_forward(forward_batch)

        captured_local_tokens = None
        if not capture:
            try:
                captured_local_tokens = self.bucket_local_tokens[static_num_tokens]
            except KeyError as exc:
                raise RuntimeError(
                    "Missing CP-local capture capacity for global prefill bucket "
                    f"{static_num_tokens}"
                ) from exc

            # Breakable graph segments retain their captured row geometry when
            # a smaller live batch reuses the same global token bucket.
            metadata = forward_batch.attn_cp_metadata
            if hasattr(metadata, "per_rank_actual_token"):
                live_physical_tokens = max(metadata.per_rank_actual_token)
                if live_physical_tokens > captured_local_tokens:
                    raise RuntimeError(
                        f"Live batch needs {live_physical_tokens} local CP rows, "
                        f"but global prefill bucket {static_num_tokens} has "
                        f"captured capacity {captured_local_tokens}"
                    )
                cp_size = len(metadata.per_rank_actual_token)
                metadata.per_rank_actual_token = [captured_local_tokens] * cp_size
                metadata.max_rank_len = [captured_local_tokens] * cp_size

        raw_tokens = int(forward_batch.extend_num_tokens)
        global_input_ids = forward_batch.input_ids[:raw_tokens]
        global_positions = forward_batch.positions[:raw_tokens]
        global_input_embeds = runner.model_runner.model.get_input_embeddings()(
            global_input_ids
        )
        local_input_embeds, local_positions = cp_split_before_forward(
            global_input_embeds,
            global_positions,
            forward_batch,
        )
        live_local_tokens = int(local_input_embeds.shape[0])

        if capture:
            captured_local_tokens = live_local_tokens
            self.bucket_local_tokens[static_num_tokens] = captured_local_tokens
        else:
            assert captured_local_tokens is not None
            if live_local_tokens > captured_local_tokens:
                raise RuntimeError(
                    f"Live batch needs {live_local_tokens} local CP rows, but global "
                    f"prefill bucket {static_num_tokens} has captured capacity "
                    f"{captured_local_tokens}"
                )

        if captured_local_tokens > self.input_embeds.shape[0]:
            raise RuntimeError(
                f"CP-local capture needs {captured_local_tokens} rows, but the "
                f"fixed input buffer has capacity {self.input_embeds.shape[0]}"
            )

        input_embeds = self.input_embeds[:captured_local_tokens]
        positions = self.positions[:captured_local_tokens]
        input_embeds.zero_()
        positions.zero_()
        input_embeds[:live_local_tokens].copy_(local_input_embeds)
        positions[:live_local_tokens].copy_(local_positions)
        forward_batch.input_embeds = input_embeds
        forward_batch.positions = positions
        self.live_local_tokens = live_local_tokens


def execute_prefill_cp_bcg(
    runner: PrefillCudaGraphRunner,
    forward_batch: ForwardBatch,
    static_forward_batch: ForwardBatch,
    static_num_tokens: int,
    raw_num_tokens: int,
    **kwargs,
):
    """Replay a CP-local body and run the global gather/logits tail eagerly."""
    # Importing a runner submodule while server arguments are being resolved
    # executes runner/__init__.py, which imports this module through the prefill
    # runner. Defer the runtime-only dependency until runner initialization has
    # completed to keep the server-argument compatibility check acyclic.
    from sglang.srt.model_executor.runner.shape_key import ShapeKey

    cp_input = runner.prefill_cp_bcg_input
    assert cp_input is not None
    model = runner.model_runner.model
    with runner._prefill_forward_context(
        static_forward_batch,
        num_tokens=static_num_tokens,
        raw_num_tokens=raw_num_tokens,
    ):
        local_output = runner.backend.replay(
            ShapeKey(size=static_num_tokens),
            static_forward_batch,
            **kwargs,
        )
        local_output = _slice_output_rows(local_output, cp_input.live_local_tokens)

        capture_aux_hidden_states = getattr(model, "capture_aux_hidden_states", False)
        aux_hidden_states = None
        if capture_aux_hidden_states:
            hidden_states, aux_hidden_states = local_output
        else:
            hidden_states = local_output

        if not model.pp_group.is_last_rank:
            return (
                (hidden_states, aux_hidden_states)
                if capture_aux_hidden_states
                else hidden_states
            )

        hidden_states = cp_gather_after_forward(
            hidden_states,
            static_forward_batch,
            torch.cuda.current_stream(),
        )
        return model.logits_processor(
            forward_batch.input_ids,
            hidden_states,
            model.lm_head,
            forward_batch,
            aux_hidden_states,
        )
