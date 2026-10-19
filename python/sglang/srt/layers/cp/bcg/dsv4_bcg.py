"""DeepSeek-V4.1 interleave CP inputs for breakable prefill graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.layers.cp.bcg.bcg import PrefillCPBCGInput
from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
from sglang.srt.layers.cp.padding import get_cp_padding_align_size
from sglang.srt.layers.cp.utils import (
    cp_interleave_input_ids,
    cp_shard_hidden_states,
    cp_split_before_forward,
    prepare_cp_forward,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )


@dataclass
class DSV4CPBCGInput(PrefillCPBCGInput):
    input_ids: Optional[torch.Tensor] = None
    global_input_ids: Optional[torch.Tensor] = None
    num_token_non_padded: Optional[torch.Tensor] = None
    supports_max_context_size = True

    def _max_local_tokens(self, extend_seq_lens):
        size = get_parallel().attn_cp_size
        return (sum(int(length) for length in extend_seq_lens) + size - 1) // size

    def model_positions(self, forward_batch):
        if forward_batch.cp_positions is None:
            raise RuntimeError("CP BCG positions must be prepared before capture")
        return forward_batch.cp_positions

    def model_input_ids(self, forward_batch):
        if forward_batch.cp_input_ids is None:
            raise RuntimeError("CP BCG token IDs must be prepared before capture")
        return forward_batch.cp_input_ids

    @classmethod
    def create(cls, runner: PrefillCudaGraphRunner) -> PrefillCPBCGInput:
        strategy = get_cp_strategy()
        global_id_capacity = 0
        if (
            isinstance(strategy, InterleaveCPStrategy)
            and get_moe_a2a_backend().is_none()
        ):
            align = get_cp_padding_align_size()
            local_capacity = (
                runner.max_num_tokens + strategy.cp_size - 1
            ) // strategy.cp_size
            local_capacity = (local_capacity + align - 1) // align * align
            global_id_capacity = strategy.cp_size * local_capacity
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
                input_ids=torch.zeros((runner.max_num_tokens,), dtype=torch.int64),
                global_input_ids=(
                    torch.zeros((global_id_capacity,), dtype=torch.int64)
                    if global_id_capacity
                    else None
                ),
                num_token_non_padded=torch.zeros((), dtype=torch.int32),
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
        forward_batch.cp_positions = None
        forward_batch.cp_input_ids = None
        out_cache_loc = forward_batch.out_cache_loc
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
        local_input_ids = cp_shard_hidden_states(global_input_ids, forward_batch)
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
        assert self.input_ids is not None
        input_ids = self.input_ids[:captured_local_tokens]
        input_embeds.zero_()
        positions.zero_()
        input_ids.zero_()
        input_embeds[:live_local_tokens].copy_(local_input_embeds)
        positions[:live_local_tokens].copy_(local_positions)
        input_ids[:live_local_tokens].copy_(local_input_ids)
        forward_batch.input_embeds = input_embeds
        forward_batch.cp_positions = positions
        # Keep the global input_ids field intact: the runner uses its length to
        # select the global capture bucket. MegaMoE routes local rows, whereas
        # TP-MoE all-gathers hidden rows in CP rank order before routing.
        forward_batch.cp_input_ids = input_ids
        if self.global_input_ids is not None:
            global_ids = cp_interleave_input_ids(global_input_ids, forward_batch)
            if global_ids.numel() > self.global_input_ids.numel():
                raise RuntimeError("CP global token IDs exceed the capture buffer")
            forward_batch.input_ids_global = self.global_input_ids[: global_ids.numel()]
            forward_batch.input_ids_global.copy_(global_ids)
        else:
            forward_batch.input_ids_global = input_ids
        if forward_batch.num_token_non_padded is not None:
            assert self.num_token_non_padded is not None
            metadata = forward_batch.attn_cp_metadata
            logical_tokens = (
                metadata.per_rank_logical_token or metadata.per_rank_actual_token
            )
            strategy = get_cp_strategy()
            assert strategy is not None
            self.num_token_non_padded.fill_(logical_tokens[strategy.cp_rank])
            forward_batch.num_token_non_padded = self.num_token_non_padded
        self.live_local_tokens = live_local_tokens

        if isinstance(get_cp_strategy(), InterleaveCPStrategy):
            metadata = forward_batch.attn_cp_metadata
            cp_size = len(metadata.per_rank_actual_token)
            base, remainder = divmod(static_num_tokens, cp_size)
            metadata.total_seq_lens = static_num_tokens
            metadata.per_rank_logical_token = [
                base + int(rank < remainder) for rank in range(cp_size)
            ]
            forward_batch.out_cache_loc = out_cache_loc
