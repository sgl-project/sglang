from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import msgspec
import torch

from sglang.srt.runtime_context import get_serving
from sglang.srt.utils.weight_versions import WeightVersionSpan, WeightVersionSpans

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs


_UNWRITTEN_VERSION_ID = -1


class KvWeightVersionRecord(msgspec.Struct):
    slot_indices: torch.Tensor
    version: str

    @classmethod
    def maybe_capture(
        cls, model_runner: ModelRunner, forward_batch: ForwardBatch
    ) -> Optional[KvWeightVersionRecord]:
        if (
            not model_runner.is_draft_worker
            and model_runner.server_args.enable_prefill_weight_versions
            and forward_batch.out_cache_loc is not None
        ):
            return cls.capture(
                slot_indices=forward_batch.out_cache_loc,
                version=get_serving().weight_version,
            )
        return None

    @classmethod
    def capture(
        cls, *, slot_indices: torch.Tensor, version: Optional[str]
    ) -> KvWeightVersionRecord:
        assert version is not None
        return cls(slot_indices=slot_indices.clone(), version=version)

    def map_device_tensors(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.slot_indices = fn(self.slot_indices)

    def finalize(self, *, tracker: Optional[KvWeightVersionTracker]) -> None:
        assert tracker is not None
        tracker.record(slot_indices=self.slot_indices, version=self.version)


class KvWeightVersionTracker:
    @classmethod
    def maybe_create(
        cls,
        *,
        server_args: ServerArgs,
        model_config: ModelConfig,
        allocator: BaseTokenToKVPoolAllocator,
        req_to_token_pool: ReqToTokenPool,
    ) -> Optional[KvWeightVersionTracker]:
        if not server_args.enable_prefill_weight_versions:
            return None

        assert (
            not model_config.is_encoder_decoder
        ), "--enable-prefill-weight-versions does not support encoder-decoder models"
        assert (
            model_config.is_generation
        ), "--enable-prefill-weight-versions does not support embedding or reward models"
        assert (
            server_args.pp_size == 1
        ), "--enable-prefill-weight-versions does not support pipeline parallelism"

        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

        if isinstance(allocator, SWATokenToKVPoolAllocator):
            allocator = allocator.full_attn_allocator

        return cls(
            num_slots=allocator.size_full + allocator.page_size,
            device=allocator.device,
            req_to_token_pool=req_to_token_pool,
        )

    def __init__(
        self, *, num_slots: int, device: str, req_to_token_pool: ReqToTokenPool
    ):
        self._slot_version_ids = torch.full(
            (num_slots,), _UNWRITTEN_VERSION_ID, dtype=torch.int32, device=device
        )
        self._req_to_token_pool = req_to_token_pool
        self._versions = _StringInterner()

    def record(self, *, slot_indices: torch.Tensor, version: Optional[str]) -> None:
        assert version is not None
        self._slot_version_ids[slot_indices] = self._versions.intern(version)

    def fill_req_prefill_weight_versions(self, req: Req) -> None:
        num_prompt_tokens = len(req.origin_input_ids)
        assert req.kv_committed_len >= num_prompt_tokens, (
            f"prefill finished with {req.kv_committed_len} committed KV tokens "
            f"for a {num_prompt_tokens}-token prompt"
        )
        req.prefill_weight_versions = self._lookup_spans(
            self._req_to_token_pool.req_to_token[req.req_pool_idx, :num_prompt_tokens]
        )

    def _lookup_spans(self, slot_indices: torch.Tensor) -> WeightVersionSpans:
        version_ids = self._slot_version_ids[slot_indices]
        if len(version_ids) == 0:
            return []
        if (is_unwritten := version_ids == _UNWRITTEN_VERSION_ID).any():
            raise ValueError(
                "KV slots without a recorded weight version were looked up: "
                f"{slot_indices[is_unwritten].tolist()}"
            )

        version_changes_at: List[int] = (
            (version_ids[1:] != version_ids[:-1]).nonzero().flatten().tolist()
        )
        run_starts = [0] + [position + 1 for position in version_changes_at]
        run_ends = run_starts[1:] + [len(version_ids)]
        run_version_ids: List[int] = version_ids[run_starts].tolist()

        return [
            WeightVersionSpan(
                version=self._versions.lookup(version_id), start=start, end=end
            )
            for version_id, start, end in zip(
                run_version_ids, run_starts, run_ends, strict=True
            )
        ]


class _StringInterner:
    def __init__(self):
        self._str_by_id: List[str] = []
        self._id_by_str: Dict[str, int] = {}

    def intern(self, value: str) -> int:
        if (value_id := self._id_by_str.get(value)) is not None:
            return value_id

        value_id = len(self._str_by_id)
        self._str_by_id.append(value)
        self._id_by_str[value] = value_id
        return value_id

    def lookup(self, value_id: int) -> str:
        return self._str_by_id[value_id]
