from dataclasses import dataclass, field
from typing import ClassVar, List, Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocation import alloc_for_spec_decode
from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_reserve_per_decode,
    page_aligned_decode_alloc_lens,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


@dataclass
class UnoDraftInput(SpecInput):
    """UNO state carried from one engine iteration to the next."""

    # Previously emitted token at logical position C. It has no KV yet.
    bonus_tokens: torch.Tensor

    # Target-correct KV frontier C.
    new_seq_lens: torch.Tensor

    # Number of queries in each UNO forward.
    forward_width: int

    # FutureMap compatibility. UNO relays only bonus_tokens, but the generic
    # relay payload reads these optional Eagle-shaped fields.
    topk_p: ClassVar[Optional[torch.Tensor]] = None
    topk_index: ClassVar[Optional[torch.Tensor]] = None
    hidden_states: ClassVar[Optional[torch.Tensor]] = None

    # Filled by the scheduler after an overlapped dispatch.
    future_indices: Optional[torch.Tensor] = None

    # Host-side upper bound for the allocator mapping prepared for this step.
    reserved_seq_lens_cpu: Optional[torch.Tensor] = None
    reserved_seq_lens_sum: Optional[int] = None

    # Sampling metadata prepared before either internal forward.
    max_top_k: int = 1
    uniform_top_k_value: Optional[int] = None

    def __post_init__(self):
        super().__init__(SpecInputType.UNO_STATE)

        if self.forward_width < 1:
            raise ValueError("UNO forward_width must be positive.")

        # The carried state represents one request row, not an F-row forward.
        self.num_tokens_per_req = 1
        self.num_tokens_for_logprob_per_req = 1

    @property
    def tail_width(self) -> int:
        return self.forward_width + 1

    @classmethod
    def create_idle_input(
        cls,
        *,
        device,
        forward_width: int,
    ) -> "UnoDraftInput":
        return cls(
            bonus_tokens=torch.empty(
                (0,),
                dtype=torch.int64,
                device=device,
            ),
            new_seq_lens=torch.empty(
                (0,),
                dtype=torch.int64,
                device=device,
            ),
            forward_width=forward_width,
        )

    def prepare_for_decode(self, batch: ScheduleBatch) -> None:
        batch.maybe_evict_swa()

        batch_size = batch.batch_size()
        if batch_size == 0:
            return

        if self.future_indices is None:
            if self.bonus_tokens.numel() != batch_size:
                raise RuntimeError("UNO seed count does not match decode batch size.")

            if self.new_seq_lens.numel() != batch_size:
                raise RuntimeError(
                    "UNO frontier count does not match decode batch size."
                )
        elif self.future_indices.numel() != batch_size:
            raise RuntimeError(
                "UNO future-index count does not match decode batch size."
            )

        committed_lengths: list[int] = []
        reserve_width = int(get_alloc_reserve_per_decode())

        max_top_k = 1
        uniform_top_k_value = None
        uniform_top_k = True

        for index, req in enumerate(batch.reqs):
            if req.kv is None:
                raise RuntimeError("UNO decode request has no KV allocation.")

            committed = int(req.kv.kv_committed_len)
            allocated = int(req.kv.kv_allocated_len)

            if allocated < committed:
                raise RuntimeError(
                    "UNO encountered an invalid KV watermark: "
                    f"committed={committed}, allocated={allocated}."
                )

            committed_lengths.append(committed)

            top_k = int(req.sampling_params.top_k)
            max_top_k = max(max_top_k, top_k)
            if index == 0:
                uniform_top_k_value = top_k
            elif uniform_top_k and top_k != uniform_top_k_value:
                uniform_top_k = False

        self.max_top_k = max_top_k
        self.uniform_top_k_value = uniform_top_k_value if uniform_top_k else None

        page_size = batch.token_to_kv_pool_allocator.page_size
        current_lengths, next_lengths, num_needed_tokens = (
            page_aligned_decode_alloc_lens(
                batch.reqs,
                reserve=reserve_width,
                page_size=page_size,
            )
        )
        row_width = int(batch.req_to_token_pool.req_to_token.shape[1])
        if max(next_lengths) > row_width:
            raise RuntimeError(
                "UNO allocation exceeds the req_to_token row: "
                f"needed={max(next_lengths)}, available={row_width}."
            )

        current_cpu = torch.tensor(
            current_lengths,
            dtype=torch.int32,
            device="cpu",
        )
        next_cpu = torch.tensor(
            next_lengths,
            dtype=torch.int32,
            device="cpu",
        )
        current_device = current_cpu.to(
            batch.device,
            non_blocking=True,
        )
        next_device = next_cpu.to(
            batch.device,
            non_blocking=True,
        )

        alloc_for_spec_decode(
            batch.tree_cache,
            batch.req_to_token_pool,
            reqs=batch.reqs,
            req_pool_indices=batch.req_pool_indices,
            cur_kv_lens=current_device,
            cur_kv_lens_cpu=current_cpu,
            nxt_kv_lens=next_device,
            nxt_kv_lens_cpu=next_cpu,
            num_needed_tokens=num_needed_tokens,
            batch=batch,
        )

        for req in batch.reqs:
            req.decode_batch_idx += 1

        batch.seq_lens_cpu = torch.tensor(
            committed_lengths,
            dtype=torch.int64,
            device="cpu",
        )
        batch.seq_lens_sum = sum(committed_lengths)
        self.reserved_seq_lens_cpu = next_cpu
        self.reserved_seq_lens_sum = sum(next_lengths)

    def filter_batch(
        self,
        new_indices: torch.Tensor,
        new_indices_cpu: Optional[List[int]] = None,
    ) -> None:
        if self.reserved_seq_lens_cpu is not None:
            host_indices = (
                new_indices_cpu if new_indices_cpu is not None else new_indices.cpu()
            )
            self.reserved_seq_lens_cpu = self.reserved_seq_lens_cpu[host_indices]
            self.reserved_seq_lens_sum = int(self.reserved_seq_lens_cpu.sum().item())

        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
            return

        self.bonus_tokens = self.bonus_tokens[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]

    def merge_batch(self, other: "UnoDraftInput") -> None:
        if not isinstance(other, UnoDraftInput):
            raise TypeError(f"Cannot merge UnoDraftInput with {type(other).__name__}.")

        if self.forward_width != other.forward_width:
            raise RuntimeError("Cannot merge UNO states with different forward widths.")

        self_has_reservation = self.reserved_seq_lens_cpu is not None
        other_has_reservation = other.reserved_seq_lens_cpu is not None
        if self_has_reservation != other_has_reservation:
            raise RuntimeError("Cannot merge prepared and unprepared UNO states.")

        if self_has_reservation:
            self.reserved_seq_lens_cpu = torch.cat(
                (
                    self.reserved_seq_lens_cpu,
                    other.reserved_seq_lens_cpu,
                )
            )
            self.reserved_seq_lens_sum = int(self.reserved_seq_lens_cpu.sum().item())

        if self.future_indices is not None:
            assert other.future_indices is not None
            self.future_indices = torch.cat((self.future_indices, other.future_indices))
            return

        self.bonus_tokens = torch.cat((self.bonus_tokens, other.bonus_tokens))
        self.new_seq_lens = torch.cat((self.new_seq_lens, other.new_seq_lens))


@dataclass
class UnoForwardInput(SpecInput):
    """Metadata for one fixed-width UNO forward."""

    # constructor
    spec_input_type: SpecInputType
    positions: torch.Tensor
    # For UNO, this is forward width F, not proposal count F - 1.
    draft_token_num: int

    # expected by interface
    custom_mask: Optional[torch.Tensor] = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL
    hidden_states: Optional[torch.Tensor] = None

    # derived
    num_tokens_per_req: int = field(init=False)
    num_tokens_for_logprob_per_req: int = field(init=False)

    def __post_init__(self):
        if self.spec_input_type not in {
            SpecInputType.UNO_DRAFT,
            SpecInputType.UNO_VERIFY,
        }:
            raise ValueError(f"Invalid UNO input type: {self.spec_input_type}")

        if self.draft_token_num < 1:
            raise ValueError("UNO forward width must be positive.")

        # Dataclass-generated __init__ does not call the non-dataclass base
        # initializer. This currently reassigns the same field, while also
        # preserving the SpecInput initialization contract.
        super().__init__(self.spec_input_type)

        self.num_tokens_per_req = self.draft_token_num
        self.num_tokens_for_logprob_per_req = self.draft_token_num
