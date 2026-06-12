import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.environ import envs
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info_v2 import (
    EagleDraftInputV2Mixin,
    EagleVerifyInputV2Mixin,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    assign_req_to_token_pool_func,
    create_extend_after_decode_spec_info,
)
from sglang.srt.utils import next_power_of_2
from sglang.srt.utils.async_probe import maybe_detect_oob

logger = logging.getLogger(__name__)


def _draft_runner_of(worker):
    """Draft model_runner accessor across worker shapes.

    v2 draft workers (`EagleDraftWorker` and subclasses) expose the draft
    model_runner as `draft_runner`; fall back to `model_runner` for workers
    that run the draft model directly.
    """
    return (
        worker.draft_runner if hasattr(worker, "draft_runner") else worker.model_runner
    )


@dataclass
class EagleVerifyInput(SpecInput, EagleVerifyInputV2Mixin):
    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
    retrieve_index: torch.Tensor
    retrieve_next_token: torch.Tensor
    retrieve_next_sibling: torch.Tensor
    retrieve_cum_len: torch.Tensor
    spec_steps: int
    topk: int
    draft_token_num: int
    capture_hidden_mode: CaptureHiddenMode
    seq_lens_sum: int
    seq_lens_cpu: torch.Tensor
    grammar: BaseGrammarObject = None

    # Shape info for padding
    num_tokens_per_req: int = -1  # -1 auto-fills from draft_token_num.

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_VERIFY)
        if self.num_tokens_per_req < 0:
            self.num_tokens_per_req = self.draft_token_num

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    @classmethod
    def create_idle_input(cls, topk: int, spec_steps: int, num_verify_tokens: int):
        return cls(
            draft_token=torch.empty((0,), dtype=torch.long, device="cuda"),
            custom_mask=torch.full((0,), True, dtype=torch.bool, device="cuda"),
            positions=torch.empty((0,), dtype=torch.int64, device="cuda"),
            retrieve_index=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_next_token=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_next_sibling=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_cum_len=None,
            topk=topk,
            draft_token_num=num_verify_tokens,
            spec_steps=spec_steps,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
        )

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):

        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token
        maybe_detect_oob(
            batch.input_ids,
            0,
            batch.model_config.vocab_size,
            "eagle prepare_for_verify input_ids",
        )

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache,
                len(batch.input_ids),
            )
            end_offset = batch.seq_lens + self.draft_token_num
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.draft_token_num
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )

        bs = batch.batch_size()
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        if get_global_server_args().enable_mamba_extra_buffer():
            batch.mamba_track_indices = torch.tensor(
                [
                    req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx]
                    for req in batch.reqs
                ],
                dtype=torch.int64,
                device=batch.device,
            )
            batch.mamba_track_mask = None
            batch.mamba_track_seqlens = None

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        batch_size = len(req_pool_indices)
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )
        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=device
        )

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * batch_size,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask_numel = (
            paged_kernel_lens_sum * self.draft_token_num
            + (self.draft_token_num**2) * batch_size
        )
        if self.custom_mask.numel() < mask_numel:
            # FIXME(attn): temporary fix for custom mask padding with cuda graph
            self.custom_mask = torch.cat(
                [
                    self.custom_mask,
                    torch.full(
                        (mask_numel - self.custom_mask.numel(),),
                        True,
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=0,
            )

        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask


@dataclass
class EagleDraftInput(SpecInput, EagleDraftInputV2Mixin):
    # For idle stubs use `create_idle_input`, not the bare ctor: `filter_batch`
    # / `merge_batch` slice / cat `topk_p` / `topk_index` / `hidden_states` /
    # `bonus_tokens` unconditionally.

    # shape: (b, topk)
    topk_p: torch.Tensor = None
    topk_index: torch.Tensor = None
    # shape: (b, hidden_size) - one hidden per req, consumed by `draft` forward.
    # None when the spec algorithm's draft doesn't read hidden_states
    # (e.g., STANDALONE — vanilla LLM draft).
    hidden_states: Optional[torch.Tensor] = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Per-req bonus token (the "+1" target prediction at end of each accept
    # chain). Written by `EagleDraftExtendInput.prepare_extend_after_decode`;
    # the worker copies it here for next iter's draft.
    bonus_tokens: torch.Tensor = None

    # shape: (b + 1,)
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None

    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = -1

    # V2 overlap worker only: req_pool_indices used as buf slot keys.
    future_indices: Optional[torch.Tensor] = None
    # V2 reuses `EagleDraftInput` across phases (V1 has a separate
    # `EagleDraftExtendInput` for these). Set during V2's draft-extend.
    num_correct_drafts: Optional[torch.Tensor] = None
    num_accept_tokens: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.num_tokens_per_req, self.num_tokens_for_logprob_per_req

    @classmethod
    def hidden_size_for(cls, worker) -> Optional[int]:
        """Decode-phase `hidden_states` width: draft self-chain output
        (draft model writes its own last hidden back via `capture_for_decode`
        and the draft loop). Returns None when the draft architecture doesn't
        consume the field (e.g., STANDALONE)."""
        if worker.speculative_algorithm.is_standalone():
            return None
        return _draft_runner_of(worker).model_config.spec_hidden_size

    @classmethod
    def dtype_for(cls, worker) -> Optional[torch.dtype]:
        if worker.speculative_algorithm.is_standalone():
            return None
        return _draft_runner_of(worker).model_config.dtype

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: Optional[int],
        dtype: Optional[torch.dtype],
        topk: int,
        capture_hidden_mode: CaptureHiddenMode,
    ):
        return cls(
            bonus_tokens=torch.empty((0,), device=device, dtype=torch.int32),
            hidden_states=(
                torch.empty((0, hidden_size), device=device, dtype=dtype)
                if hidden_size is not None
                else None
            ),
            topk_p=torch.empty((0, topk), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, topk), device=device, dtype=torch.int64),
            capture_hidden_mode=capture_hidden_mode,
        )

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
            return

        strict_check = envs.SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK.get()
        if has_been_filtered:
            # in eagle_utils.py:verify, we have already filtered the batch by `unfinished_index`
            # therefore, we don't need to filter the batch again in scheduler
            error_msg = f"length of new_indices: {len(new_indices)} != length of topk_p: {len(self.topk_p)}, this should not happen"
            if len(new_indices) != len(self.topk_p):
                if strict_check:
                    raise ValueError(error_msg)
                else:
                    logger.warning(error_msg)

            self.topk_p = self.topk_p[: len(new_indices)]
            self.topk_index = self.topk_index[: len(new_indices)]
            if self.hidden_states is not None:
                self.hidden_states = self.hidden_states[: len(new_indices)]
            self.bonus_tokens = self.bonus_tokens[: len(new_indices)]
        else:
            # in some cases(e.g draft_extend), we have not filtered the batch by `unfinished_index`
            self.topk_p = self.topk_p[new_indices]
            self.topk_index = self.topk_index[new_indices]
            if self.hidden_states is not None:
                self.hidden_states = self.hidden_states[new_indices]
            self.bonus_tokens = self.bonus_tokens[new_indices]

    def merge_batch(self, spec_info: "EagleDraftInput"):
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = torch.cat(
                [self.future_indices, spec_info.future_indices]
            )
            return

        # Detect idle stub by `topk_index` length (idle inputs have
        # shape[0] == 0 across all fields). Don't use `hidden_states is None`:
        # for STANDALONE all non-idle inputs also have None hidden_states.
        if len(self.topk_index) == 0:
            self.hidden_states = spec_info.hidden_states
            self.bonus_tokens = spec_info.bonus_tokens
            self.topk_p = spec_info.topk_p
            self.topk_index = spec_info.topk_index
            return
        if len(spec_info.topk_index) == 0:
            return
        if self.hidden_states is not None and spec_info.hidden_states is not None:
            self.hidden_states = torch.cat(
                [self.hidden_states, spec_info.hidden_states], axis=0
            )
        self.bonus_tokens = torch.cat(
            [self.bonus_tokens, spec_info.bonus_tokens], axis=0
        )
        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p])
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index])


@dataclass
class EagleDraftExtendInput(SpecInput):
    """Inputs to the draft-extend forward (the per-accepted-token pass after verify).

    Produced by `EagleVerifyInput.verify`, installed on `batch.spec_info` for
    the draft-extend forward, then replaced with a fresh `EagleDraftInput` for
    the next iter's draft.
    """

    # shape: (total_accepted, hidden_size). Sliced from verify-time hidden_states
    # by accept_index; consumed by the draft-extend forward. None when the spec
    # algorithm's draft doesn't read hidden_states (e.g., STANDALONE).
    hidden_states: Optional[torch.Tensor] = None

    # Per-req accept counts. `num_accept_tokens = num_correct_drafts + 1`.
    # Both kept for cuda-graph buffer indexing and the
    # `create_extend_after_decode_spec_info` kernel.
    num_correct_drafts: torch.Tensor = None
    num_accept_tokens: torch.Tensor = None
    # CPU view, read by attention backends during the extend forward.
    num_accept_tokens_cpu: List[int] = None

    # Batch-state slices for the draft-extend forward. Set by verify (sliced to
    # reqs continuing into next iter). `prepare_extend_after_decode` copies
    # these onto `batch.{input_ids, seq_lens, seq_lens_cpu, req_pool_indices}`.
    #   - input_ids:        accept tokens flat over surviving reqs
    #   - seq_lens / _cpu:  per-req sequence length (post-accept)
    #   - req_pool_indices: per-req kv-pool slot
    input_ids: torch.Tensor = None
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    req_pool_indices: torch.Tensor = None

    # Set by `prepare_extend_after_decode`:
    #   - positions: kernel-written, shape `[total_accepted]`.
    #   - bonus_tokens: kernel-written, shape `[bs]`. The worker reads this
    #     post-extend to populate next iter's `EagleDraftInput.bonus_tokens`.
    positions: Optional[torch.Tensor] = None
    bonus_tokens: Optional[torch.Tensor] = None

    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.LAST
    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = 1

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT_EXTEND)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.num_tokens_per_req, self.num_tokens_for_logprob_per_req

    @classmethod
    def hidden_size_for(cls, worker) -> Optional[int]:
        """Extend-phase `hidden_states` width: target's `spec_hidden_size`,
        widened to `num_aux * target_hidden` for EAGLE-3 aux mode. Returns
        None when the draft architecture doesn't consume the field
        (e.g., STANDALONE)."""
        if worker.speculative_algorithm.is_standalone():
            return None
        target_cfg = worker.target_worker.model_runner.model_config
        if not (
            worker.speculative_algorithm.is_eagle3()
            and worker.eagle_use_aux_hidden_state
        ):
            return target_cfg.spec_hidden_size

        hf_config = target_cfg.hf_config

        # `num_aux` resolution: explicit attr > eagle_config layer_ids > default 3.
        num_aux = getattr(hf_config, "num_aux_hidden_states", None)
        if num_aux is None:
            eagle_config = getattr(hf_config, "eagle_config", None) or {}
            layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
            num_aux = len(layer_ids) if layer_ids else 3

        target_hidden = getattr(hf_config, "target_hidden_size", target_cfg.hidden_size)
        return target_hidden * num_aux

    @classmethod
    def dtype_for(cls, worker) -> Optional[torch.dtype]:
        if worker.speculative_algorithm.is_standalone():
            return None
        return worker.target_worker.model_runner.model_config.dtype

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: Optional[int],
        dtype: Optional[torch.dtype],
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.LAST,
    ) -> "EagleDraftExtendInput":
        return cls(
            hidden_states=(
                torch.empty((0, hidden_size), device=device, dtype=dtype)
                if hidden_size is not None
                else None
            ),
            num_correct_drafts=torch.empty((0,), device=device, dtype=torch.int32),
            num_accept_tokens=torch.empty((0,), device=device, dtype=torch.int32),
            num_accept_tokens_cpu=[],
            input_ids=torch.empty((0,), device=device, dtype=torch.long),
            seq_lens=torch.empty((0,), device=device, dtype=torch.int64),
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
            req_pool_indices=torch.empty((0,), device=device, dtype=torch.int64),
            capture_hidden_mode=capture_hidden_mode,
        )

    def prepare_extend_after_decode(
        self,
        batch: ScheduleBatch,
        speculative_num_steps: int,
    ):
        # Caller must have installed `self` as `batch.spec_info` before calling.
        assert batch.spec_info is self
        if batch.forward_mode.is_idle():
            return

        # The kernel below populates `self.positions` and `self.bonus_tokens`;
        # the worker reads `self.bonus_tokens` to construct next iter's
        # `EagleDraftInput`.
        batch.input_ids = self.input_ids
        batch.extend_lens = self.num_accept_tokens_cpu
        batch.extend_num_tokens = sum(batch.extend_lens)
        batch.seq_lens = self.seq_lens
        batch.seq_lens_cpu = self.seq_lens_cpu
        batch.req_pool_indices = self.req_pool_indices
        batch.return_logprob = False
        batch.return_hidden_states = False

        self.capture_hidden_mode = CaptureHiddenMode.LAST
        self.positions = torch.empty_like(batch.input_ids, dtype=torch.long)
        self.bonus_tokens = torch.empty_like(self.num_accept_tokens, dtype=torch.int32)

        create_extend_after_decode_spec_info[(len(batch.seq_lens),)](
            batch.input_ids,
            batch.seq_lens,
            self.num_accept_tokens,
            self.positions,
            self.bonus_tokens,
            next_power_of_2(max(speculative_num_steps + 1, len(batch.seq_lens))),
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: Optional[int],
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = self.num_correct_drafts.numel()
        qo_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        qo_indptr[1:] = torch.cumsum(self.num_accept_tokens, dim=0)
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        if paged_kernel_lens_sum is None:
            paged_kernel_lens_sum = cum_kv_seq_len[-1]

        kv_indices = torch.empty(
            paged_kernel_lens_sum, dtype=torch.int32, device=device
        )

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, None
