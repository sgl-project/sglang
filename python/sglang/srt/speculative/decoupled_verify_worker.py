from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.srt.speculative.eagle_worker import EAGLEWorker

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput

logger = logging.getLogger(__name__)


def _get_req_tail_token_id(req) -> int:
    if req.output_ids:
        return int(req.output_ids[-1])
    if req.origin_input_ids:
        return int(req.origin_input_ids[-1])
    raise RuntimeError(
        f"Request {req.rid} has no committed token to anchor external draft verification."
    )


def _slice_tensor_head_or_empty(
    value: torch.Tensor | None,
    live_count: int,
    *,
    empty_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.empty(empty_shape, dtype=dtype, device=device)
    return value[:live_count]


def _normalize_token_id(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (list, tuple, set)):
        for item in value:
            normalized = _normalize_token_id(item)
            if normalized is not None:
                return normalized
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_linear_topk1_tree_metadata(
    batch_size: int,
    spec_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_index = torch.arange(
        spec_steps,
        dtype=torch.long,
        device=device,
    ).expand(batch_size, -1).contiguous()

    if spec_steps <= 1:
        parent_list = torch.empty((batch_size, 0), dtype=torch.long, device=device)
    else:
        parent_list = torch.arange(
            -1,
            spec_steps - 1,
            dtype=torch.long,
            device=device,
        ).expand(batch_size, -1).contiguous()

    return selected_index, parent_list

def normalize_external_draft_batch_spec_info(batch: ScheduleBatch) -> None:
    spec_info = getattr(batch, "spec_info", None)
    if not isinstance(spec_info, EagleDraftInput):
        return

    seq_lens = getattr(batch, "seq_lens", None)
    seq_lens_cpu = getattr(batch, "seq_lens_cpu", None)
    req_pool_indices = getattr(batch, "req_pool_indices", None)
    seq_lens_dtype = seq_lens.dtype if isinstance(seq_lens, torch.Tensor) else torch.int32
    seq_lens_cpu_dtype = (
        seq_lens_cpu.dtype if isinstance(seq_lens_cpu, torch.Tensor) else torch.int32
    )
    req_pool_indices_dtype = (
        req_pool_indices.dtype if isinstance(req_pool_indices, torch.Tensor) else torch.int32
    )

    live_count = sum(1 for req in batch.reqs if not req.is_retracted and not req.finished())
    if live_count == 0:
        batch.spec_info = EagleDraftInput.create_idle_input(
            device=batch.device,
            hidden_size=batch.model_config.hidden_size,
            dtype=batch.model_config.dtype,
            topk=1,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        return

    hidden_states = _slice_tensor_head_or_empty(
        spec_info.hidden_states,
        live_count,
        empty_shape=(live_count, batch.model_config.hidden_size),
        dtype=batch.model_config.dtype,
        device=batch.device,
    )
    verified_dtype = (
        spec_info.verified_id.dtype
        if isinstance(spec_info.verified_id, torch.Tensor)
        else torch.int32
    )
    capture_hidden_mode = getattr(
        spec_info, "capture_hidden_mode", CaptureHiddenMode.LAST
    )

    batch.spec_info = EagleDraftInput(
        hidden_states=hidden_states,
        verified_id=_slice_tensor_head_or_empty(
            spec_info.verified_id,
            live_count,
            empty_shape=(live_count,),
            dtype=verified_dtype,
            device=batch.device,
        ),
        topk_p=torch.empty((live_count, 1), dtype=torch.float32, device=batch.device),
        topk_index=torch.empty((live_count, 1), dtype=torch.int64, device=batch.device),
        capture_hidden_mode=capture_hidden_mode,
        accept_length=torch.zeros((live_count,), dtype=torch.int32, device=batch.device),
        accept_length_cpu=[0] * live_count,
        seq_lens_for_draft_extend=_slice_tensor_head_or_empty(
            getattr(spec_info, "seq_lens_for_draft_extend", seq_lens),
            live_count,
            empty_shape=(live_count,),
            dtype=seq_lens_dtype,
            device=batch.device,
        ),
        seq_lens_for_draft_extend_cpu=getattr(
            spec_info, "seq_lens_for_draft_extend_cpu", seq_lens_cpu
        )[:live_count]
        if getattr(spec_info, "seq_lens_for_draft_extend_cpu", seq_lens_cpu) is not None
        else torch.empty((0,), dtype=seq_lens_cpu_dtype),
        req_pool_indices_for_draft_extend=_slice_tensor_head_or_empty(
            getattr(spec_info, "req_pool_indices_for_draft_extend", req_pool_indices),
            live_count,
            empty_shape=(live_count,),
            dtype=req_pool_indices_dtype,
            device=batch.device,
        ),
    )


class VerifyWorker:
    verify = EAGLEWorker.verify
    _mamba_verify_update = EAGLEWorker._mamba_verify_update

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int | None,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        del gpu_id, moe_ep_rank, moe_dp_rank, nccl_port
        self.server_args = server_args
        self.target_worker = target_worker
        self.tp_rank = int(tp_rank)
        self.attn_cp_rank = int(attn_cp_rank)
        self.dp_rank = 0 if dp_rank is None else int(dp_rank)
        self.pp_rank = int(getattr(target_worker, "pp_rank", 0))
        self.model_runner = target_worker.model_runner
        self.model_config = target_worker.model_config
        self.page_size = server_args.page_size
        self.topk = 1
        self.speculative_num_steps = int(server_args.speculative_num_steps)
        self.speculative_num_draft_tokens = int(server_args.speculative_num_draft_tokens)
        self.enable_nan_detection = bool(server_args.enable_nan_detection)
        self.device = self.model_runner.device
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        self.total_accept_length = 0
        self.total_num_verified_reqs = 0

    def clear_cache_pool(self):
        return

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        return self.target_worker.update_weights_from_tensor(recv_req)

    def _get_verify_buffers(self, draft_token_num: int):
        if draft_token_num != self.speculative_num_draft_tokens:
            return None, None

        attn_backend = getattr(self.target_worker.model_runner, "attn_backend", None)
        if attn_backend is None:
            return None, None

        get_buffers = getattr(
            attn_backend, "get_verify_buffers_to_fill_after_draft", None
        )
        if get_buffers is None:
            return None, None

        try:
            return get_buffers()
        except Exception as exc:
            logger.debug("Falling back to eager verify buffers: %s", exc)
            return None, None

    def _get_pad_token_id(self) -> int:
        """Return an EOS token id used to pad short external draft tails."""
        hf_generation_config = getattr(self.model_config, "hf_generation_config", None)
        eos_token_id = _normalize_token_id(
            getattr(hf_generation_config, "eos_token_id", None)
        )
        if eos_token_id is not None:
            return eos_token_id

        hf_config = getattr(self.model_config, "hf_config", None)
        eos_token_id = _normalize_token_id(getattr(hf_config, "eos_token_id", None))
        if eos_token_id is not None:
            return eos_token_id

        get_text_config = getattr(hf_config, "get_text_config", None)
        text_config = (
            get_text_config()
            if callable(get_text_config)
            else getattr(hf_config, "text_config", None)
        )
        eos_token_id = _normalize_token_id(getattr(text_config, "eos_token_id", None))
        if eos_token_id is not None:
            return eos_token_id

        eos_token_ids = getattr(self.model_config, "hf_eos_token_id", None)
        if eos_token_ids:
            return min(int(token_id) for token_id in eos_token_ids)

        raise RuntimeError("External draft verification requires an EOS token id.")

    def _build_req_verify_tokens(self, req, pad_token_id: int) -> list[int]:
        tail_token = _get_req_tail_token_id(req)
        draft_buffer = list(getattr(req, "draft_buffer", []) or [])
        spec_depth = self.speculative_num_draft_tokens - 1
        draft_tokens = list(draft_buffer[:spec_depth])
        if len(draft_tokens) < spec_depth:
            draft_tokens.extend([int(pad_token_id)] * (spec_depth - len(draft_tokens)))
        return [tail_token, *draft_tokens]

    def _assert_verify_output_within_snapshot_tail(
        self, batch: ScheduleBatch, verify_output
    ):
        # req.draft_buffer is a per-forward snapshot bound before verify. Any
        # concurrent drafter appends belong to later verify rounds.
        real_tail_lens = [
            min(
                len(list(getattr(req, "draft_buffer", []) or [])),
                self.speculative_num_draft_tokens - 1,
            )
            for req in batch.reqs
        ]
        raw_accept_lens = [int(x) for x in verify_output.accept_length_per_req_cpu]
        for req, raw_accept_len, real_tail_len in zip(
            batch.reqs, raw_accept_lens, real_tail_lens
        ):
            assert raw_accept_len <= real_tail_len, (
                "Decoupled verify has accepted padded draft tokens: "
                f"request_id={req.rid} "
                f"raw_accept_len={raw_accept_len} "
                f"snapshot_tail_len={real_tail_len}"
            )

        return verify_output.verified_id, raw_accept_lens

    def _build_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        draft_token_num = self.speculative_num_draft_tokens
        if draft_token_num < 2:
            raise RuntimeError(
                "External draft verification requires at least one draft token per request."
            )

        pad_token_id = self._get_pad_token_id()
        full_draft_tokens_by_req = [
            self._build_req_verify_tokens(req, pad_token_id) for req in batch.reqs
        ]
        spec_steps = draft_token_num - 1
        verified_id = torch.tensor(
            [tokens[0] for tokens in full_draft_tokens_by_req],
            dtype=torch.long,
            device=batch.device,
        )
        draft_tokens = torch.tensor(
            [tokens[1:] for tokens in full_draft_tokens_by_req],
            dtype=torch.long,
            device=batch.device,
        )

        batch_size = batch.batch_size()
        seq_lens_sum = int(torch.sum(batch.seq_lens).item())
        selected_index, parent_list = _build_linear_topk1_tree_metadata(
            batch_size,
            spec_steps,
            batch.device,
        )

        tree_mask_buf, position_buf = self._get_verify_buffers(draft_token_num)
        (
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            flat_draft_tokens,
        ) = build_tree_kernel_efficient(
            verified_id=verified_id,
            parent_list=parent_list,
            top_scores_index=selected_index,
            draft_tokens=draft_tokens,
            seq_lens=batch.seq_lens,
            seq_lens_sum=seq_lens_sum,
            topk=1,
            spec_steps=spec_steps,
            num_verify_tokens=draft_token_num,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
            tree_mask_buf=tree_mask_buf,
            position_buf=position_buf,
        )

        return EagleVerifyInput(
            draft_token=flat_draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=spec_steps,
            topk=1,
            draft_token_num=draft_token_num,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            result = self.target_worker.forward_batch_generation(model_worker_batch)
            return result

        spec_info = self._build_verify_input(batch)
        can_use_full_graph_path = (
            spec_info.draft_token_num == self.speculative_num_draft_tokens
        )
        logits_output, verify_output, _, can_run_cuda_graph = self.verify(batch, spec_info)
        verified_id, accept_length_per_req_cpu = (
            self._assert_verify_output_within_snapshot_tail(batch, verify_output)
        )

        normalize_external_draft_batch_spec_info(batch)
        result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verified_id,
            num_accepted_tokens=sum(accept_length_per_req_cpu),
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph and can_use_full_graph_path,
        )
        num_verified_reqs = len(accept_length_per_req_cpu)
        self.total_accept_length += int(result.num_accepted_tokens)
        self.total_num_verified_reqs += num_verified_reqs
        return result
