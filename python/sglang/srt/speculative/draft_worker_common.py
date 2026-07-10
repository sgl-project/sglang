from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Optional

import msgspec
import torch

from sglang.srt.configs.model_config import is_deepseek_v4
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_context, get_server_args
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.utils.hf_transformers_utils import get_config

logger = logging.getLogger(__name__)

_SUPPORTED_DRAFT_BACKENDS = ("flashinfer", "fa3", "fa4", "triton", "ascend")

_DEEPSEEK_V4_DRAFT_BACKEND = "dsv4"


class DraftWorkerBundle(msgspec.Struct, frozen=True):
    draft_worker: TpModelWorker
    draft_model_runner: Any
    draft_model: Any
    resolved_attention_backend: str


def _resolve_draft_attention_backend(
    *, draft_server_args: ServerArgs, algo_label: str
) -> str:
    draft_hf_config = _load_draft_hf_config(draft_server_args=draft_server_args)
    return _select_draft_attention_backend(
        draft_hf_config=draft_hf_config,
        draft_server_args=draft_server_args,
        algo_label=algo_label,
    )


def _load_draft_hf_config(*, draft_server_args: ServerArgs) -> Optional[Any]:
    draft_model_path = draft_server_args.speculative_draft_model_path
    if not draft_model_path:
        return None
    model_override_args = json.loads(draft_server_args.json_model_override_args)
    return get_config(
        draft_model_path,
        trust_remote_code=draft_server_args.trust_remote_code,
        revision=draft_server_args.speculative_draft_model_revision,
        model_override_args=model_override_args,
        model_config_parser=draft_server_args.model_config_parser,
    )


def draft_is_deepseek_v4(*, server_args: ServerArgs) -> bool:
    draft_hf_config = _load_draft_hf_config(draft_server_args=server_args)
    return draft_hf_config is not None and is_deepseek_v4(draft_hf_config)


def _select_draft_attention_backend(
    *, draft_hf_config: Optional[Any], draft_server_args: ServerArgs, algo_label: str
) -> str:
    if draft_hf_config is not None and is_deepseek_v4(draft_hf_config):
        return _DEEPSEEK_V4_DRAFT_BACKEND
    return _resolve_draft_attention_backend_fallback(
        draft_server_args=draft_server_args, algo_label=algo_label
    )


def _resolve_draft_attention_backend_fallback(
    *, draft_server_args: ServerArgs, algo_label: str
) -> str:
    draft_backend = draft_server_args.speculative_draft_attention_backend
    if draft_backend is None:
        draft_backend, _ = draft_server_args.get_attention_backends()
    if draft_backend is None:
        return "triton" if torch.version.hip else "flashinfer"
    if draft_backend not in _SUPPORTED_DRAFT_BACKENDS:
        fallback = "triton" if torch.version.hip else "flashinfer"
        logger.warning(
            "%s draft worker only supports attention_backend in %s for now, "
            "but got %r. Falling back to '%s'.",
            algo_label,
            _SUPPORTED_DRAFT_BACKENDS,
            draft_backend,
            fallback,
        )
        return fallback
    return draft_backend


def build_draft_tp_worker(
    *,
    server_args: ServerArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    moe_ep_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    nccl_port: int,
    target_model_config: Any,
    algo_label: str,
) -> DraftWorkerBundle:
    draft_server_args = deepcopy(server_args)
    draft_backend = _resolve_draft_attention_backend(
        draft_server_args=draft_server_args, algo_label=algo_label
    )
    # Post-resolution ServerArgs rejects bare assignment; route the draft-copy
    # adjustments through the audited mutation point. The backend fields make
    # the draft worker explicit and self-contained (no further overrides);
    # context_length keeps the draft aligned with the target.
    draft_server_args.override(
        "draft_worker.build",
        skip_tokenizer_init=True,
        speculative_draft_attention_backend=None,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        attention_backend=draft_backend,
        context_length=target_model_config.context_len,
    )

    saved_server_args = get_server_args()
    draft_worker = TpModelWorker(
        server_args=draft_server_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        moe_ep_rank=moe_ep_rank,
        pp_rank=0,
        attn_cp_rank=attn_cp_rank,
        moe_dp_rank=moe_dp_rank,
        dp_rank=dp_rank,
        nccl_port=nccl_port,
        is_draft_worker=True,
    )
    get_context().set_server_args(saved_server_args)

    draft_model_runner = draft_worker.model_runner
    draft_worker.draft_runner = draft_model_runner
    return DraftWorkerBundle(
        draft_worker=draft_worker,
        draft_model_runner=draft_model_runner,
        draft_model=draft_model_runner.model,
        resolved_attention_backend=draft_backend,
    )


def make_draft_input_v2(
    *,
    bonus_tokens: torch.Tensor,
    new_seq_lens: torch.Tensor,
) -> DFlashDraftInputV2:
    bs = int(new_seq_lens.numel())
    device = bonus_tokens.device
    return DFlashDraftInputV2(
        topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
        topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
        bonus_tokens=bonus_tokens.to(dtype=torch.int64),
        new_seq_lens=new_seq_lens.to(dtype=torch.int64),
        hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
    )


def make_draft_block_spec_info(
    *,
    draft_token_num: int,
    device: torch.device,
) -> DFlashVerifyInput:
    return DFlashVerifyInput(
        draft_token=torch.empty((0,), dtype=torch.long, device=device),
        positions=torch.empty((0,), dtype=torch.int64, device=device),
        draft_token_num=int(draft_token_num),
        custom_mask=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


def make_draft_sampler_capture_hook(draft_sampler):

    def capture_hook(runner, out, forward_batch, num_tokens):
        del runner, num_tokens
        if not isinstance(out, LogitsProcessorOutput) or out.hidden_states is None:
            raise RuntimeError(
                "draft sampler set but the draft forward has no "
                "hidden_states to capture into the graph."
            )
        draft_sampler(out.hidden_states, forward_batch.input_ids)

    return capture_hook


def build_block_pos_offsets(*, length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(int(length), device=device, dtype=torch.int64)
