from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_context, get_schedule, get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# trtllm_mha: decode-only dense-MQA drafts (dspark). DFLASH excludes it
# earlier, at arg resolution (speculative_hook.py) -- its draft path needs
# per-layer DFlash attention -- so it never reaches this gate with it.
_SUPPORTED_DRAFT_BACKENDS = (
    "flashinfer",
    "fa3",
    "fa4",
    "triton",
    "ascend",
    "trtllm_mha",
)


class DraftWorkerBundle(msgspec.Struct, frozen=True):
    draft_worker: TpModelWorker
    draft_model_runner: ModelRunner
    draft_model: torch.nn.Module
    resolved_attention_backend: str


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


def _draft_load_format_fields() -> dict:
    draft_load_format = get_spec().speculative_draft_load_format
    if draft_load_format is None:
        return {}
    return dict(load_format=draft_load_format)


def draft_server_args_overrides(target_model_config, draft_backend) -> dict:
    """Pre-publish field adjustments for a draft ``ServerArgs`` copy.

    Downstream draft-worker logic keys on ``speculative_draft_attention_backend``
    (backend selection in ``_get_attention_backend``, the fa4-draft KV dtype
    override in ``configure_kv_cache_dtype``); ``context_length`` keeps the
    draft aligned with the target; ``disable_chunked_prefix_cache`` is the
    target's resolved gate (bag-only, absent from the pristine copy).
    """
    return dict(
        skip_tokenizer_init=True,
        speculative_draft_attention_backend=draft_backend,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        attention_backend=draft_backend,
        context_length=target_model_config.context_len,
        disable_chunked_prefix_cache=get_schedule().disable_chunked_prefix_cache,
        **_draft_load_format_fields(),
    )


def draft_server_args_copy(server_args: ServerArgs, target_model_config) -> ServerArgs:
    """A draft-only ``ServerArgs`` for the workers that build their own draft.

    Starts from the config the process resolved, not from the pristine seed:
    the copy is published while the draft builds, and load-time overrides made
    before this point (the chunked-prefix gate, the SM100 GDN prefill default)
    are part of what the draft's layers must see. On top of that,
    ``context_length`` follows the target (the draft reads target KV) and
    ``load_format`` follows ``--speculative-draft-load-format``. The target's
    own instance is untouched.
    """
    draft_load_format = get_spec().speculative_draft_load_format
    if draft_load_format is not None:
        logger.info(f"Using draft model load_format: '{draft_load_format}'")

    resolved = {}
    for _source, fields in get_context().overrides_log():
        resolved.update(fields)

    draft_server_args = deepcopy(server_args)
    draft_server_args.override(
        "draft_worker.copy",
        **{
            **resolved,
            "context_length": target_model_config.context_len,
            **_draft_load_format_fields(),
        },
    )
    return draft_server_args


def build_draft_tp_worker(
    *,
    server_args: ServerArgs,
    gpu_id: int,
    ps: ParallelState,
    nccl_port: int,
    target_model_config: ModelConfig,
    algo_label: str,
    attention_backend_override: Optional[str] = None,
) -> DraftWorkerBundle:
    draft_server_args = deepcopy(server_args)
    # An override names a draft-specific backend the caller has already
    # validated (e.g. a self-drafting architecture); it skips the generic
    # supported-backend fallback below.
    draft_backend = attention_backend_override or (
        _resolve_draft_attention_backend_fallback(
            draft_server_args=draft_server_args, algo_label=algo_label
        )
    )
    # Post-resolution ServerArgs rejects bare assignment; route the draft-copy
    # adjustments through the audited mutation point.
    draft_server_args.override(
        "draft_worker.build",
        **draft_server_args_overrides(target_model_config, draft_backend),
    )

    # The draft's layers must resolve config from the draft's own bags.
    with get_context().preserve_config():
        get_context().set_server_args(draft_server_args)
        draft_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            ps=ps,
            nccl_port=nccl_port,
            is_draft_worker=True,
        )

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
