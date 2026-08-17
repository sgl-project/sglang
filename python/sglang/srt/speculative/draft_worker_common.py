from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
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
    *, server_args: ServerArgs, algo_label: str
) -> str:
    draft_backend = server_args.speculative_draft_attention_backend
    if draft_backend is None:
        draft_backend, _ = server_args.get_attention_backends()
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
    ps: ParallelState,
    nccl_port: int,
    target_model_config: ModelConfig,
    algo_label: str,
    attention_backend_override: Optional[str] = None,
) -> DraftWorkerBundle:
    # An override names a draft-specific backend the caller has already
    # validated (e.g. a self-drafting architecture); it skips the generic
    # supported-backend fallback below.
    draft_backend = attention_backend_override or (
        _resolve_draft_attention_backend_fallback(
            server_args=server_args, algo_label=algo_label
        )
    )
    from sglang.srt.layers.moe.utils import draft_model_build_scope

    # The draft's model construction runs its own MoE gates; the scope routes
    # their fusion decision to the speculative leaf and gives the target its
    # ACTIVE value back. It deliberately does not swap runner_backend: these
    # workers run the draft outside speculative_moe_backend_context, so a
    # construction-only swap would build and execute under different backends.
    with draft_model_build_scope():
        draft_worker = TpModelWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            ps=ps,
            nccl_port=nccl_port,
            is_draft_worker=True,
            # The draft runs at absolute target positions.
            context_length=target_model_config.context_len,
            draft_attention_backend=draft_backend,
        )

    draft_model_runner = draft_worker.model_runner
    draft_worker.draft_runner = draft_model_runner

    # DFlash drafts have no vocab; borrow the target's.
    if draft_model_runner.model_config.vocab_size is None:
        draft_model_runner.model_config.vocab_size = target_model_config.vocab_size
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
