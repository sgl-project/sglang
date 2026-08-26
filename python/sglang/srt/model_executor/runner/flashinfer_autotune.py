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
from __future__ import annotations

import contextlib
import datetime
import functools
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_model,
    get_spec,
)
from sglang.srt.utils import empty_context, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

logger = logging.getLogger(__name__)

FLASHINFER_AUTOTUNE_WORKAROUND_SKIPS = frozenset()


def get_flashinfer_autotune_skip_ops(model_runner: ModelRunner) -> set[str]:
    skip_ops = set(get_exec().kernel.flashinfer_autotune_skip_ops or ())
    skip_ops.update(FLASHINFER_AUTOTUNE_WORKAROUND_SKIPS)
    return skip_ops


def should_run_flashinfer_autotune(
    model_runner: ModelRunner, *, for_speculative_draft: bool = False
) -> bool:
    """Check if flashinfer autotune should be run."""
    mr = model_runner
    if mr.device != "cuda":
        return False
    if get_exec().kernel.disable_flashinfer_autotune:
        return False
    if get_exec().deterministic.enable_deterministic_inference:
        # Tuned configs are per problem shape, so the reduction order would follow
        # the batch shape.
        return False

    if for_speculative_draft:
        backend_str = (
            get_spec().speculative_moe_runner_backend
            or get_exec().moe.moe_runner_backend
        )
        a2a_backend_str = (
            get_spec().speculative_moe_a2a_backend or get_exec().moe.moe_a2a_backend
        )
    else:
        backend_str = get_exec().moe.moe_runner_backend
        a2a_backend_str = get_exec().moe.moe_a2a_backend

    # Autotune can run before the MoE backend globals are initialized, so read
    # the configured backends -- the draft leaves (`get_spec()`) or the target
    # leaves (`get_exec().moe`) above. CuteDSL v1 bypasses MoeRunner, and its
    # dummy dispatch can exceed DeepEP low-latency's token limit.
    if backend_str == "flashinfer_cutedsl" and a2a_backend_str == "deepep":
        return False

    # TODO smor- support other cases for flashinfer autotune, such as, mamba backend

    moe_needs_autotune = backend_str in [
        "flashinfer_trtllm",
        "flashinfer_trtllm_routed",
        "flashinfer_mxfp4",
        "flashinfer_cutedsl",
        "flashinfer_cutlass",
    ]

    from sglang.srt.layers.quantization.fp4_utils import (
        get_fp4_gemm_runner_backend,
    )

    model_quantization = mr.model_config.quantization
    model_uses_fp4 = model_quantization in (
        "modelopt_fp4",
        "modelopt_mixed",
    )
    fp4_gemm_needs_autotune = model_uses_fp4 and (
        get_fp4_gemm_runner_backend().is_flashinfer_cutlass()
        or get_fp4_gemm_runner_backend().is_flashinfer_cutedsl()
    )

    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_per_tensor_fp8_supported,
        resolve_mxfp8_dense_gemm_backend,
    )

    if model_quantization == "mxfp8":
        fp8_gemm_needs_autotune = resolve_mxfp8_dense_gemm_backend().is_flashinfer()
    elif model_quantization in ("modelopt", "modelopt_fp8", "modelopt_mixed"):
        fp8_gemm_needs_autotune = flashinfer_per_tensor_fp8_supported()
    else:
        fp8_gemm_needs_autotune = False

    if not (moe_needs_autotune or fp4_gemm_needs_autotune or fp8_gemm_needs_autotune):
        return False

    if torch.cuda.get_device_capability()[0] < 9:
        return False

    if mr.spec_algorithm.is_speculative():
        return mr.is_draft_worker if for_speculative_draft else not mr.is_draft_worker

    return True


def flashinfer_autotune_cache_path(model_runner: ModelRunner) -> Path:
    import flashinfer

    mr = model_runner
    major, minor = torch.cuda.get_device_capability(mr.device)
    arch = f"sm{major}{minor}"
    flashinfer_version = getattr(flashinfer, "__version__", "unknown")

    model_key_parts = [
        str(get_model().model_path),
        str(mr.dtype),
        str(get_model().quantization),
        str(get_exec().moe.moe_runner_backend),
        str(mr.ps.tp_size),
        str(mr.ps.pp_size),
        str(mr.ps.attn_dp_size),
        str(mr.ps.moe_ep_size),
        str(mr.model_config.hf_config.__class__.__name__),
    ]
    # A different skip policy must not reuse previously tuned tactics.
    skip_ops = get_flashinfer_autotune_skip_ops(mr)
    model_key_parts.append("skip_ops=" + ",".join(sorted(skip_ops)))
    if mr.is_draft_worker:
        model_key_parts.append(f"draft_quant={mr.model_config.quantization}")
    model_key = "|".join(model_key_parts)
    cache_key = hashlib.sha256(model_key.encode()).hexdigest()[:16]
    cache_dir = (
        Path(envs.SGLANG_CACHE_DIR.get())
        / "flashinfer"
        / "autotune"
        / flashinfer_version
        / arch
        / cache_key
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir
        / f"rank_tp{mr.ps.tp_rank}_pp{mr.ps.pp_rank}_dp{mr.ps.dp_rank or 0}.json"
    )


@contextlib.contextmanager
def flashinfer_autotune_context(model_runner: ModelRunner, *, run_lm_head: bool):
    from flashinfer.autotuner import autotune

    mr = model_runner
    cache_path = flashinfer_autotune_cache_path(mr)
    if envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.get():
        autotune_cache = cache_path
        logger.info("Running FlashInfer autotune with cache: %s", autotune_cache)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_dir = cache_path.parent / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        autotune_cache = runs_dir / f"{cache_path.stem}.{timestamp}{cache_path.suffix}"
        logger.info(
            "Running FlashInfer autotune (cache reuse DISABLED via "
            "SGLANG_FLASHINFER_AUTOTUNE_CACHE=0); writing fresh result to: %s",
            autotune_cache,
        )

    # Run warmup on the non-default stream to avoid NCCL 2.29+ cudaMemcpyBatchAsync
    # calls on default stream (unsupported by CUDA) when --enable-symm-mem is used.
    mr.forward_stream.wait_stream(torch.cuda.current_stream())
    with torch.get_device_module(mr.device).stream(mr.forward_stream):
        from sglang.srt.layers.logits_processor import autotune_dummy_run_mode

        skip_ops = get_flashinfer_autotune_skip_ops(mr)
        with autotune(
            True,
            cache=str(autotune_cache),
            skip_ops=skip_ops,
        ), autotune_dummy_run_mode(run_lm_head=run_lm_head):
            yield
    torch.cuda.current_stream().wait_stream(mr.forward_stream)
    logger.info("FlashInfer autotune completed.")


def run_flashinfer_autotune_forward(
    model_runner: ModelRunner, forward_fn: Callable[[], None], *, run_lm_head: bool
) -> None:
    """Run flashinfer autotune forward."""
    with flashinfer_autotune_context(model_runner, run_lm_head=run_lm_head):
        forward_fn()


def maybe_flashinfer_autotune_speculative_draft(
    runner: BaseRunner,
    forward_fn: Callable[[], None],
    *,
    post_warmup_hook: Optional[Callable[[], None]] = None,
    run_lm_head: bool = True,
) -> None:
    """Run speculative draft flashinfer autotune."""
    mr = runner.model_runner
    phase_key = f"{runner.__class__.__module__}.{runner.__class__.__qualname__}"
    tuned_phases = getattr(mr, "_flashinfer_spec_draft_autotuned_phases", None)
    if tuned_phases is None:
        tuned_phases = set()
        mr._flashinfer_spec_draft_autotuned_phases = tuned_phases
    if phase_key in tuned_phases:
        return
    if (
        not mr.spec_algorithm.is_speculative()
        or not mr.is_draft_worker
        or not should_run_flashinfer_autotune(mr, for_speculative_draft=True)
    ):
        return

    def run_and_reset():
        forward_fn()
        if post_warmup_hook is not None:
            post_warmup_hook()

    run_flashinfer_autotune_forward(mr, run_and_reset, run_lm_head=run_lm_head)
    tuned_phases.add(phase_key)


def maybe_flashinfer_autotune_extend(
    runner: BaseRunner, *, decode_num_tokens: int
) -> None:
    """Also autotune one EXTEND-shaped dummy forward.

    The decode-shaped autotune only covers token counts up to the decode
    batch size, so larger prefill/extend batches fall outside the tuned
    buckets and run flashinfer's default heuristic — which can be far
    slower than the tuned tactic (e.g. trtllm-gen fp4 MoE is ~30% slower
    untuned at >=8k tokens on sm100). One extra forward at the largest
    per-rank extend token count tunes all buckets up to it.
    """
    if not envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.get():
        return
    mr = runner.model_runner
    # Prefer the per-rank scheduler buffer while preserving the legacy ceiling
    # when chunked prefill is disabled.
    num_tokens = (
        mr.server_args.max_prefill_buffer_tokens() or mr.server_args.max_prefill_tokens
    )
    if num_tokens <= (decode_num_tokens or 0):
        return  # decode-shaped autotune already covered these buckets
    is_pd_prefill_target = (
        get_disagg().disaggregation_mode == "prefill" and not mr.is_draft_worker
    )
    if not mr.is_generation or (
        mr.spec_algorithm.is_speculative() and not is_pd_prefill_target
    ):
        # Ordinary speculative runners force TARGET_VERIFY; PD prefill targets
        # have no draft-side state and preserve the requested EXTEND mode.
        return
    # Multimodal generation wrappers can still run this text-only EXTEND dummy;
    # an incompatible model should fail the explicit opt-in visibly.

    if mr.attn_backend.extend_dummy_seqs_capped_by_req_pool:
        pool_size = mr.req_to_token_pool.size
        num_tokens_per_req = (num_tokens + pool_size - 1) // pool_size
    else:
        # Packed dummies tune measurably worse tactics for the same token
        # bucket, so pack only where the backend would otherwise crash. None
        # (not 1) keeps the backend's own seq_len_fill_value in _dummy_run.
        num_tokens_per_req = None
    per_req = num_tokens_per_req or 1
    batch_size = (num_tokens + per_req - 1) // per_req
    num_tokens = batch_size * per_req

    buffers = runner._alloc_dummy_decode_buffers(
        batch_size,
        num_tokens_per_req=per_req,
        allocate_logits_buffer=False,
    )
    canary_run_ctx = (
        c.with_active_single_forward_manager(0)
        if (c := mr.canary_manager) is not None
        else empty_context()
    )

    forward_fn = functools.partial(
        runner._dummy_run,
        batch_size=batch_size,
        buffers=buffers,
        run_ctx=canary_run_ctx,
        forward_mode_override=ForwardMode.EXTEND,
        extend_num_tokens_per_req=num_tokens_per_req,
    )

    log_info_on_rank0(
        logger,
        f"FlashInfer autotune: extra EXTEND pass at {num_tokens} tokens "
        f"({batch_size} seqs x {per_req} tokens).",
    )
    try:
        run_flashinfer_autotune_forward(mr, forward_fn, run_lm_head=False)
    except torch.OutOfMemoryError:
        # The pass is an optimization; without headroom for the extend-shaped
        # forward, fall back to untuned extend buckets instead of failing.
        log_info_on_rank0(
            logger,
            "FlashInfer extend autotune skipped: not enough free memory "
            f"for a {num_tokens}-token dummy forward.",
        )
    finally:
        # release dummy buffers before capture measures free memory
        del forward_fn, buffers
        torch.cuda.empty_cache()
