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
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_flags

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

logger = logging.getLogger(__name__)


def should_run_flashinfer_autotune(
    model_runner: ModelRunner, *, for_speculative_draft: bool = False
) -> bool:
    """Check if flashinfer autotune should be run."""
    mr = model_runner
    if mr.device != "cuda":
        return False
    if mr.server_args.disable_flashinfer_autotune:
        return False

    # CuteDSL v1 (cutedsl runner + deepep a2a) bypasses MoeRunner and must not
    # be autotuned -- its _dummy_run would dispatch more tokens per rank than
    # SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK, tripping a DeepEP assert.
    # Read server_args directly to avoid depending on initialize_moe_config()
    # having already populated the MoE backend globals.
    if (
        get_flags().moe.runner_backend == "flashinfer_cutedsl"
        and get_flags().moe_a2a_backend == "deepep"
    ):
        return False

    backend_str = get_flags().moe.runner_backend

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
        get_fp8_gemm_runner_backend,
    )
    from sglang.srt.utils import is_sm100_supported

    model_uses_modelopt_fp8 = model_quantization in (
        "modelopt",
        "modelopt_fp8",
        "modelopt_mixed",
    )
    # Online MXFP8 (microscaling) linears dispatch to flashinfer's
    # ``mm_mxfp8``, which the flashinfer fp8 autotune dummy run does not
    # exercise correctly -- it triggers an illegal memory access inside the
    # mxfp8 cutlass cubin. The mxfp8 gemm is fixed-config and needs no
    # tuning, so skip autotune for these models.
    model_uses_mxfp8 = "mxfp8" in (model_quantization or "")
    fp8_gemm_needs_autotune = not model_uses_mxfp8 and (
        get_fp8_gemm_runner_backend().is_flashinfer_cutlass()
        or (model_uses_modelopt_fp8 and is_sm100_supported())
    )

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

    server_args = mr.server_args
    model_key_parts = [
        str(server_args.model_path),
        str(mr.dtype),
        str(server_args.quantization),
        str(get_flags().moe.runner_backend),
        str(mr.tp_size),
        str(mr.pp_size),
        str(mr.dp_size),
        str(mr.moe_ep_size),
        str(mr.model_config.hf_config.__class__.__name__),
    ]
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
    return cache_dir / f"rank_tp{mr.tp_rank}_pp{mr.pp_rank}_dp{mr.dp_rank or 0}.json"


@contextlib.contextmanager
def flashinfer_autotune_context(model_runner: ModelRunner, *, skip_logits: bool):
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
        maybe_skip_logits = contextlib.nullcontext()
        if skip_logits:
            from sglang.srt.layers.logits_processor import autotune_dummy_run_mode

            maybe_skip_logits = autotune_dummy_run_mode()
        with torch.inference_mode(), autotune(
            True, cache=str(autotune_cache)
        ), maybe_skip_logits:
            yield
    torch.cuda.current_stream().wait_stream(mr.forward_stream)
    logger.info("FlashInfer autotune completed.")


def run_flashinfer_autotune_forward(
    model_runner: ModelRunner, forward_fn: Callable[[], None], *, skip_logits: bool
) -> None:
    """Run flashinfer autotune forward."""
    with flashinfer_autotune_context(model_runner, skip_logits=skip_logits):
        forward_fn()


def maybe_flashinfer_autotune_speculative_draft(
    runner: BaseRunner,
    forward_fn: Callable[[], None],
    *,
    post_warmup_hook: Optional[Callable[[], None]] = None,
    skip_logits: bool = False,
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

    run_flashinfer_autotune_forward(mr, run_and_reset, skip_logits=skip_logits)
    tuned_phases.add(phase_key)
