from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _should_run_flashinfer_autotune(
    *,
    server_args: ServerArgs,
    spec_algorithm: SpeculativeAlgorithm,
    is_draft_worker: bool,
) -> bool:
    """Check if flashinfer autotune should be run."""
    if server_args.disable_flashinfer_autotune:
        return False

    # CuteDSL v1 (cutedsl runner + deepep a2a) bypasses MoeRunner and must not
    # be autotuned -- its _dummy_run would dispatch more tokens per rank than
    # SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK, tripping a DeepEP assert.
    # Read server_args directly to avoid depending on initialize_moe_config()
    # having already populated the MoE backend globals.
    if (
        server_args.moe_runner_backend == "flashinfer_cutedsl"
        and server_args.moe_a2a_backend == "deepep"
    ):
        return False

    backend_str = server_args.moe_runner_backend

    # TODO smor- support other cases for flashinfer autotune, such as, mamba backend

    if backend_str not in [
        "flashinfer_trtllm",
        # TODO: Enable for flashinfer_trtllm_routed once https://github.com/flashinfer-ai/flashinfer/issues/2749 is fixed.
        # "flashinfer_trtllm_routed",
        "flashinfer_mxfp4",
        "flashinfer_cutedsl",
        # TODO: flashinfer_cutlass will cause some flashinfer compilation errors. To be fixed.
        # "flashinfer_cutlass",
    ]:
        return False

    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        return False

    if spec_algorithm.is_speculative():
        return not is_draft_worker

    return True


def _flashinfer_autotune_cache_path(
    *,
    server_args: ServerArgs,
    model_config: ModelConfig,
    dtype: torch.dtype,
    device: str,
    tp_rank: int,
    tp_size: int,
    pp_rank: int,
    pp_size: int,
    dp_rank: int,
    dp_size: int,
    moe_ep_size: int,
) -> Path:
    import flashinfer

    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm{major}{minor}"
    flashinfer_version = getattr(flashinfer, "__version__", "unknown")

    server_args = server_args
    model_key = "|".join(
        [
            str(server_args.model_path),
            str(dtype),
            str(server_args.quantization),
            str(server_args.moe_runner_backend),
            str(tp_size),
            str(pp_size),
            str(dp_size),
            str(moe_ep_size),
            str(model_config.hf_config.__class__.__name__),
        ]
    )
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
    return cache_dir / f"rank_tp{tp_rank}_pp{pp_rank}_dp{dp_rank or 0}.json"


def kernel_warmup(
    *,
    device: str,
    server_args: ServerArgs,
    spec_algorithm: SpeculativeAlgorithm,
    is_draft_worker: bool,
    model_config: ModelConfig,
    dtype: torch.dtype,
    forward_stream: torch.cuda.Stream,
    req_to_token_pool_size: int,
    tp_rank: int,
    tp_size: int,
    pp_rank: int,
    pp_size: int,
    dp_rank: int,
    dp_size: int,
    moe_ep_size: int,
    dummy_run_callable,
):
    """
    Warmup and tune kernels before cuda graph capture.
    Currently only doing FlashInfer autotune.
    """
    if device != "cuda":
        return

    if _should_run_flashinfer_autotune(
        server_args=server_args,
        spec_algorithm=spec_algorithm,
        is_draft_worker=is_draft_worker,
    ):
        _run_flashinfer_autotune(
            server_args=server_args,
            model_config=model_config,
            dtype=dtype,
            device=device,
            forward_stream=forward_stream,
            req_to_token_pool_size=req_to_token_pool_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            dp_rank=dp_rank,
            dp_size=dp_size,
            moe_ep_size=moe_ep_size,
            dummy_run_callable=dummy_run_callable,
        )


def _run_flashinfer_autotune(
    *,
    server_args: ServerArgs,
    model_config: ModelConfig,
    dtype: torch.dtype,
    device: str,
    forward_stream: torch.cuda.Stream,
    req_to_token_pool_size: int,
    tp_rank: int,
    tp_size: int,
    pp_rank: int,
    pp_size: int,
    dp_rank: int,
    dp_size: int,
    moe_ep_size: int,
    dummy_run_callable,
):
    """Run flashinfer autotune."""
    from flashinfer.autotuner import autotune

    cache_path = _flashinfer_autotune_cache_path(
        server_args=server_args,
        model_config=model_config,
        dtype=dtype,
        device=device,
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        dp_rank=dp_rank,
        dp_size=dp_size,
        moe_ep_size=moe_ep_size,
    )
    logger.info("Running FlashInfer autotune with cache: %s", cache_path)

    # Run warmup on the non-default stream to avoid NCCL 2.29+ cudaMemcpyBatchAsync
    # calls on default stream (unsupported by CUDA) when --enable-symm-mem is used.
    forward_stream.wait_stream(torch.cuda.current_stream())
    with torch.get_device_module(device).stream(forward_stream):
        with torch.inference_mode(), autotune(True, cache=str(cache_path)):
            dummy_run_callable(batch_size=req_to_token_pool_size)
    torch.cuda.current_stream().wait_stream(forward_stream)
    logger.info("FlashInfer autotune completed.")
