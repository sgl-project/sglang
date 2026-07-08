# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.server_args import set_global_server_args


@pytest.fixture(scope="module", autouse=True)
def _omnidreams_model_parallel(request):
    """Initialise a 1-rank model-parallel group for OmniDreams test modules.

    OmniDreams' DiT builds Column/Row/MergedColumnParallelLinear directly (like
    ltx_2 / flux_2), which require an initialised TP group. Mirror the runtime
    (always initialised, world_size==1 on a single card). Idempotent across
    modules; gated to OmniDreams modules so other unit tests are unaffected.
    """
    if "omnidreams" not in request.module.__name__:
        yield
        return
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29505")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    # The forward/AR tests build untrained random-weight models; the flash /
    # mem-efficient SDPA backends are non-deterministic, which can flip a deep AR
    # rollout to NaN run-to-run. Pin the deterministic math backend for stability.
    import torch

    if torch.cuda.is_available():
        prev_flash = torch.backends.cuda.flash_sdp_enabled()
        prev_mem = torch.backends.cuda.mem_efficient_sdp_enabled()
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        try:
            yield
        finally:
            torch.backends.cuda.enable_flash_sdp(prev_flash)
            torch.backends.cuda.enable_mem_efficient_sdp(prev_mem)
        return
    yield


def _make_unit_server_args():
    dit_config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        boundary_ratio=None,
        arch_config=SimpleNamespace(in_channels=16, patch_size=2),
    )
    vae_config = SimpleNamespace(
        vae_tiling=False,
        arch_config=SimpleNamespace(
            vae_scale_factor=8,
            spatial_compression_ratio=8,
            z_dim=16,
            scale_factor_spatial=8,
        ),
        get_vae_scale_factor=lambda: 8,
    )
    pipeline_config = SimpleNamespace(
        dit_config=dit_config,
        vae_config=vae_config,
        dit_precision="bfloat16",
        vae_precision="bfloat16",
        get_latent_dtype=lambda dtype: dtype,
    )
    return SimpleNamespace(
        attention_backend=None,
        attention_backend_config=None,
        comfyui_mode=False,
        disable_autocast=False,
        enable_cfg_parallel=False,
        enable_breakable_cuda_graph=False,
        enable_layerwise_nvtx_marker=False,
        enable_torch_compile=False,
        model_loaded={},
        model_paths={},
        pipeline_config=pipeline_config,
    )


@pytest.fixture(autouse=True)
def default_global_server_args():
    previous = server_args_module._global_server_args
    set_global_server_args(_make_unit_server_args())
    try:
        yield
    finally:
        set_global_server_args(previous)
