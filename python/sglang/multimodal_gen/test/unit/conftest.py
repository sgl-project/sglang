# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.srt.utils.network import get_free_port_below_ephemeral


@pytest.fixture(scope="module")
def single_process_model_parallel():
    with pytest.MonkeyPatch.context() as patch:
        if not model_parallel_is_initialized():
            for key, value in dict(
                MASTER_ADDR="127.0.0.1",
                MASTER_PORT=str(get_free_port_below_ephemeral()),
                RANK="0",
                LOCAL_RANK="0",
                WORLD_SIZE="1",
            ).items():
                patch.setenv(key, value)
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
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
        kv_gather_degree=1,
        sp_split_auto=False,
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


@contextmanager
def mock_cuda_platform(
    *,
    memory_gb: int = 80,
    available_memory_gb: int | dict[int, int] | None = None,
):
    """current_platform as an idle CUDA device of memory_gb GB, so ServerArgs resolution
    and parallelism auto-tuning run without a GPU."""

    def get_available_gpu_memory(device_id=0, **_kwargs):
        if isinstance(available_memory_gb, dict):
            return available_memory_gb[device_id]
        if available_memory_gb is not None:
            return available_memory_gb
        return memory_gb

    with (
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_cpu",
            return_value=False,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_mps",
            return_value=False,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_cuda",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_rocm",
            return_value=False,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.get_device_total_memory",
            return_value=memory_gb * 1024**3,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.get_available_gpu_memory",
            side_effect=get_available_gpu_memory,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.enable_dit_layerwise_offload_by_default",
            return_value=True,
        ),
        # checkpoint sizing reads the local HF cache; tests that need it pass a size
        patch(
            "sglang.multimodal_gen.runtime.server_args.auto_tune.checkpoint_component_sizes",
            return_value=None,
        ),
    ):
        yield


@pytest.fixture
def cuda_platform():
    with mock_cuda_platform():
        yield
