# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.server_args import set_global_server_args


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
