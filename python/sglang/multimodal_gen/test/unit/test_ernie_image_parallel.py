# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from sglang.multimodal_gen.configs.models.dits.ernie_image import (
    ErnieImageArchConfig,
    ErnieImageDitConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import ernie_image
from sglang.multimodal_gen.runtime.server_args import set_global_server_args


def _worker(mode):
    set_global_server_args(
        SimpleNamespace(
            attention_backend="torch_sdpa",
            attention_backend_config={},
            kv_gather_degree=1,
            sp_split_auto=False,
        )
    )
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=2,
        ulysses_degree=2 if mode == "ulysses" else 1,
        ring_degree=1 if mode == "ulysses" else 2,
    )
    config = ErnieImageDitConfig(
        arch_config=ErnieImageArchConfig(
            in_channels=4,
            out_channels=4,
            num_layers=2,
            num_attention_heads=4,
            attention_head_dim=16,
            ffn_hidden_size=128,
            text_in_dim=32,
            rope_axes_dim=(4, 6, 6),
        )
    )
    torch.manual_seed(42)
    model = ernie_image.ErnieImageTransformer2DModel(config, {}).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.1)
    with torch.inference_mode(), set_forward_context(0, None):
        for height, width, length, masked in [
            (3, 4, 3, False),
            (3, 4, 5, False),
            (3, 3, 3, False),
            (3, 4, 5, True),
            (3, 4, 3, False),
        ]:
            torch.manual_seed(length)
            latents = torch.randn(2, 4, height, width)
            text = torch.randn(2, length, 32)
            mask = None
            if masked:
                mask = torch.ones(2, length, dtype=torch.bool)
                mask[0, -2:] = False
            args = (latents, text, torch.tensor([630.0, 500.0]))
            kwargs = {"encoder_hidden_states_mask": mask}
            with patch.object(ernie_image, "get_sp_world_size", return_value=1):
                expected = model(*args, **kwargs)
            with patch.object(
                ernie_image, "gather_seq", wraps=ernie_image.gather_seq
            ) as gather:
                actual = model(*args, **kwargs)
            sharded = mode == "ulysses" and height * width % 2 == 0 and not masked
            assert gather.call_count == int(sharded)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.version.hip is not None, reason="CPU workers hide GPUs")
@pytest.mark.parametrize("mode", ["ulysses", "ring"])
def test_two_rank_ernie_image_matches_dense(mode):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            __file__,
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "cpu",
            "ERNIE_IMAGE_PARALLEL_WORKER": mode,
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    if mode := os.environ.get("ERNIE_IMAGE_PARALLEL_WORKER"):
        _worker(mode)
    else:
        sys.exit(pytest.main([__file__, "-v"]))
