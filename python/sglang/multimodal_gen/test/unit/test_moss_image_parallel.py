# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from sglang.multimodal_gen.configs.models.dits.moss_image import (
    MOSSImageArchConfig,
    MOSSImageDiTConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.moss_image import (
    MOSSImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.moss_image import MOSSImageSamplingParams
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import moss_image
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.moss_image import (
    MOSSImageDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args


def _worker(mode):
    set_global_server_args(
        SimpleNamespace(
            attention_backend="torch_sdpa",
            attention_backend_config={},
            kv_gather_degree=1,
            sp_split_auto=False,
            enable_cfg_parallel=mode == "cfg",
        )
    )
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=2 if mode == "sp" else 1,
        ulysses_degree=2 if mode == "sp" else 1,
        cfg_degree=2 if mode == "cfg" else 1,
    )
    config = MOSSImageDiTConfig(
        arch_config=MOSSImageArchConfig(
            dim=64,
            ffn_dim=128,
            num_heads=4,
            num_layers=2,
            text_dim=32,
            freq_dim=16,
            latent_channels=8,
        )
    )
    torch.manual_seed(42)
    with patch.object(moss_image, "get_sp_world_size", return_value=1):
        reference = moss_image.MOSSImageTransformer(config).eval()
    candidate = moss_image.MOSSImageTransformer(config).eval()
    candidate.load_state_dict(reference.state_dict())
    with torch.inference_mode(), set_forward_context(0, None):
        for length in (4, 5, 7):
            torch.manual_seed(length)
            x, text = torch.randn(2, 8, 2, 3), torch.randn(2, length, 32)
            timestep = torch.tensor([630.0])
            if mode == "sp":
                for skip in ((), (0,), (1,)):
                    expected = reference(x, timestep, text, skip_layers=skip)
                    prepared = candidate.prepare_conditioning(
                        text, None, 2, 3, torch.float32, batch_samples=True
                    )
                    actual = candidate(
                        x,
                        timestep,
                        text,
                        prepared_conditioning=prepared,
                        skip_layers=skip,
                    )
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            else:
                pipeline = MOSSImagePipelineConfig(dit_config=config)
                stage = object.__new__(MOSSImageDenoisingStage)
                stage._predict_noise = lambda **kw: candidate(
                    kw["latent_model_input"],
                    kw["timestep"],
                    kw["encoder_hidden_states"],
                    skip_layers=kw.get("skip_layers", ()),
                )
                for slg_scale in (0.0, 3.0):
                    params = MOSSImageSamplingParams(
                        uncond_skip_layers=[1], slg_layers=[0], slg_scale=slg_scale
                    )
                    batch = Req(
                        sampling_params=params, timesteps=torch.tensor([1000.0, 630.0])
                    )
                    policy = pipeline.cfg_policy.build(
                        batch,
                        {},
                        {"encoder_hidden_states": text},
                        {"encoder_hidden_states": -text},
                    )
                    results = [
                        stage._predict_noise_with_cfg(
                            current_model=candidate,
                            latent_model_input=x,
                            timestep=timestep,
                            batch=batch,
                            timestep_index=1,
                            attn_metadata=None,
                            target_dtype=torch.float32,
                            current_guidance_scale=3.0,
                            cfg_policy=policy,
                            cfg_gate_state=None,
                            server_args=SimpleNamespace(
                                pipeline_config=pipeline, enable_cfg_parallel=parallel
                            ),
                            guidance=None,
                            latents=x,
                        )
                        for parallel in (False, True)
                    ]
                    torch.testing.assert_close(*results, rtol=0, atol=0)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["sp", "cfg"])
def test_two_rank_moss_image_matches_serial(mode):
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
            "MOSS_IMAGE_PARALLEL_WORKER": mode,
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    if mode := os.environ.get("MOSS_IMAGE_PARALLEL_WORKER"):
        _worker(mode)
    else:
        sys.exit(pytest.main([__file__, "-v"]))
