# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args

LATENT_SHAPE = (1, 16, 1, 32, 32)
SOUND_SHAPE = (1, 64, 32)
COND_TOKEN = 1
UNCOND_TOKEN = 101


class _Transformer:
    """BF16 stand-in whose output depends on the latents, the control map and the
    text tokens that fall inside ``max_text_seq_len`` and the mask."""

    def __init__(self):
        self.weights = {}

    def _weight(self, key, shape):
        if (key, shape) not in self.weights:
            generator = torch.Generator().manual_seed(hash(key) % (2**31))
            self.weights[(key, shape)] = torch.randn(shape, generator=generator)
        return self.weights[(key, shape)]

    def __call__(
        self,
        latents,
        text_ids,
        text_mask,
        max_text_seq_len,
        control_latents,
        sound_latents,
    ):
        visible = text_ids[:, :max_text_seq_len] * text_mask[:, :max_text_seq_len]
        keys = [(tuple(row.tolist()), control_latents is not None) for row in visible]

        def rows(x, shape):
            return torch.cat(
                [
                    (x[b : b + 1].float() * 0.5 + self._weight(key, shape)).to(
                        torch.bfloat16
                    )
                    for b, key in enumerate(keys)
                ]
            )

        video = rows(latents, LATENT_SHAPE)
        if sound_latents is None:
            return video
        return video, rows(sound_latents, SOUND_SHAPE)

    def set_denoising_step(self, **kwargs):
        pass

    def reset_denoising_step(self):
        pass

    def reset_cache(self):
        pass


class _Scheduler:
    def set_timesteps(self, *args, **kwargs):
        pass

    def step(self, noise_pred, t, latents, **kwargs):
        return (latents - 0.25 * noise_pred,)


def _make_stage():
    stage = Cosmos3DenoisingStage.__new__(Cosmos3DenoisingStage)
    stage.server_args = SimpleNamespace(comfyui_mode=True)
    stage.scheduler = _Scheduler()
    stage.transformer = _Transformer()
    stage._logged_parallel_config = False
    stage._logged_cfg_split = False
    stage._run_transformer = lambda **kw: stage.transformer(
        kw["latents"],
        kw["text_ids"],
        kw["text_mask"],
        kw["max_text_seq_len"],
        kw["control_latents"],
        kw["sound_latents"],
    )
    return stage


def _tokens(first, length, padded_len):
    """Right-padded IDs and mask, as the encoding stage stores them."""
    ids = torch.zeros((1, padded_len), dtype=torch.long)
    ids[0, :length] = torch.arange(first, first + length)
    mask = torch.zeros_like(ids)
    mask[0, :length] = 1
    return ids, mask


def _make_batch(cond_len, uncond_len, control, sound):
    generator = torch.Generator().manual_seed(0)
    padded_len = max(cond_len, uncond_len)
    cond_ids, cond_mask = _tokens(COND_TOKEN, cond_len, padded_len)
    uncond_ids, uncond_mask = _tokens(UNCOND_TOKEN, uncond_len, padded_len)
    return SimpleNamespace(
        latents=torch.randn(LATENT_SHAPE, generator=generator).to(torch.bfloat16),
        audio_latents=(
            torch.randn(SOUND_SHAPE, generator=generator).to(torch.bfloat16)
            if sound
            else None
        ),
        timesteps=torch.tensor([900.0, 600.0, 300.0]),
        guidance_scale=4.0,
        generator=None,
        seed=0,
        scheduler=None,
        rollout=False,
        rollout_controllers={},
        profile=False,
        is_warmup=False,
        sampling_params=SimpleNamespace(
            control_guidance=1.5 if control else 1.0,
            guidance_interval=None,
            control_guidance_interval=None,
        ),
        extra={
            "cond_text_ids": cond_ids,
            "cond_text_mask": cond_mask,
            "uncond_text_ids": uncond_ids,
            "uncond_text_mask": uncond_mask,
            "cond_text_seq_len": cond_len,
            "uncond_text_seq_len": uncond_len,
            "video_shape": (1, 4, 4),
            "control_latents": [torch.zeros(1)] if control else None,
        },
    )


def _denoise(enable_cfg_parallel, cond_len, uncond_len, control, sound):
    batch = _make_batch(cond_len, uncond_len, control, sound)
    server_args = SimpleNamespace(
        enable_cfg_parallel=enable_cfg_parallel, pipeline_config=None
    )
    out = _make_stage()._denoise_once(batch, server_args)
    return [out.latents] + ([out.audio_latents] if sound else [])


def _worker():
    set_global_server_args(
        SimpleNamespace(
            attention_backend="torch_sdpa",
            attention_backend_config={},
            kv_gather_degree=1,
            sp_split_auto=False,
            enable_cfg_parallel=True,
        )
    )
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=1, ulysses_degree=1, cfg_degree=2
    )
    # Control-free text CFG with equal (batched) and unequal (serial) text
    # lengths, and transfer control-CFG with three branches over two ranks.
    mismatches = []
    for cond_len, uncond_len, control in (
        (8, 8, False),
        (8, 3, False),
        (3, 8, False),
        (8, 3, True),
    ):
        for sound in (False, True):
            case = dict(cond=cond_len, uncond=uncond_len, control=control, sound=sound)
            serial = _denoise(False, cond_len, uncond_len, control, sound)
            parallel = _denoise(True, cond_len, uncond_len, control, sound)
            for expected, actual in zip(serial, parallel, strict=True):
                try:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                except AssertionError as e:
                    mismatches.append(f"{case}: {e}")
                    break
    rank = get_classifier_free_guidance_rank()
    assert not mismatches, f"rank {rank}:\n" + "\n".join(mismatches)
    print(f"rank {rank} ok")
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.version.hip is not None,
    reason="ROCm builds query GPU 0 while importing sglang, which fails with the "
    "GPUs hidden for these CPU workers",
)
def test_two_rank_cfg_parallel_matches_single_gpu_bf16():
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
            "COSMOS3_CFG_PARALLEL_WORKER": "1",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    if os.environ.get("COSMOS3_CFG_PARALLEL_WORKER"):
        _worker()
    else:
        sys.exit(pytest.main([__file__, "-v"]))
