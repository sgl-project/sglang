import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2 import (
    denoising as ltx2_denoising,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)
from sglang.test.test_utils import CustomTestCase

_PASS_NAMES = ("cond", "neg", "perturbed", "modality")
_SCALES = dict(
    video_cfg_scale=3.0,
    video_stg_scale=1.0,
    video_rescale_scale=0.7,
    video_modality_scale=3.0,
    audio_cfg_scale=7.0,
    audio_stg_scale=1.0,
    audio_rescale_scale=0.7,
    audio_modality_scale=3.0,
)


class TestLTX2CFGParallelGuidance(CustomTestCase):
    """Rank-split LTX-2 multi-guidance must combine exactly like one GPU."""

    def setUp(self):
        gen = torch.Generator().manual_seed(0)
        self.sigma = 0.73
        self.video_latents = torch.randn(1, 24, 16, generator=gen).bfloat16()
        self.audio_latents = torch.randn(1, 6, 8, generator=gen).bfloat16()
        self.video_velocities = {
            name: torch.randn(1, 24, 16, generator=gen).bfloat16().float()
            for name in _PASS_NAMES
        }
        self.audio_velocities = {
            name: torch.randn(1, 6, 8, generator=gen).bfloat16().float()
            for name in _PASS_NAMES
        }

    def _single_gpu(self, latents, velocities, prefix):
        x0 = {
            name: LTX2DenoisingStage._ltx2_velocity_to_x0(latents, v, self.sigma)
            for name, v in velocities.items()
        }
        return LTX2DenoisingStage._ltx2_calculate_guided_x0(
            cond=x0["cond"],
            uncond_text=x0["neg"],
            uncond_perturbed=x0["perturbed"],
            uncond_modality=x0["modality"],
            cfg_scale=_SCALES[f"{prefix}_cfg_scale"],
            stg_scale=_SCALES[f"{prefix}_stg_scale"],
            rescale_scale=_SCALES[f"{prefix}_rescale_scale"],
            modality_scale=_SCALES[f"{prefix}_modality_scale"],
        )

    def _combine_on_rank(self, rank, world_size, all_reduce):
        local = [n for i, n in enumerate(_PASS_NAMES) if i % world_size == rank]
        with patch.object(ltx2_denoising, "cfg_model_parallel_all_reduce", all_reduce):
            return LTX2DenoisingStage._ltx2_combine_guided_x0_parallel_av(
                video_latents=self.video_latents,
                audio_latents=self.audio_latents,
                local_video_velocities={n: self.video_velocities[n] for n in local},
                local_audio_velocities={n: self.audio_velocities[n] for n in local},
                video_sigma=self.sigma,
                audio_sigma=self.sigma,
                **_SCALES,
            )

    def test_rank_split_matches_single_gpu_bitwise(self):
        expected_video = self._single_gpu(
            self.video_latents, self.video_velocities, "video"
        )
        expected_audio = self._single_gpu(
            self.audio_latents, self.audio_velocities, "audio"
        )
        for world_size in (2, 4):
            # First pass records each rank's all-reduce contributions; the second
            # replays the summed tensors as the collective would return them.
            contributions = [[] for _ in range(world_size)]
            for rank in range(world_size):

                def record(t, rank=rank):
                    contributions[rank].append(t.clone())
                    return t

                self._combine_on_rank(rank, world_size, record)

            for rank in range(world_size):
                calls = iter(range(len(_PASS_NAMES)))

                def summed(t):
                    i = next(calls)
                    total = contributions[0][i].clone()
                    for other in contributions[1:]:
                        total += other[i]
                    return total

                video, audio = self._combine_on_rank(rank, world_size, summed)
                self.assertTrue(torch.equal(video, expected_video))
                self.assertTrue(torch.equal(audio, expected_audio))


if __name__ == "__main__":
    unittest.main()
