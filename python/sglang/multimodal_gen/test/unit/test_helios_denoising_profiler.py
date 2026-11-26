import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.helios import (
    HeliosArchConfig,
    HeliosConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.helios import HeliosT2VConfig
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import helios
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.helios_denoising import (
    HeliosChunkedDenoisingStage,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import tiny_dit_runtime

# the model's fused layers need a GPU on a GPU platform; elsewhere run on the CPU
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Transformer:
    def __call__(self, **kwargs):
        return torch.zeros_like(kwargs["hidden_states"])


class _Scheduler:
    def step(self, noise_pred, timestep, latents, return_dict=False):
        return (latents,)


class _Profiler:
    def __init__(self):
        self.steps = 0

    def step_denoising_step(self):
        self.steps += 1


class TestHeliosDenoisingProfiler(unittest.TestCase):
    def test_stage1_advances_profiler_once_per_timestep(self):
        stage = HeliosChunkedDenoisingStage.__new__(HeliosChunkedDenoisingStage)
        stage.transformer = _Transformer()
        stage.scheduler = _Scheduler()
        profiler = _Profiler()
        timesteps = torch.tensor([2.0, 1.0])

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.helios_denoising."
            "SGLDiffusionProfiler.get_instance",
            return_value=profiler,
        ):
            output = stage._denoise_one_chunk(
                latents=torch.ones(1, 2),
                prompt_embeds=torch.ones(1, 2),
                negative_prompt_embeds=torch.ones(1, 2),
                timesteps=timesteps,
                guidance_scale=1.0,
                indices_hidden_states=None,
                indices_latents_history_short=None,
                indices_latents_history_mid=None,
                indices_latents_history_long=None,
                latents_history_short=None,
                latents_history_mid=None,
                latents_history_long=None,
                target_dtype=torch.float32,
                device=torch.device("cpu"),
                batch=None,
                scheduler=stage.scheduler,
            )

        torch.testing.assert_close(output, torch.ones(1, 2))
        self.assertEqual(profiler.steps, len(timesteps))


@pytest.mark.usefixtures("single_process_model_parallel")
class TestHeliosStepHostOverhead(unittest.TestCase):
    def test_frame_indices_live_on_the_latent_device(self):
        # meta stands in for CUDA: host-built indices would be uploaded by every
        # forward's four RoPE builds with a blocking copy
        stage = HeliosChunkedDenoisingStage.__new__(HeliosChunkedDenoisingStage)
        stage.transformer = SimpleNamespace(in_channels=4, patch_size=(1, 2, 2))
        stage.scheduler = SimpleNamespace(
            set_timesteps=lambda *args, **kwargs: None, timesteps=torch.ones(2)
        )
        stage.server_args = SimpleNamespace(comfyui_mode=False)
        stage._current_batch_is_warmup = False
        stage._maybe_toggle_quality_fusions = lambda batch: None
        chunks = []

        def denoise_one_chunk(**kwargs):
            chunks.append(kwargs)
            return kwargs["latents"]

        stage._denoise_one_chunk = denoise_one_chunk
        pipeline_config = HeliosT2VConfig(
            num_latent_frames_per_chunk=2, history_sizes=[4, 2, 1]
        )
        expected = {
            "indices_latents_history_long": [1, 2, 3, 4],
            "indices_latents_history_mid": [5, 6],
            "indices_latents_history_short": [0, 7],
            "indices_hidden_states": [8, 9],
        }
        for device in ("cpu", "meta"):
            batch = SimpleNamespace(
                latents=torch.zeros(1, device=device),
                scheduler=None,
                guidance_scale=1.0,
                num_inference_steps=2,
                prompt_embeds=[torch.zeros(1, 2, 4)],
                negative_prompt_embeds=None,
                height=16,
                width=16,
                num_frames=5,
                generator=None,
            )
            chunks.clear()
            stage.forward(batch, SimpleNamespace(pipeline_config=pipeline_config))
            for name, values in expected.items():
                self.assertEqual(chunks[0][name].device.type, device, name)
                if device == "cpu":
                    self.assertEqual(chunks[-1][name].tolist(), [values], name)

    @torch.no_grad()
    def test_blocks_receive_fp32_modulation(self):
        with tiny_dit_runtime(torch.bfloat16), set_forward_context(0, None):
            torch.manual_seed(0)
            # one production-sized head: the GPU RoPE kernel expects it
            arch = HeliosArchConfig(
                num_attention_heads=1,
                attention_head_dim=128,
                in_channels=4,
                out_channels=4,
                text_dim=32,
                freq_dim=16,
                ffn_dim=64,
                num_layers=2,
            )
            model = helios.HeliosTransformer3DModel(
                HeliosConfig(arch_config=arch), hf_config={}
            ).to(device=_DEVICE, dtype=torch.bfloat16)
            modulation_dtypes = []
            for block in model.blocks:
                block.register_forward_pre_hook(
                    lambda module, args: modulation_dtypes.append(args[2].dtype)
                )
            indices = torch.arange(10, device=_DEVICE)
            prefix, long, mid, short, current = indices.split([1, 4, 2, 1, 2])
            bf16 = dict(device=_DEVICE, dtype=torch.bfloat16)
            model(
                hidden_states=torch.randn(1, 4, 2, 8, 8, **bf16),
                encoder_hidden_states=torch.randn(1, 6, 32, **bf16),
                timestep=torch.tensor([500.0], device=_DEVICE),
                indices_hidden_states=current[None],
                indices_latents_history_short=torch.cat([prefix, short])[None],
                indices_latents_history_mid=mid[None],
                indices_latents_history_long=long[None],
                latents_history_short=torch.randn(1, 4, 2, 8, 8, device=_DEVICE),
                latents_history_mid=torch.randn(1, 4, 2, 8, 8, device=_DEVICE),
                latents_history_long=torch.randn(1, 4, 4, 8, 8, device=_DEVICE),
            )
        # each block's temb.float() is then a no-op, not a full-sequence copy
        self.assertEqual(modulation_dtypes, [torch.float32] * arch.num_layers)


if __name__ == "__main__":
    unittest.main()
