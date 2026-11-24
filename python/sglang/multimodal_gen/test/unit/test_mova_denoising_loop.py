import unittest
from types import SimpleNamespace
from unittest.mock import DEFAULT, patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.bridges.mova_dual_tower import (
    MOVADualTowerArchConfig,
    MOVADualTowerConfig,
)
from sglang.multimodal_gen.configs.models.dits.mova_audio import (
    MOVAAudioArchConfig,
    MOVAAudioConfig,
)
from sglang.multimodal_gen.configs.models.dits.mova_video import (
    MOVAVideoArchConfig,
    MOVAVideoConfig,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.bridges.mova_dual_tower import (
    DualTowerConditionalBridge,
)
from sglang.multimodal_gen.runtime.models.dits.mova_audio_dit import WanAudioModel
from sglang.multimodal_gen.runtime.models.dits.mova_video_dit import WanModel
from sglang.multimodal_gen.runtime.models.schedulers.flow_match_pair import (
    FlowMatchPairScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    mova,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.mova import (
    MOVADenoisingStage,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import (
    CountedUploads,
    SyncOnRead,
    tiny_dit_runtime,
)

# the models' fused layers need a GPU on a GPU platform; elsewhere run on the CPU,
# in fp32 there (the time embedding's fp32 autocast is a GPU feature)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

_POS = 1.0
_NEG = 0.5
_GUIDANCE = 5.0
_STEPS = 2


class _AccumulatingScheduler:
    def step_from_to(self, model_output, timestep, next_timestep, sample):
        return sample + model_output


def _predict(visual_dit, latents, audio_latents, y, context, *args):
    value = context.flatten()[0].item()
    return torch.full_like(latents, value), torch.full_like(audio_latents, value)


def _run(*, do_cfg: bool, enable_cfg_parallel: bool = False, cfg_rank: int = 0):
    stage = object.__new__(MOVADenoisingStage)
    stage.video_dit = SimpleNamespace(require_vae_embedding=False)
    stage.video_dit_2 = None
    stage._torch_compiled = False
    stage._predict = _predict
    batch = SimpleNamespace(
        paired_timesteps=torch.tensor([[1000.0, 1000.0], [500.0, 500.0]]),
        scheduler=_AccumulatingScheduler(),
        y=torch.zeros(1),
        image_latent=None,
        latents=torch.zeros(1, 2, 1, 2, 2),
        audio_latents=torch.zeros(1, 2, 4),
        prompt_embeds=[torch.full((1, 3, 4), _POS)],
        negative_prompt_embeds=[torch.full((1, 3, 4), _NEG)],
        fps=24,
        do_classifier_free_guidance=do_cfg,
        guidance_scale=_GUIDANCE,
        guidance_rescale=0.0,
        is_warmup=True,
        metrics=None,
        perf_dump_path=None,
    )
    server_args = SimpleNamespace(
        enable_torch_compile=False,
        enable_cfg_parallel=enable_cfg_parallel,
        pipeline_config=SimpleNamespace(boundary_ratio=None),
    )
    # stands in for the cfg all-reduce by adding the other rank's partial
    other_partial = (1 - _GUIDANCE) * _NEG if cfg_rank == 0 else _GUIDANCE * _POS
    with (
        patch.object(mova, "get_local_torch_device", return_value="cpu"),
        patch.object(mova, "get_classifier_free_guidance_rank", return_value=cfg_rank),
        patch.object(
            mova,
            "cfg_model_parallel_all_reduce",
            side_effect=lambda partial: partial + other_partial,
        ),
    ):
        stage.forward(batch, server_args)
    return batch


class _HostSchedule(CountedUploads):
    # the uploaded copy is a device tensor: reading it on the host raises
    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs).as_subclass(SyncOnRead)


def _new_stage(video_dit, audio_dit, bridge):
    stage = object.__new__(MOVADenoisingStage)
    stage.video_dit = video_dit
    stage.video_dit_2 = None
    stage.audio_dit = audio_dit
    stage.dual_tower_bridge = bridge
    stage._component_residency_manager = None
    stage._rope_freqs_cache_key = None
    stage._rope_freqs_cache = None
    return stage


class TestMOVADenoisingLoop(unittest.TestCase):
    def _assert_stepped_by(self, batch, value):
        torch.testing.assert_close(
            batch.latents, torch.full_like(batch.latents, _STEPS * value)
        )
        torch.testing.assert_close(
            batch.audio_latents, torch.full_like(batch.audio_latents, _STEPS * value)
        )

    def test_without_cfg_steps_latents(self):
        self._assert_stepped_by(_run(do_cfg=False), _POS)

    def test_serial_cfg_steps_with_combined_prediction(self):
        self._assert_stepped_by(_run(do_cfg=True), _NEG + _GUIDANCE * (_POS - _NEG))

    def test_cfg_parallel_steps_with_combined_prediction_on_every_rank(self):
        for cfg_rank in (0, 1):
            with self.subTest(cfg_rank=cfg_rank):
                batch = _run(do_cfg=True, enable_cfg_parallel=True, cfg_rank=cfg_rank)
                self._assert_stepped_by(batch, _NEG + _GUIDANCE * (_POS - _NEG))

    def test_steps_never_upload_or_read_device_timesteps(self):
        stage = _new_stage(SimpleNamespace(require_vae_embedding=False), None, None)
        stage._rope_freqs_cache_key = "previous request"
        seen = []

        def inference_single_step(**kwargs):
            seen.append((kwargs["timestep"], kwargs["audio_timestep"]))
            return kwargs["visual_latents"], kwargs["audio_latents"]

        stage.inference_single_step = inference_single_step
        scheduler = FlowMatchPairScheduler(shift=5.0)
        scheduler.set_timesteps(3, denoising_strength=1.0, shift=5.0)
        pairs = scheduler.get_pairs()
        batch = SimpleNamespace(
            paired_timesteps=pairs.as_subclass(_HostSchedule),
            scheduler=scheduler,
            y=None,
            image_latent=None,
            latents=torch.ones(1, 2, 1, 2, 2),
            audio_latents=torch.ones(1, 2, 3),
            prompt_embeds=[torch.ones(1, 2, 4)],
            negative_prompt_embeds=[torch.zeros(1, 2, 4)],
            do_classifier_free_guidance=True,
            guidance_scale=5.0,
            guidance_rescale=0.0,
            fps=24.0,
            is_warmup=False,
            metrics=None,
            perf_dump_path=None,
            extra_step_kwargs=None,
        )
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(boundary_ratio=None),
            enable_cfg_parallel=False,
        )
        CountedUploads.uploads = 0
        with (
            patch.multiple(
                mova,
                get_local_torch_device=lambda: torch.device("cpu"),
                get_classifier_free_guidance_rank=lambda: 0,
            ),
            patch.multiple(
                stage,
                create=True,
                _maybe_compile_dits=DEFAULT,
                _ensure_shared_models_on_device=DEFAULT,
                _manage_video_dit_use=DEFAULT,
            ),
        ):
            stage.forward(batch, server_args)

        # one upload per request; each CFG branch gets its step's device slice
        self.assertEqual(CountedUploads.uploads, 1)
        torch.testing.assert_close(
            [
                (v.as_subclass(torch.Tensor), a.as_subclass(torch.Tensor))
                for v, a in seen
            ],
            [(pair[:1], pair[1:]) for pair in pairs for _ in range(2)],
            rtol=0,
            atol=0,
        )
        # the request-sized RoPE tables are freed before decoding
        self.assertIsNone(stage._rope_freqs_cache_key)


@pytest.mark.usefixtures("single_process_model_parallel")
class TestMOVARopeTables(unittest.TestCase):
    @torch.no_grad()
    def test_rope_tables_built_once_per_request(self):
        with tiny_dit_runtime(_DTYPE), set_forward_context(0, None):
            torch.manual_seed(0)
            # one production-sized head per tower: the GPU RoPE kernel expects it
            tower = dict(dim=128, ffn_dim=64, text_dim=32, freq_dim=16, num_heads=1)
            video_dit = WanModel(
                MOVAVideoConfig(
                    arch_config=MOVAVideoArchConfig(
                        in_dim=4,
                        out_dim=4,
                        num_layers=3,
                        require_vae_embedding=False,
                        **tower,
                    )
                ),
                hf_config={},
            )
            audio_dit = WanAudioModel(
                MOVAAudioConfig(
                    arch_config=MOVAAudioArchConfig(
                        in_dim=8, out_dim=8, num_layers=2, patch_size=(1,), **tower
                    )
                ),
                hf_config={},
            )
            bridge = DualTowerConditionalBridge(
                MOVADualTowerConfig(
                    arch_config=MOVADualTowerArchConfig(
                        visual_layers=3,
                        audio_layers=2,
                        visual_hidden_dim=128,
                        audio_hidden_dim=128,
                        head_dim=128,
                    )
                )
            )
            for module in (video_dit, audio_dit, bridge):
                module.to(device=_DEVICE, dtype=_DTYPE)
                for parameter in module.parameters():
                    parameter.normal_(std=0.05)
            stage = _new_stage(video_dit, audio_dit, bridge)
            visual = torch.randn(1, 4, 3, 4, 6, device=_DEVICE)
            audio = torch.randn(1, 8, 10, device=_DEVICE)
            context = torch.randn(1, 5, 32, device=_DEVICE)

            def run_steps(rebuild_each_step):
                outputs = []
                for t in (900.0, 500.0):
                    if rebuild_each_step:
                        stage._rope_freqs_cache_key = None
                    outputs.append(
                        stage.inference_single_step(
                            video_dit,
                            visual,
                            audio,
                            None,
                            context,
                            torch.tensor([t], device=_DEVICE),
                            torch.tensor([t - 1.0], device=_DEVICE),
                            24.0,
                        )
                    )
                return outputs

            with (
                patch.object(
                    bridge, "build_aligned_freqs", wraps=bridge.build_aligned_freqs
                ) as build_bridge_freqs,
                patch.object(
                    mova, "rope_cos_sin_cache", wraps=mova.rope_cos_sin_cache
                ) as build_bridge_cache,
            ):
                built_once = run_steps(rebuild_each_step=False)
            # one build per request: the bridge layers share its two cos/sin caches
            self.assertEqual(build_bridge_freqs.call_count, 1)
            self.assertEqual(build_bridge_cache.call_count, 2)
            # the steps leave the shared tables intact
            torch.testing.assert_close(
                built_once, run_steps(rebuild_each_step=True), rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
