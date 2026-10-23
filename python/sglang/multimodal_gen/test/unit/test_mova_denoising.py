import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

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
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention import layer
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
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
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import SyncOnRead
from sglang.test.test_utils import CustomTestCase


class _HostSchedule(torch.Tensor):
    # a host schedule that counts its uploads to the device
    uploads = 0

    def to(self, *args, **kwargs):
        _HostSchedule.uploads += 1
        return super().to(*args, **kwargs).as_subclass(SyncOnRead)


def _new_stage(video_dit, audio_dit, bridge):
    stage = object.__new__(mova.MOVADenoisingStage)
    stage.video_dit = video_dit
    stage.video_dit_2 = None
    stage.audio_dit = audio_dit
    stage.dual_tower_bridge = bridge
    stage._component_residency_manager = None
    stage._rope_freqs_cache_key = None
    stage._rope_freqs_cache = None
    return stage


class TestMOVADenoisingLoop(CustomTestCase):
    def _run_loop(self, paired_timesteps):
        stage = _new_stage(SimpleNamespace(require_vae_embedding=False), None, None)
        stage._rope_freqs_cache_key = "previous request"
        forward_timesteps = []

        def inference_single_step(**kwargs):
            forward_timesteps.append(
                (
                    kwargs["timestep"].as_subclass(torch.Tensor),
                    kwargs["audio_timestep"].as_subclass(torch.Tensor),
                )
            )
            return (
                kwargs["visual_latents"] * 0.1 + kwargs["context"].mean(),
                kwargs["audio_latents"] * 0.2 + kwargs["context"].mean(),
            )

        stage.inference_single_step = inference_single_step
        scheduler = FlowMatchPairScheduler(shift=5.0)
        scheduler.set_timesteps(3, denoising_strength=1.0, shift=5.0)
        batch = SimpleNamespace(
            paired_timesteps=paired_timesteps(scheduler.get_pairs()),
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
        with ExitStack() as stack:
            for name, value in (
                ("get_local_torch_device", torch.device("cpu")),
                ("get_classifier_free_guidance_rank", 0),
            ):
                stack.enter_context(patch.object(mova, name, return_value=value))
            for name in (
                "_maybe_compile_dits",
                "_ensure_shared_models_on_device",
                "_manage_video_dit_use",
            ):
                stack.enter_context(patch.object(stage, name, create=True))
            stage.forward(batch, server_args)
        return batch, forward_timesteps, stage

    def test_steps_never_upload_or_read_device_timesteps(self):
        expected, expected_timesteps, _ = self._run_loop(lambda pairs: pairs)
        _HostSchedule.uploads = 0
        actual, actual_timesteps, stage = self._run_loop(
            lambda pairs: pairs.as_subclass(_HostSchedule)
        )
        # one upload per request; the scheduler steps on the host schedule
        self.assertEqual(_HostSchedule.uploads, 1)
        for name in ("latents", "audio_latents"):
            self.assertTrue(
                torch.equal(
                    getattr(actual, name).as_subclass(torch.Tensor),
                    getattr(expected, name),
                )
            )
        self.assertEqual(len(actual_timesteps), 6)
        for (video, audio), (ref_video, ref_audio) in zip(
            actual_timesteps, expected_timesteps, strict=True
        ):
            self.assertEqual(video.shape, (1,))
            self.assertTrue(torch.equal(video, ref_video))
            self.assertTrue(torch.equal(audio, ref_audio))
        # the request-sized RoPE tables are freed before decoding
        self.assertIsNone(stage._rope_freqs_cache_key)


# the models' fused layers need a GPU on a GPU platform; elsewhere run on the CPU,
# in fp32 there (the time embedding's fp32 autocast is a GPU feature)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


class _CountedUploads(torch.Tensor):
    uploads = 0

    def to(self, *args, **kwargs):
        _CountedUploads.uploads += 1
        return super().to(*args, **kwargs).as_subclass(torch.Tensor)


class TestMOVARopeTables(CustomTestCase):
    @torch.no_grad()
    def test_rope_tables_built_once_per_request(self):
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
        with ExitStack() as stack:
            for name, value in (
                ("get_attn_backend", SDPABackend),
                ("get_compute_dtype", _DTYPE),
                ("_resolve_sp_attention_mode", ("ulysses", False)),
                ("_maybe_install_backend_autotune", None),
            ):
                stack.enter_context(patch.object(layer, name, return_value=value))
            # tiny shapes: run every custom op's reference path
            stack.enter_context(
                patch.object(CustomOp, "dispatch_forward", lambda op: op.forward_native)
            )
            stack.enter_context(set_forward_context(0, None))
            torch.manual_seed(0)
            # one production-sized head per tower: the GPU RoPE kernel expects it
            video_dit = WanModel(
                MOVAVideoConfig(
                    arch_config=MOVAVideoArchConfig(
                        dim=128,
                        in_dim=4,
                        ffn_dim=64,
                        out_dim=4,
                        text_dim=32,
                        freq_dim=16,
                        num_heads=1,
                        num_layers=3,
                        require_vae_embedding=False,
                    )
                ),
                hf_config={},
            )
            audio_dit = WanAudioModel(
                MOVAAudioConfig(
                    arch_config=MOVAAudioArchConfig(
                        dim=128,
                        in_dim=8,
                        ffn_dim=64,
                        out_dim=8,
                        text_dim=32,
                        freq_dim=16,
                        num_heads=1,
                        num_layers=2,
                        patch_size=(1,),
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
            for dit in (video_dit, audio_dit):
                dit._init_freqs()
                dit.freqs = tuple(
                    freq.as_subclass(_CountedUploads) for freq in dit.freqs
                )
            stage = _new_stage(video_dit, audio_dit, bridge)
            visual = torch.randn(1, 4, 3, 4, 6, device=_DEVICE)
            audio = torch.randn(1, 8, 10, device=_DEVICE)
            context = torch.randn(1, 5, 32, device=_DEVICE)

            def run_steps(reset_each_step):
                outputs = []
                for t in (900.0, 500.0):
                    if reset_each_step:
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

            # the per-forward construction the memo replaced
            per_forward = run_steps(reset_each_step=True)
            stage._rope_freqs_cache_key = None
            _CountedUploads.uploads = 0
            with patch.object(
                bridge, "build_aligned_freqs", wraps=bridge.build_aligned_freqs
            ) as build_bridge_freqs:
                built_once = run_steps(reset_each_step=False)
            # three frequency tables per tower and one bridge build for the request
            self.assertEqual(_CountedUploads.uploads, 6)
            self.assertEqual(build_bridge_freqs.call_count, 1)
            for actual, expected in zip(built_once, per_forward, strict=True):
                for actual_tensor, expected_tensor in zip(
                    actual, expected, strict=True
                ):
                    self.assertTrue(torch.equal(actual_tensor, expected_tensor))

            # the bridge layers share the two cos/sin caches instead of each
            # converting the bridge RoPE again
            stage._rope_freqs_cache_key = None
            with patch.object(
                mova, "rope_cos_sin_cache", wraps=mova.rope_cos_sin_cache
            ) as build_bridge_cache:
                run_steps(reset_each_step=False)
            self.assertEqual(build_bridge_cache.call_count, 2)


if __name__ == "__main__":
    unittest.main()
