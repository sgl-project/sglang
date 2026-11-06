# SPDX-License-Identifier: Apache-2.0
"""Same-step KV, window rebasing and sampler contracts for Physis H3."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.physis_h3 import (
    PhysisH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.physis_h3 import (
    PhysisH3SamplingParams,
    validate_physis_h3_sampling,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3DiTModel,
    MiniMaxH3Rope,
)
from sglang.multimodal_gen.runtime.models.dits.physis_h3 import (
    PhysisH3AttentionContext,
    physis_h3_frame_positions,
    physis_h3_positions,
    physis_h3_step,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.physis_denoising import (
    PhysisH3Denoiser,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPhysisH3(CustomTestCase):
    def test_seed_validation_precedes_device_sampling(self):
        """Out-of-range seeds must fail before rank-local generator overflow."""
        params = PhysisH3SamplingParams()
        for seed in (0, 2**31 - 1):
            params.seed = seed
            validate_physis_h3_sampling(params)
        for seed in (True, -1, 2**31, 2**63 - 1, [42]):
            with self.subTest(seed=seed):
                params.seed = seed
                with self.assertRaisesRegex(ValueError, "seed"):
                    validate_physis_h3_sampling(params)

    def test_rejects_ignored_acceleration_options(self):
        def args(**overrides):
            values = dict(
                encoder_parallel="replicate",
                attention_backend=None,
                resolve_component_attention_backend=lambda name: (None, None),
                kv_cache_quant_config=SimpleNamespace(enabled=False),
                cache_dit_config=None,
                tp_size=1,
                ring_degree=1,
                dp_size=1,
                enable_cfg_parallel=False,
                enable_torch_compile=False,
                enable_breakable_cuda_graph=False,
                use_fsdp_inference=False,
                has_layerwise_offload_components=lambda: False,
                lora_path=None,
                comfyui_mode=False,
                quantization=None,
                component_quantizations={},
                minimax_h3_adaln_online=False,
                minimax_h3_adaln_cache_path=None,
                batching_max_size=1,
                warmup_mode="off",
            )
            values.update(overrides)
            return SimpleNamespace(**values)

        with (
            patch(
                "sglang.multimodal_gen.configs.pipeline_configs.physis_h3.current_platform.is_cuda",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.selector.get_global_forced_attn_backend",
                return_value=None,
            ),
            patch(
                "sglang.multimodal_gen.configs.pipeline_configs.physis_h3.envs.SGLANG_CACHE_DIT_ENABLED",
                False,
            ),
        ):
            config = PhysisH3PipelineConfig()
            valid = args()
            config.validate_server_args(valid)
            self.assertEqual(valid.attention_backend, "torch_sdpa")
            for overrides, message in (
                ({"attention_backend": "fa"}, "torch_sdpa"),
                ({"cache_dit_config": {}}, "cache-dit"),
                (
                    {"kv_cache_quant_config": SimpleNamespace(enabled=True)},
                    "KV quantization",
                ),
            ):
                with self.subTest(overrides=overrides):
                    with self.assertRaisesRegex(ValueError, message):
                        config.validate_server_args(args(**overrides))

    def test_reference_noise_seed_and_initial_dtype(self):
        """A matching seed must reproduce reference noise, including bf16 draws."""
        runner = PhysisH3Denoiser.__new__(PhysisH3Denoiser)
        runner.seed = 42
        runner.device = torch.device("cpu")
        runner.shape = (1, 4, 1, 2, 2)
        # Reference chunk 2 initial noise uses seed*1000003 + chunk*4096.
        initial = torch.randn(
            runner.shape,
            generator=torch.Generator().manual_seed(42_008_318),
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(runner.initial_latent(2), initial, rtol=0, atol=0)
        # Slot 2 is the noise after denoising pass 1 for chunk 3.
        renoise = torch.randn(
            runner.shape,
            generator=torch.Generator().manual_seed(42_012_416),
            dtype=torch.float32,
        )
        torch.testing.assert_close(runner.noise(3, 2), renoise, rtol=0, atol=0)
        runner.shape = (1, 4, 3, 2, 2)
        grouped = torch.cat(
            [
                torch.randn(
                    (1, 4, 1, 2, 2),
                    generator=torch.Generator().manual_seed(seed),
                    dtype=torch.bfloat16,
                )
                for seed in (42_008_318, 42_012_414, 42_016_510)
            ],
            dim=2,
        )
        torch.testing.assert_close(runner.initial_latent(2), grouped, rtol=0, atol=0)

    def test_joint_bootstrap_caches_every_frame_before_rope(self):
        """A five-latent prefix must retain all five pages for later chunks."""
        rope = MiniMaxH3Rope(1)
        rope.inv_freq.fill_(1)
        context = PhysisH3AttentionContext(
            SimpleNamespace(rope=rope), 2, 2, 2, history=18, chunk_frames=5
        )
        context.sink_only = True
        context.set_chunk(0)
        attention = SimpleNamespace(
            q_norm=torch.nn.RMSNorm(8).to(torch.bfloat16),
            k_norm=torch.nn.RMSNorm(8).to(torch.bfloat16),
            head_dim=8,
        )
        q, k, v = torch.randn(3, 7, 2, 8).to(torch.bfloat16)
        output = context.attend(attention, q, k, v)
        self.assertEqual(output.shape, q.shape)
        cache = context.caches[(0, 0)]
        self.assertEqual(cache.local_end_index_int, 5)
        self.assertEqual(cache.global_end_index_int, 5)
        torch.testing.assert_close(cache.k[0], attention.k_norm(k)[2:])
        torch.testing.assert_close(cache.v[0], v[2:])

    def test_diffusers_checkpoint_reconstructs_only_missing_rope(self):
        model = SimpleNamespace(
            arch=SimpleNamespace(rope_inv_freq_len=4), hf_config={"rope_theta": 256.0}
        )
        state = dict(MiniMaxH3DiTModel._preprocess_diffusers_state_dict(model, []))
        torch.testing.assert_close(
            state["rope.inv_freq"], torch.tensor([1, 0.25, 0.0625, 0.015625])
        )
        existing = torch.rand(4)
        state = dict(
            MiniMaxH3DiTModel._preprocess_diffusers_state_dict(
                model, [("rope.inv_freq", existing)]
            )
        )
        self.assertIs(state["rope.inv_freq"], existing)

    def test_window_rebases_only_whole_vae_periods(self):
        self.assertEqual(physis_h3_frame_positions(0, 18), [0])
        self.assertEqual(physis_h3_frame_positions(19, 18), list(range(20)))
        self.assertEqual(physis_h3_frame_positions(20, 18), [0, *range(2, 21)])
        self.assertEqual(physis_h3_frame_positions(24, 18), list(range(20)))
        for chunk in range(1, 100):
            frames = physis_h3_frame_positions(chunk, 18)
            self.assertEqual(frames[0], 0)
            self.assertEqual((chunk - frames[-1]) % 5, 0)
            self.assertEqual(len(frames), 1 + min(chunk, 19))
            self.assertEqual(len(set(frames)), len(frames))

    def test_positions_follow_h3_video_clock(self):
        positions = physis_h3_positions(3, 4, 4, [0, 1, 5])
        torch.testing.assert_close(
            positions[:3, 0], torch.arange(3, dtype=torch.float64)
        )
        torch.testing.assert_close(
            positions[3::4, 0],
            torch.tensor([3, 3 + 5 / 3, 3 + 85 / 3], dtype=torch.float64),
        )

    def test_sampler_uses_dataward_velocity_and_rounds_clean_before_renoising(self):
        sample = torch.tensor([0.25, 0.375], dtype=torch.bfloat16)
        prediction = torch.tensor([0.1234, -0.4321])
        noise = torch.tensor([-0.25, 0.6789])
        clean = (sample.float() + 0.9 * prediction).to(torch.bfloat16)
        expected = (0.3 * clean.float() + 0.7 * noise).to(torch.bfloat16)
        torch.testing.assert_close(
            physis_h3_step(sample, prediction, 0.9, 0.7, noise),
            expected,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            physis_h3_step(sample, prediction, 0.9, 0, noise), clean, rtol=0, atol=0
        )

    def test_cache_is_pre_rope_and_keeps_independent_steps(self):
        torch.manual_seed(1)
        rope = MiniMaxH3Rope(1)
        rope.inv_freq.fill_(1)
        context = PhysisH3AttentionContext(
            SimpleNamespace(rope=rope), 2, 2, 2, history=2
        )
        attention = SimpleNamespace(
            q_norm=torch.nn.RMSNorm(8).to(torch.bfloat16),
            k_norm=torch.nn.RMSNorm(8).to(torch.bfloat16),
            head_dim=8,
        )
        sink_k = torch.randn(1, 1, 2, 8).to(torch.bfloat16)
        sink_v = torch.randn_like(sink_k)
        context.sinks[0] = (sink_k, sink_v)
        expected = {step: [] for step in range(4)}
        for chunk in range(1, 10):
            context.set_chunk(chunk)
            for step in range(4):
                context.step = step
                q, k, v = torch.randn(3, 3, 2, 8).to(torch.bfloat16)
                normalized = attention.k_norm(k)[-1:].detach()
                expected[step].append((normalized, v[-1:]))
                output = context.attend(attention, q, k, v)
                self.assertEqual(output.shape, q.shape)
                cache = context.caches[(step, 0)]
                end = cache.local_end_index_int
                torch.testing.assert_close(cache.k[:, :1], sink_k, rtol=0, atol=0)
                torch.testing.assert_close(cache.v[:, :1], sink_v, rtol=0, atol=0)
                recent = expected[step][-3:]
                torch.testing.assert_close(
                    cache.k[0, 1:end],
                    torch.cat([item[0] for item in recent]),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    cache.v[0, 1:end],
                    torch.cat([item[1] for item in recent]),
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
