"""Regression tests for Bailing multimodal rotary positions and config bounds."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.configs.bailing_hybrid import BailingMoeV3VLConfig
from sglang.srt.layers.rotary_embedding.bailing_mrope import BailingMRotaryEmbedding
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    compute_spec_mrope_positions,
)
from sglang.srt.multimodal.processors.bailing_mm import BailingMMMultimodalProcessor
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    PrefillPlacement,
    TargetHiddenKvInjector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _position_config(max_position_embeddings=131072):
    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        text_config=SimpleNamespace(
            image_patch_token=11,
            video_patch_token=12,
            image_start_token=10,
            video_start_token=13,
            use_interleaved_frame_timestamp=False,
            max_position_embeddings=max_position_embeddings,
        ),
    )


class TestBailingMRotaryEmbedding(CustomTestCase):
    def test_text_and_single_multimodal_position_shapes(self):
        """A singleton sequence must retain the [3, batch, seq] contract."""
        config = _position_config()
        text_positions, text_delta = (
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                torch.tensor([7]), config, None, None
            )
        )
        image_positions, image_delta = (
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                torch.tensor([10, 11]),
                config,
                image_grid_thw=torch.tensor([[1, 2, 2]]),
                video_grid_thw=None,
            )
        )

        self.assertEqual(text_positions.shape, (3, 1, 1))
        self.assertEqual(text_delta.shape, (1, 1))
        self.assertEqual(image_positions.shape, (3, 1, 2))
        self.assertEqual(image_delta.shape, (1, 1))

    def test_centered_height_positions_can_be_negative(self):
        """Tall images require negative H coordinates instead of clamping to zero."""
        config = _position_config()
        input_ids = torch.tensor([10] + [11] * 7 + [99])

        positions, _ = (
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                input_ids,
                config,
                image_grid_thw=torch.tensor([[1, 14, 2]]),
                video_grid_thw=None,
            )
        )

        self.assertEqual(positions.shape, (3, 1, 9))
        self.assertLess(int(positions[1].min()), 0)

    def test_checkpoint_position_bound_is_enforced(self):
        """Media positions at or beyond the checkpoint context must fail clearly."""
        config = _position_config(max_position_embeddings=4)
        input_ids = torch.tensor([10] + [11] * 7)

        with self.assertRaisesRegex(ValueError, "checkpoint bounds"):
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                input_ids,
                config,
                image_grid_thw=torch.tensor([[1, 14, 2]]),
                video_grid_thw=None,
            )

    def test_negative_start_cache_growth_preserves_phase(self):
        """Growing a negative-origin cache must append the next logical phase."""
        with mock.patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True):
            rotary = BailingMRotaryEmbedding(
                head_size=8,
                rotary_dim=8,
                max_position_embeddings=16,
                base=10000,
                is_neox_style=True,
                dtype=torch.float32,
                mrope_section=[2, 1, 1],
                video_rope=True,
            )
        self.assertEqual(rotary.position_start, -16)
        self.assertEqual(rotary.cos_sin_cache.shape[0], 32)

        rotary._ensure_cos_sin_cache_length(32)
        inv_freq = rotary._compute_inv_freq(rotary.base)
        expected = torch.cat(((16 * inv_freq).cos(), (16 * inv_freq).sin()))
        torch.testing.assert_close(rotary.cos_sin_cache[32], expected)

    def test_yarn_scaling_extends_positive_cache_only(self):
        """YaRN factor stretches the positive side; the negative side is fixed."""
        with mock.patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True):
            rotary = BailingMRotaryEmbedding(
                head_size=8,
                rotary_dim=8,
                max_position_embeddings=16,
                base=10000,
                is_neox_style=True,
                dtype=torch.float32,
                mrope_section=[2, 1, 1],
                video_rope=True,
                scaling_factor=2.0,
                original_max_position_embeddings=16,
            )
        self.assertEqual(rotary.position_start, -16)
        self.assertEqual(rotary.cos_sin_cache.shape[0], 16 + 32)
        self.assertGreater(rotary.mscale, 1.0)

        inv_freq = rotary._compute_inv_freq(rotary.base)
        position = 20  # beyond the native bound of 16, row = 20 - (-16)
        expected = torch.cat(
            (
                (position * inv_freq).cos() * rotary.mscale,
                (position * inv_freq).sin() * rotary.mscale,
            )
        )
        torch.testing.assert_close(rotary.cos_sin_cache[36], expected)

        # Growth past the initial cache keeps the same scaled phase.
        rotary._ensure_cos_sin_cache_length(48)
        torch.testing.assert_close(rotary.cos_sin_cache[36], expected)

    def test_yarn_scaling_relaxes_positive_position_bound(self):
        """A yarn rope_parameters entry must raise the allowed positive bound."""
        config = _position_config(max_position_embeddings=4)
        config.text_config.rope_parameters = {
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 4,
        }
        input_ids = torch.tensor([10] + [11] * 7)

        # max position is 7: fails against the native bound of 4, passes once
        # the yarn factor doubles the positive bound to 8.
        positions, _ = (
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                input_ids,
                config,
                image_grid_thw=torch.tensor([[1, 14, 2]]),
                video_grid_thw=None,
            )
        )
        self.assertEqual(positions.shape, (3, 1, 8))

    def test_rope_scaling_override_merges_into_text_config(self):
        """A top-level rope_scaling override must not drop the mrope markers."""
        config = BailingMoeV3VLConfig(
            mrope_section=[8, 12, 12],
            text_config={"max_position_embeddings": 131072},
            vision_config={},
        )
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 131072,
        }

        rope_parameters = config.text_config.rope_parameters
        self.assertEqual(rope_parameters["rope_type"], "yarn")
        self.assertEqual(rope_parameters["factor"], 2.0)
        self.assertEqual(rope_parameters["mrope_section"], [8, 12, 12])
        self.assertTrue(rope_parameters["video_rope"])

    def test_public_checkpoint_config_contract(self):
        """External Ling-3.0-flash-VL config literals must survive local parsing."""
        config = BailingMoeV3VLConfig(
            image_token_id=157157,
            video_token_id=156909,
            mrope_section=[8, 12, 12],
            text_config={
                "num_hidden_layers": 42,
                "vocab_size": 157184,
                "max_position_embeddings": 131072,
                "moe_router_enable_expert_bias": True,
                "num_experts": 512,
                "num_experts_per_tok": 8,
                "n_group": 8,
                "topk_group": 4,
                "score_function": "sigmoid",
                "routed_scaling_factor": 2.5,
                "short_conv_kernel_size": 4,
            },
            vision_config={"disable_merger_proj": True},
        )

        self.assertEqual(config.text_config.num_hidden_layers, 42)
        self.assertEqual(config.text_config.max_position_embeddings, 131072)
        self.assertTrue(config.text_config.moe_router_enable_expert_bias)
        self.assertEqual(
            config.text_config.rope_parameters["mrope_section"], [8, 12, 12]
        )
        self.assertTrue(config.text_config.rope_parameters["video_rope"])
        self.assertTrue(config.vision_config.disable_merger_proj)


class TestBailingVLConfigNormalization(CustomTestCase):
    """Nested and top-level checkpoint forms must resolve to the same roles."""

    _NESTED_TOKENS = {
        "image_patch_token": 11,
        "video_patch_token": 12,
        "image_start_token": 10,
        "video_start_token": 13,
    }
    _INPUT_IDS = torch.tensor([10, 11, 11, 11, 11, 7, 13, 12, 12, 12, 12, 8])
    _GRIDS = {
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "video_grid_thw": torch.tensor([[1, 4, 4]]),
    }

    def _positions(self, config):
        positions, _ = (
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                self._INPUT_IDS, config, **self._GRIDS
            )
        )
        return positions

    def _assert_processor_and_mrope_agree(self, config):
        resolve = BailingMMMultimodalProcessor._resolve_token_id
        self.assertEqual(
            resolve(config, None, "image_token_id", "image_patch_token", ""), 11
        )
        self.assertEqual(
            resolve(config, None, "video_token_id", "video_patch_token", ""), 12
        )
        expected = self._positions(_position_config())
        self.assertFalse(
            torch.equal(expected[0, 0], torch.arange(self._INPUT_IDS.numel()))
        )
        torch.testing.assert_close(self._positions(config), expected)

    def test_legacy_nested_tokens_and_mrope_section(self):
        config = BailingMoeV3VLConfig(
            text_config={
                "max_position_embeddings": 131072,
                "mrope_section": [16, 8, 8],
                **self._NESTED_TOKENS,
            },
            vision_config={},
        )

        self.assertEqual(config.mrope_section, [16, 8, 8])
        self.assertEqual(
            config.text_config.rope_parameters["mrope_section"], [16, 8, 8]
        )
        self._assert_processor_and_mrope_agree(config)

    def test_top_level_tokens_without_nested_aliases(self):
        config = BailingMoeV3VLConfig(
            image_token_id=11,
            video_token_id=12,
            vision_start_token_id=10,
            video_start_token_id=13,
            text_config={"max_position_embeddings": 131072},
            vision_config={},
        )

        self.assertEqual(
            config.text_config.rope_parameters["mrope_section"], [8, 12, 12]
        )
        self._assert_processor_and_mrope_agree(config)

    def test_explicit_top_level_values_override_nested_values(self):
        config = BailingMoeV3VLConfig(
            image_token_id=11,
            mrope_section=[8, 12, 12],
            text_config={
                "max_position_embeddings": 131072,
                "mrope_section": [16, 8, 8],
                **self._NESTED_TOKENS,
                "image_patch_token": 99,
            },
            vision_config={},
        )

        self.assertEqual(
            config.text_config.rope_parameters["mrope_section"], [8, 12, 12]
        )
        self._assert_processor_and_mrope_agree(config)

    def test_positional_arguments_keep_their_slots(self):
        config = BailingMoeV3VLConfig(
            {"max_position_embeddings": 131072},
            {},
            11,
            12,
            10,
            14,
            False,
            [16, 8, 8],
        )

        self.assertEqual(
            (
                config.image_token_id,
                config.video_token_id,
                config.vision_start_token_id,
                config.vision_end_token_id,
                config.video_start_token_id,
                config.video_end_token_id,
            ),
            (11, 12, 10, 14, 157160, 157161),
        )
        self.assertEqual(config.mrope_section, [16, 8, 8])


class TestDSparkMropePositions(CustomTestCase):
    def test_prefill_placement_matches_target_extend_positions(self):
        """Draft KV rotary positions must be the target extend positions."""
        image = SimpleNamespace(
            mrope_positions=torch.tensor(
                [[0, 1, 1, 1, 1, 3], [0, 1, 1, 2, 2, 3], [0, 1, 2, 1, 2, 3]]
            ),
            mrope_position_delta=torch.tensor([[-2]]),
            mrope_position_delta_repeated_cache=None,
        )
        cases = (
            ("text", None, 0, 3, None),
            ("image", image, 0, 6, None),
            ("chunked", image, 2, 3, None),
            ("retracted", image, 7, 1, None),
            ("session", image, 4, 5, object()),
        )
        batch = SimpleNamespace(
            multimodal_inputs=[case[1] for case in cases],
            prefix_lens=[case[2] for case in cases],
            extend_lens=[case[3] for case in cases],
            seq_lens_cpu=torch.tensor([case[2] + case[3] for case in cases]),
            reqs=[SimpleNamespace(session=case[4]) for case in cases],
            out_cache_loc=torch.arange(sum(case[3] for case in cases)),
            req_pool_indices=torch.arange(len(cases)),
        )
        cpu = torch.device("cpu")
        injector = TargetHiddenKvInjector(
            draft_model=SimpleNamespace(is_mrope_enabled=True),
            draft_model_runner=SimpleNamespace(
                token_to_kv_pool=SimpleNamespace(), runner_ids=None
            ),
            model_runner=SimpleNamespace(
                device=cpu,
                prefill_attention_backend_str="torch_native",
                runner_ids=SimpleNamespace(
                    own_plan=lambda batch: SimpleNamespace(
                        write_ids=lambda _ids, cols=None, kind=None: batch.out_cache_loc
                    )
                ),
            ),
            device=cpu,
            verify_num_draft_tokens=2,
            block_pos_offsets=torch.arange(2),
        )

        for rl_on_policy_target in (None, "fsdp"):
            with (
                self.subTest(rl_on_policy_target=rl_on_policy_target),
                mock.patch(
                    "sglang.srt.model_executor.forward_batch_info.get_exec",
                    return_value=SimpleNamespace(
                        deterministic=SimpleNamespace(
                            rl_on_policy_target=rl_on_policy_target
                        )
                    ),
                ),
            ):
                target = SimpleNamespace(seq_lens_cpu=batch.seq_lens_cpu)
                ForwardBatch._compute_mrope_positions_extend(
                    target, SimpleNamespace(device=cpu), batch
                )
                placement = injector.place_prefill(batch)

                torch.testing.assert_close(
                    placement.mrope_positions, target.mrope_positions
                )
                self.assertEqual(
                    placement.mrope_positions.shape[-1], placement.positions.numel()
                )

    def test_spec_positions_mix_text_and_media_without_mutating_kv_positions(self):
        seq_positions = torch.tensor([[4, 5], [7, 8]])
        mm_inputs = [
            None,
            SimpleNamespace(mrope_position_delta=torch.tensor([3])),
        ]

        positions = compute_spec_mrope_positions(
            seq_positions, mm_inputs, batch_size=2, device=torch.device("cpu")
        )

        expected = torch.tensor([4, 5, 10, 11])
        torch.testing.assert_close(positions, expected.expand(3, -1))
        torch.testing.assert_close(seq_positions, torch.tensor([[4, 5], [7, 8]]))

    def test_prefill_selection_slices_mrope_on_the_token_axis(self):
        placement = PrefillPlacement(
            cache_loc=torch.tensor([10, 11, 12, 13]),
            positions=torch.tensor([4, 5, 6, 7]),
            state_slot=None,
            final_pos=None,
            swa_loc=None,
            mrope_positions=torch.arange(12).reshape(3, 4),
        )

        selected = placement.select(torch.tensor([3, 1]))

        torch.testing.assert_close(selected.cache_loc, torch.tensor([13, 11]))
        torch.testing.assert_close(selected.positions, torch.tensor([7, 5]))
        torch.testing.assert_close(
            selected.mrope_positions,
            torch.arange(12).reshape(3, 4)[:, [3, 1]],
        )


if __name__ == "__main__":
    unittest.main()
