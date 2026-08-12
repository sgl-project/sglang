# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.multimodal_gen.runtime.models.vlas.pi05_policy as pi05_policy_module
from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
from sglang.multimodal_gen.runtime.models.vlas.pi05_policy import Pi05PolicyModel
from sglang.multimodal_gen.runtime.vla.cuda_graph import (
    VLADenoiseGraphRunner,
    VLADenoiseGraphSignature,
    VLAPrefixGraphRunner,
    _BoundedCaptureCache,
)
from sglang.multimodal_gen.runtime.vla.prompt_bucketing import (
    bucket_prompt_tokens,
    effective_token_length,
    select_prompt_token_bucket,
)
from sglang.test.test_utils import CustomTestCase


class _FakeGraph:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


def _fake_capture():
    return SimpleNamespace(graph=_FakeGraph())


class TestPi05PromptBucketing(CustomTestCase):
    def test_selects_smallest_bucket(self):
        buckets = (32, 64, 128, 200)
        self.assertEqual(select_prompt_token_bucket(1, buckets), 32)
        self.assertEqual(select_prompt_token_bucket(32, buckets), 32)
        self.assertEqual(select_prompt_token_bucket(33, buckets), 64)
        self.assertEqual(select_prompt_token_bucket(200, buckets), 200)
        self.assertIsNone(select_prompt_token_bucket(201, buckets))

    def test_effective_length_preserves_valid_tokens_after_mask_holes(self):
        masks = torch.tensor(
            [
                [True, False, False, True, False],
                [True, True, False, False, False],
            ]
        )
        self.assertEqual(effective_token_length(masks), 4)

    def test_bucket_preserves_logical_mask_and_existing_padding_values(self):
        tokens = torch.arange(10).unsqueeze(0)
        masks = torch.tensor([[True] * 5 + [False] * 5])

        padded_tokens, padded_masks, logical_length, bucket = bucket_prompt_tokens(
            tokens,
            masks,
            (8, 16),
        )

        self.assertEqual(logical_length, 5)
        self.assertEqual(bucket, 8)
        torch.testing.assert_close(padded_tokens, tokens[:, :8])
        torch.testing.assert_close(
            padded_masks,
            torch.tensor([[True] * 5 + [False] * 3]),
        )

    def test_bucket_can_extend_externally_supplied_short_tokens(self):
        tokens = torch.tensor([[11, 12, 13]])
        masks = torch.tensor([[True, True, True]])

        padded_tokens, padded_masks, logical_length, bucket = bucket_prompt_tokens(
            tokens,
            masks,
            (8,),
            pad_token_id=99,
        )

        self.assertEqual((logical_length, bucket), (3, 8))
        self.assertEqual(padded_tokens.tolist(), [[11, 12, 13, 99, 99, 99, 99, 99]])
        self.assertEqual(
            padded_masks.tolist(),
            [[True, True, True, False, False, False, False, False]],
        )

    def test_length_above_largest_bucket_uses_exact_shape(self):
        tokens = torch.arange(70).unsqueeze(0)
        masks = torch.tensor([[True] * 65 + [False] * 5])

        exact_tokens, exact_masks, logical_length, bucket = bucket_prompt_tokens(
            tokens,
            masks,
            (32, 64),
        )

        self.assertEqual(logical_length, 65)
        self.assertIsNone(bucket)
        self.assertEqual(exact_tokens.shape, (1, 65))
        self.assertTrue(exact_masks.all())

    def test_prefix_cache_separates_exact_and_bucketed_layouts(self):
        model = Pi05PolicyModel.__new__(Pi05PolicyModel)
        model.config = Pi05PipelineConfig(prompt_token_buckets=(32, 64))
        model.device = torch.device("cuda")
        model.dtype = torch.bfloat16
        model.model_path = "lerobot/pi05_base"
        model.runtime_role = "all"
        model.prefix_graph_runner = SimpleNamespace(enabled=True)
        model.graph_runner = SimpleNamespace(enabled=True)
        model._prefix_tensor_parallel_enabled = lambda: False
        observation = SimpleNamespace(
            metadata={"camera_order": ("front",)},
            images={"front": torch.zeros(1, 3, 2, 2)},
            image_masks={"front": torch.tensor(True)},
            tokens=torch.arange(40).unsqueeze(0),
            token_masks=torch.tensor([[True] * 20 + [False] * 20]),
        )
        with patch.object(pi05_policy_module, "get_vla_split_group", return_value=None):
            exact = model.build_prefix_cache_key(observation, bucket_prompt=False)
            bucketed = model.build_prefix_cache_key(observation, bucket_prompt=True)

        self.assertNotEqual(exact, bucketed)

        observation.tokens = torch.arange(70).unsqueeze(0)
        observation.token_masks = torch.ones(1, 70, dtype=torch.bool)
        with patch.object(pi05_policy_module, "get_vla_split_group", return_value=None):
            exact = model.build_prefix_cache_key(observation, bucket_prompt=False)
            bucket_miss = model.build_prefix_cache_key(observation, bucket_prompt=True)
        self.assertNotEqual(exact, bucket_miss)

    def test_prefix_cache_key_keeps_visible_tokens_after_mask_holes(self):
        model = Pi05PolicyModel.__new__(Pi05PolicyModel)
        model.config = Pi05PipelineConfig()
        model.dtype = torch.bfloat16
        model.model_path = "lerobot/pi05_base"
        common = dict(
            metadata={"camera_order": ("front",)},
            images={"front": torch.zeros(1, 3, 2, 2)},
            image_masks={"front": torch.tensor(True)},
            token_masks=torch.tensor([[True, False, False, True, False]]),
        )
        first = SimpleNamespace(tokens=torch.tensor([[1, 2, 3, 4, 0]]), **common)
        second = SimpleNamespace(tokens=torch.tensor([[1, 2, 3, 9, 0]]), **common)

        self.assertNotEqual(
            model.build_prefix_cache_key(first),
            model.build_prefix_cache_key(second),
        )

    def test_bucket_miss_disables_action_graph_capture(self):
        model = Pi05PolicyModel.__new__(Pi05PolicyModel)
        torch.nn.Module.__init__(model)
        model.action_expert = lambda context, x_t, timestep, **kwargs: x_t + 1
        model.graph_runner = SimpleNamespace(
            capture_or_run=lambda *args, **kwargs: self.fail(
                "bucket misses must stay eager"
            )
        )
        context = SimpleNamespace(
            layout={"cuda_graph_eligible": False},
            prefix_len=900,
        )
        x_t = torch.zeros(1, 50, 32)

        output = model.denoise_step(
            context,
            x_t,
            torch.ones(1),
            use_cuda_graph=True,
        )

        torch.testing.assert_close(output, torch.ones_like(x_t))


class TestPi05CudaGraphLRU(CustomTestCase):
    def test_lru_hit_updates_eviction_order_and_releases_graph(self):
        cache = _BoundedCaptureCache("test", max_entries=2)
        first = _fake_capture()
        second = _fake_capture()
        third = _fake_capture()
        cache.put("first", first)
        cache.put("second", second)

        self.assertIs(cache.get("first"), first)
        cache.put("third", third)

        self.assertEqual(tuple(cache.entries), ("first", "third"))
        self.assertEqual(second.graph.reset_calls, 1)
        info = cache.info()
        self.assertEqual(info.size, 2)
        self.assertEqual(info.hits, 1)
        self.assertEqual(info.captures, 3)
        self.assertEqual(info.evictions, 1)

        self.assertIsNone(cache.get("missing"))
        self.assertEqual(cache.info().misses, 1)

        cache.clear()
        self.assertEqual(first.graph.reset_calls, 1)
        self.assertEqual(third.graph.reset_calls, 1)

    def test_non_evicting_cache_rejects_new_signature_at_capacity(self):
        cache = _BoundedCaptureCache("test", max_entries=1, evict_on_miss=False)
        first = _fake_capture()
        rejected = _fake_capture()
        cache.put("first", first)

        self.assertFalse(cache.can_admit("second"))
        self.assertTrue(cache.can_admit("first"))
        self.assertFalse(cache.put("second", rejected))
        self.assertEqual(tuple(cache.entries), ("first",))
        self.assertEqual(first.graph.reset_calls, 0)
        self.assertEqual(rejected.graph.reset_calls, 1)

    def test_zero_capacity_releases_capture_and_disables_runners(self):
        cache = _BoundedCaptureCache("test", max_entries=0)
        capture = _fake_capture()
        cache.put("unused", capture)

        self.assertEqual(capture.graph.reset_calls, 1)
        self.assertEqual(cache.info().size, 0)
        self.assertFalse(VLAPrefixGraphRunner(enabled=True, max_entries=0).enabled)
        self.assertFalse(VLADenoiseGraphRunner(enabled=True, max_entries=0).enabled)

    def test_attention_layout_is_part_of_action_graph_signature(self):
        common = dict(
            batch_size=1,
            prefix_len=800,
            action_horizon=50,
            action_dim=32,
            dtype="float32",
            parallel_layout="pi05-v1",
        )
        full = VLADenoiseGraphSignature(**common, full_attention=True)
        masked = VLADenoiseGraphSignature(**common, full_attention=False)
        self.assertNotEqual(full, masked)


class TestPi05CudaGraphConfig(CustomTestCase):
    def test_defaults_preserve_exact_prompt_shapes_and_bound_action_graphs(self):
        config = Pi05PipelineConfig()
        self.assertEqual(config.prompt_token_buckets, ())
        self.assertEqual(config.prefix_cuda_graph_max_entries, 1)
        self.assertGreater(config.action_cuda_graph_max_entries, 0)

    def test_prefix_graph_is_disabled_by_partial_language_offload(self):
        model = Pi05PolicyModel.__new__(Pi05PolicyModel)
        model.config = Pi05PipelineConfig(
            offload_prefix_language_layer_count_after_prefix=1
        )
        model.device = torch.device("cuda")
        model.runtime_role = "all"
        model._prefix_tensor_parallel_enabled = lambda: False

        self.assertFalse(model._prefix_cuda_graph_enabled())

    def test_json_style_bucket_list_is_normalized(self):
        config = Pi05PipelineConfig()
        config.update_pipeline_config(
            {
                "prompt_token_buckets": [16, 48, 96],
                "action_cuda_graph_max_entries": 3,
            }
        )
        self.assertEqual(config.prompt_token_buckets, (16, 48, 96))
        self.assertEqual(config.action_cuda_graph_max_entries, 3)

    def test_invalid_graph_configs_are_rejected(self):
        cases = (
            ({"prompt_token_buckets": (32, 32)}, "strictly increasing"),
            ({"prompt_token_buckets": (64, 32)}, "strictly increasing"),
            ({"prompt_token_buckets": (0, 32)}, "positive"),
            ({"prompt_token_buckets": (32, 256)}, "max_token_len"),
            ({"prefix_cuda_graph_max_entries": -1}, "prefix_cuda_graph"),
            ({"action_cuda_graph_max_entries": -1}, "action_cuda_graph"),
        )
        for overrides, match in cases:
            with self.subTest(overrides=overrides):
                config = Pi05PipelineConfig()
                for name, value in overrides.items():
                    setattr(config, name, value)
                with self.assertRaisesRegex(ValueError, match):
                    config.check_pipeline_config()


if __name__ == "__main__":
    unittest.main()
