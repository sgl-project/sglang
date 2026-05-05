"""Unit tests for π0.5 (Pi-0.5) VLA model.

These tests are deliberately lightweight — they exercise pure utility code
(config validation, processor helpers, normalization, image preprocessing,
and weight-loader key remaps) that can run without downloading a checkpoint.

The full end-to-end behavior/accuracy check against LeRobot lives in
``test_pi05_parity.py``.
"""

import logging
import unittest
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"builtin type Swig.* has no __module__ attribute",
)

import numpy as np
import torch


class TestPi05Config(unittest.TestCase):
    def test_default_config(self):
        from sglang.srt.configs.pi05 import Pi05Config

        config = Pi05Config()
        self.assertEqual(config.model_type, "pi05")
        self.assertEqual(config.architectures, ["Pi05ForActionPrediction"])
        self.assertEqual(config.chunk_size, 50)
        self.assertEqual(config.max_action_dim, 32)
        self.assertEqual(config.image_resolution, (224, 224))
        self.assertEqual(config.image_size, 224)
        self.assertEqual(config.image_token_index, config.vocab_size)

    def test_custom_config(self):
        from sglang.srt.configs.pi05 import Pi05Config

        config = Pi05Config(
            chunk_size=100,
            max_action_dim=16,
            num_inference_steps=5,
            vocab_size=1234,
            image_token_index=2222,
        )
        self.assertEqual(config.chunk_size, 100)
        self.assertEqual(config.max_action_dim, 16)
        self.assertEqual(config.num_inference_steps, 5)
        self.assertEqual(config.vocab_size, 1234)
        self.assertEqual(config.image_token_index, 2222)

    def test_non_square_resolution_rejected(self):
        from sglang.srt.configs.pi05 import Pi05Config

        with self.assertRaises(ValueError):
            Pi05Config(image_resolution=(224, 320))
        with self.assertRaises(ValueError):
            Pi05Config(image_resolution=(160, 224))

    def test_square_resolution_accepted(self):
        from sglang.srt.configs.pi05 import Pi05Config

        cfg = Pi05Config(image_resolution=(224, 224))
        self.assertEqual(cfg.image_resolution, (224, 224))
        self.assertEqual(cfg.image_size, 224)

        cfg = Pi05Config(image_resolution=[160, 160])
        self.assertEqual(cfg.image_resolution, (160, 160))
        self.assertEqual(cfg.image_size, 160)

    def test_malformed_resolution_rejected(self):
        from sglang.srt.configs.pi05 import Pi05Config

        with self.assertRaises(ValueError):
            Pi05Config(image_resolution=224)
        with self.assertRaises(ValueError):
            Pi05Config(image_resolution=(224, 224, 3))


class TestPi05ArchitectureRegistration(unittest.TestCase):
    def test_is_multimodal_and_is_vla(self):
        from sglang.srt.configs.model_config import is_multimodal_model, is_vla_model

        archs = ["Pi05ForActionPrediction"]
        self.assertTrue(is_vla_model(archs), "Pi05 must be routed through the VLA path")
        self.assertTrue(
            is_multimodal_model(archs),
            "Pi05 must also be registered as multimodal so get_processor() / "
            "get_mm_processor() fire on startup",
        )


class TestPi05ImageProcessor(unittest.TestCase):
    def test_resize_with_pad_square(self):
        from sglang.srt.multimodal.processors.pi05 import resize_with_pad

        out = resize_with_pad(torch.ones(1, 3, 224, 224), 224, 224)
        self.assertEqual(out.shape, (1, 3, 224, 224))

    def test_resize_with_pad_wide(self):
        from sglang.srt.multimodal.processors.pi05 import resize_with_pad

        out = resize_with_pad(torch.zeros(1, 3, 100, 200), 224, 224)
        self.assertEqual(out.shape, (1, 3, 224, 224))
        self.assertEqual(out[0, 0, 0, 112].item(), -1.0)

    def test_resize_with_pad_tall(self):
        from sglang.srt.multimodal.processors.pi05 import resize_with_pad

        out = resize_with_pad(torch.zeros(1, 3, 400, 100), 224, 224)
        self.assertEqual(out.shape, (1, 3, 224, 224))
        self.assertEqual(out[0, 0, 112, 0].item(), -1.0)

    def test_resize_with_pad_rejects_non_4d(self):
        from sglang.srt.multimodal.processors.pi05 import resize_with_pad

        with self.assertRaises(ValueError):
            resize_with_pad(torch.zeros(3, 100, 200), 224, 224)

    def test_pil_to_tensor(self):
        from PIL import Image
        from sglang.srt.multimodal.processors.pi05 import pil_image_to_tensor

        img = Image.fromarray(np.full((10, 10, 3), 128, dtype=np.uint8))
        t = pil_image_to_tensor(img)
        self.assertEqual(t.shape, (1, 3, 10, 10))
        expected = 128.0 / 255.0 * 2.0 - 1.0
        self.assertAlmostEqual(t[0, 0, 5, 5].item(), expected, places=4)

    def test_pil_to_tensor_rgba_to_rgb(self):
        from PIL import Image
        from sglang.srt.multimodal.processors.pi05 import pil_image_to_tensor

        rgba = np.full((8, 8, 4), 200, dtype=np.uint8)
        img = Image.fromarray(rgba, mode="RGBA")
        t = pil_image_to_tensor(img)
        self.assertEqual(t.shape, (1, 3, 8, 8))

    def test_make_empty_image(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05ImageProcessor

        proc = Pi05ImageProcessor(image_size=224)
        empty = proc.make_empty_image()
        self.assertEqual(empty.shape, (1, 3, 224, 224))
        self.assertTrue((empty == -1.0).all())

    def test_preprocess_single_resizes_if_needed(self):
        from PIL import Image
        from sglang.srt.multimodal.processors.pi05 import Pi05ImageProcessor

        proc = Pi05ImageProcessor(image_size=224)
        img = Image.fromarray(np.zeros((120, 120, 3), dtype=np.uint8))
        t = proc.preprocess_single(img)
        self.assertEqual(t.shape, (1, 3, 224, 224))


class TestPi05ProcessorHelpers(unittest.TestCase):
    def test_extract_state_from_dict(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Req:
            extra_body = {"state": [1.0, 2.0, 3.0]}

        state = Pi05Processor._extract_state(None, _Req())
        self.assertEqual(state, [1.0, 2.0, 3.0])

    def test_extract_state_missing(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        self.assertIsNone(Pi05Processor._extract_state(None, None))

        class _Req:
            extra_body = None

        self.assertIsNone(Pi05Processor._extract_state(None, _Req()))

    def test_extract_num_steps(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Req:
            extra_body = {"num_inference_steps": 7}

        self.assertEqual(Pi05Processor._extract_num_steps(None, _Req()), 7)
        self.assertIsNone(Pi05Processor._extract_num_steps(None, None))

    def test_tokenize_prompt_empty(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        ids, mask = Pi05Processor._tokenize_prompt(None, "   ")
        self.assertEqual(ids, [])
        self.assertEqual(mask, [])

    def test_tokenize_prompt_truncates(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Proc:
            max_token_len = 4

            @staticmethod
            def _tokenizer(text, truncation, max_length, padding, return_tensors):
                return {
                    "input_ids": list(range(max_length)),
                    "attention_mask": [1] * max_length,
                }

        ids, mask = Pi05Processor._tokenize_prompt(_Proc(), "hello world")
        self.assertEqual(ids, [0, 1, 2, 3])
        self.assertEqual(mask, [1, 1, 1, 1])

    def test_extract_state_norm_stats_prefers_request(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Proc:
            global_state_norm_stats = {"mode": "mean_std", "mean": [0.0], "std": [1.0]}

        class _Req:
            extra_body = {"state_norm_stats": {"mode": "min_max", "min": [-1.0], "max": [1.0]}}

        out = Pi05Processor._extract_state_norm_stats(_Proc(), _Req())
        self.assertEqual(out["mode"], "min_max")

    def test_extract_state_norm_stats_falls_back_to_global(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Proc:
            global_state_norm_stats = {"mode": "mean_std", "mean": [0.0], "std": [1.0]}

        out = Pi05Processor._extract_state_norm_stats(_Proc(), None)
        self.assertEqual(out["mode"], "mean_std")

    def test_pad_or_truncate_state(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Proc:
            max_state_dim = 4

        padded = Pi05Processor._pad_or_truncate_state(_Proc(), np.array([1.0, 2.0], dtype=np.float32))
        self.assertEqual(tuple(padded.shape), (4,))
        self.assertTrue(np.allclose(padded, np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32)))

        truncated = Pi05Processor._pad_or_truncate_state(
            _Proc(), np.array([1, 2, 3, 4, 5], dtype=np.float32)
        )
        self.assertTrue(np.allclose(truncated, np.array([1, 2, 3, 4], dtype=np.float32)))

    def test_pad_or_truncate_state_rejects_non_1d(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Proc:
            max_state_dim = 4

        with self.assertRaises(ValueError):
            Pi05Processor._pad_or_truncate_state(_Proc(), np.zeros((2, 2), dtype=np.float32))

    def test_normalize_state_mean_std(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Proc:
            max_state_dim = 2
            _pad_or_truncate_state = Pi05Processor._pad_or_truncate_state

        state = np.array([1.0, 0.0], dtype=np.float32)
        stats = {"mode": "mean_std", "mean": [0.5, -1.0], "std": [0.5, 0.5]}
        out = Pi05Processor._normalize_state(_Proc(), state, stats)
        self.assertTrue(np.allclose(out, np.array([1.0, 1.0], dtype=np.float32)))

    def test_normalize_state_min_max(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Proc:
            max_state_dim = 2
            _pad_or_truncate_state = Pi05Processor._pad_or_truncate_state

        state = np.array([0.25, 0.5], dtype=np.float32)
        stats = {"mode": "min_max", "min": [0.0, -1.0], "max": [1.0, 1.0]}
        out = Pi05Processor._normalize_state(_Proc(), state, stats)
        self.assertTrue(np.allclose(out, np.array([-0.5, 0.5], dtype=np.float32), atol=1e-5))

    def test_normalize_state_quantile_q01_q99(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Proc:
            max_state_dim = 2
            _pad_or_truncate_state = Pi05Processor._pad_or_truncate_state

        state = np.array([0.25, 0.75], dtype=np.float32)
        stats = {"q01": [0.0, 0.5], "q99": [1.0, 1.0]}
        out = Pi05Processor._normalize_state(_Proc(), state, stats)
        self.assertTrue(np.allclose(out, np.array([-0.5, 0.0], dtype=np.float32), atol=1e-5))

    def test_normalize_state_quantile_low_high(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Proc:
            max_state_dim = 2
            _pad_or_truncate_state = Pi05Processor._pad_or_truncate_state

        state = np.array([0.25, 0.75], dtype=np.float32)
        stats = {"low": [0.0, 0.5], "high": [1.0, 1.0]}
        out = Pi05Processor._normalize_state(_Proc(), state, stats)
        self.assertTrue(np.allclose(out, np.array([-0.5, 0.0], dtype=np.float32), atol=1e-5))

    def test_normalize_state_none_clips(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        state = np.array([-2.0, 0.2, 3.0], dtype=np.float32)
        out = Pi05Processor._normalize_state(None, state, None)
        self.assertTrue(np.allclose(out, np.array([-1.0, 0.2, 1.0], dtype=np.float32)))

    def test_discretize_state_range(self):
        from sglang.srt.multimodal.processors.pi05 import PI05_NUM_BINS, Pi05Processor

        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        out = Pi05Processor._discretize_state(None, x)
        self.assertTrue((out >= 0).all())
        self.assertTrue((out < PI05_NUM_BINS).all())

    def test_build_prompt_contains_task_and_state(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Proc:
            max_state_dim = 4
            _pad_or_truncate_state = Pi05Processor._pad_or_truncate_state
            _normalize_state = Pi05Processor._normalize_state
            _discretize_state = Pi05Processor._discretize_state

        prompt = Pi05Processor._build_pi05_prompt(
            _Proc(),
            input_text="pick_up_red_block",
            state=[0.1, 0.2],
            state_norm_stats=None,
        )
        self.assertIn("Task: pick up red block", prompt)
        self.assertIn("State:", prompt)
        self.assertTrue(prompt.endswith("Action: "))

    def test_build_prompt_missing_state_uses_zero_state(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Proc:
            max_state_dim = 4
            _pad_or_truncate_state = Pi05Processor._pad_or_truncate_state
            _normalize_state = Pi05Processor._normalize_state
            _discretize_state = Pi05Processor._discretize_state

        prompt = Pi05Processor._build_pi05_prompt(
            _Proc(),
            input_text="pick up red block",
            state=None,
            state_norm_stats=None,
        )
        self.assertIn("Task: pick up red block", prompt)
        self.assertIn("State:", prompt)

    def test_process_precomputed_requires_lang_metadata(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Proc:
            image_token_id = 99
            num_image_tokens = 3

        with self.assertRaises(ValueError):
            Pi05Processor._process_precomputed(
                _Proc(),
                [{"format": "processor_output", "feature": torch.zeros(2, 3, 4, 4)}],
            )

    def test_process_precomputed_builds_input_ids_and_offsets(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Proc:
            image_token_id = 99
            num_image_tokens = 3

        out = Pi05Processor._process_precomputed(
            _Proc(),
            [
                {
                    "format": "processor_output",
                    "feature": torch.zeros(2, 3, 4, 4),
                    "model_specific_data": {
                        "lang_tokens": [10, 11, 12],
                        "lang_attention_mask": [1, 1, 1],
                    },
                }
            ],
        )
        self.assertEqual(out.input_ids, [99, 99, 99, 99, 99, 99, 10, 11, 12])
        self.assertEqual(out.im_token_id, 99)
        self.assertEqual(len(out.mm_items), 1)
        self.assertEqual(out.mm_items[0].offsets, [(0, 5)])

    def test_process_precomputed_preserves_explicit_input_ids(self):
        from sglang.srt.multimodal.processors.pi05 import Pi05Processor

        class _Proc:
            image_token_id = 99
            num_image_tokens = 3

        out = Pi05Processor._process_precomputed(
            _Proc(),
            [{
                "format": "processor_output",
                "feature": torch.zeros(2, 3, 4, 4),
                "input_ids": [7, 8, 9],
                "model_specific_data": {"lang_tokens": [10], "lang_attention_mask": [1]},
            }],
        )
        self.assertEqual(out.input_ids, [7, 8, 9])


class TestPi05ModelContracts(unittest.TestCase):
    def test_extract_request_inputs_missing_required_metadata_raises(self):
        from types import SimpleNamespace

        from sglang.srt.models.pi05 import Pi05ForActionPrediction

        mm_input = SimpleNamespace(
            mm_items=[
                SimpleNamespace(
                    feature=torch.zeros(3, 3, 224, 224),
                    model_specific_data={"lang_tokens": [1, 2, 3]},
                )
            ]
        )
        model = object.__new__(Pi05ForActionPrediction)
        with self.assertRaises(ValueError):
            Pi05ForActionPrediction._extract_request_inputs(model, mm_input, torch.device("cpu"))

    def test_extract_request_inputs_promotes_1d_tokens_and_masks(self):
        from types import SimpleNamespace

        from sglang.srt.models.pi05 import Pi05ForActionPrediction

        mm_input = SimpleNamespace(
            mm_items=[
                SimpleNamespace(
                    feature=torch.zeros(3, 3, 224, 224),
                    model_specific_data={
                        "lang_tokens": [1, 2, 3],
                        "lang_attention_mask": [1, 1, 1],
                        "image_masks": [1, 0, 1],
                        "num_inference_steps": 7,
                    },
                )
            ]
        )
        model = object.__new__(Pi05ForActionPrediction)
        pixel_values, image_masks, lang_tokens, lang_masks, num_steps = (
            Pi05ForActionPrediction._extract_request_inputs(model, mm_input, torch.device("cpu"))
        )
        self.assertEqual(tuple(pixel_values.shape), (3, 3, 224, 224))
        self.assertEqual(tuple(image_masks.shape), (3,))
        self.assertEqual(tuple(lang_tokens.shape), (1, 3))
        self.assertEqual(tuple(lang_masks.shape), (1, 3))
        self.assertEqual(num_steps, 7)

    def test_forward_extend_with_empty_mm_input_returns_zero_actions(self):
        from types import SimpleNamespace

        from sglang.srt.models.pi05 import Pi05ForActionPrediction

        model = object.__new__(Pi05ForActionPrediction)
        model.action_horizon = 2
        model.action_dim = 3

        forward_batch = SimpleNamespace(
            batch_size=1,
            mm_inputs=[None],
            forward_mode=SimpleNamespace(is_extend=lambda: True, is_decode=lambda: False),
        )
        out = Pi05ForActionPrediction.forward(
            model,
            input_ids=torch.zeros(1, dtype=torch.long),
            positions=torch.zeros(1, dtype=torch.long),
            forward_batch=forward_batch,
        )
        self.assertEqual(tuple(out.next_token_logits.shape), (1, 6))
        self.assertTrue(torch.equal(out.next_token_logits, torch.zeros_like(out.next_token_logits)))


class TestPi05LoadWeightsRemap(unittest.TestCase):
    """Cover the key rewrite rules in ``Pi05ForActionPrediction.load_weights``."""

    @classmethod
    def setUpClass(cls):
        cls._pi05_logger = logging.getLogger("sglang.srt.models.pi05")
        cls._saved_level = cls._pi05_logger.level
        cls._pi05_logger.setLevel(logging.ERROR)

    @classmethod
    def tearDownClass(cls):
        cls._pi05_logger.setLevel(cls._saved_level)

    def _model(self):
        from sglang.srt.configs.pi05 import Pi05Config
        from sglang.srt.models.pi05 import Pi05ForActionPrediction

        cfg = Pi05Config(max_action_dim=8, max_state_dim=8, chunk_size=2)
        return Pi05ForActionPrediction(cfg)

    def test_remap_paligemma_submodules_to_nested_layout(self):
        model = self._model()
        params = dict(model.named_parameters())

        target = next(
            k
            for k in params
            if k.startswith("paligemma_with_expert.paligemma.model.vision_tower.")
        )
        flat_key = "model." + target.replace(
            "paligemma_with_expert.paligemma.model.vision_tower.",
            "paligemma_with_expert.paligemma.vision_tower.",
        )
        payload = torch.full(params[target].shape, 0.12345)
        before = params[target].detach().clone()

        model.load_weights([(flat_key, payload)])

        after = dict(model.named_parameters())[target]
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(torch.allclose(after, payload))

    def test_lm_head_remap_to_embed_tokens(self):
        model = self._model()
        params = dict(model.named_parameters())

        target = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        self.assertIn(target, params)

        payload = torch.full(params[target].shape, 0.5)
        before = params[target].detach().clone()

        model.load_weights(
            [("model.paligemma_with_expert.paligemma.lm_head.weight", payload)]
        )

        after = dict(model.named_parameters())[target]
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(torch.allclose(after, payload))


class TestVlaOutputPromotion(unittest.TestCase):
    class _FakeRecv:
        def __init__(self, vla_actions):
            self.vla_actions = vla_actions

    def test_vla_actions_promoted_to_out_dict(self):
        from sglang.srt.managers.tokenizer_manager import _promote_vla_actions

        actions = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        recv = self._FakeRecv(vla_actions=[actions])
        out = {"text": "", "output_ids": [], "meta_info": {}}
        _promote_vla_actions(out, recv, 0)
        self.assertIn("actions", out)
        self.assertEqual(out["actions"], actions)

    def test_none_vla_actions_leaves_out_dict_untouched(self):
        from sglang.srt.managers.tokenizer_manager import _promote_vla_actions

        recv = self._FakeRecv(vla_actions=None)
        out = {"text": "hello", "output_ids": [1, 2, 3], "meta_info": {}}
        _promote_vla_actions(out, recv, 0)
        self.assertNotIn("actions", out)
        self.assertEqual(out["text"], "hello")

    def test_none_slot_in_mixed_batch_leaves_out_dict_untouched(self):
        from sglang.srt.managers.tokenizer_manager import _promote_vla_actions

        recv = self._FakeRecv(vla_actions=[[[0.0]], None])
        out_vla = {"meta_info": {}}
        out_plain = {"meta_info": {}}
        _promote_vla_actions(out_vla, recv, 0)
        _promote_vla_actions(out_plain, recv, 1)
        self.assertIn("actions", out_vla)
        self.assertNotIn("actions", out_plain)

    def test_index_out_of_bounds_leaves_out_dict_untouched(self):
        from sglang.srt.managers.tokenizer_manager import _promote_vla_actions

        recv = self._FakeRecv(vla_actions=[[[0.0]]])
        out = {"meta_info": {}}
        _promote_vla_actions(out, recv, 5)
        self.assertNotIn("actions", out)

    def test_helper_ignores_missing_vla_actions_attr(self):
        from sglang.srt.managers.tokenizer_manager import _promote_vla_actions

        class _NoAttr:
            pass

        out = {"embedding": [0.1, 0.2], "meta_info": {}}
        _promote_vla_actions(out, _NoAttr(), 0)
        self.assertNotIn("actions", out)


class TestFinishVlaJson(unittest.TestCase):
    def test_finish_vla_to_json_shape(self):
        from sglang.srt.managers.schedule_batch import FINISH_VLA

        out = FINISH_VLA().to_json()
        self.assertEqual(out, {"type": "stop", "matched": "vla_done"})

    def test_finish_vla_is_not_error(self):
        from sglang.srt.managers.schedule_batch import FINISH_VLA

        self.assertFalse(FINISH_VLA().is_error)


if __name__ == "__main__":
    unittest.main()