"""Unit tests for π0 (Pi-Zero) VLA model.

These tests are deliberately lightweight — they exercise pure utility code
(mask building, sinusoidal embeddings, normalization, image preprocessing,
config plumbing) that can run on CPU without downloading any weights. The
full end-to-end parity check against LeRobot lives in
``test_pi0_parity.py`` and requires a model download.

Usage:
    python -m pytest test/srt/models/test_pi0.py -v
"""
import logging
import math
import unittest
import warnings

# Swallow noisy third-party DeprecationWarnings that surface at import time.
# They come from a transitively imported C extension (``SwigPyObject`` etc.)
# that's unrelated to this package.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"builtin type Swig.* has no __module__ attribute",
)

import torch
import torch.nn.functional as F



class TestPi0Utils(unittest.TestCase):
    def test_create_sinusoidal_pos_embedding_shape(self):
        from sglang.srt.models.pi0 import create_sinusoidal_pos_embedding
        emb = create_sinusoidal_pos_embedding(torch.tensor([0.0, 0.5, 1.0]), dimension=64)
        self.assertEqual(emb.shape, (3, 64))

    def test_create_sinusoidal_pos_embedding_values_at_zero(self):
        from sglang.srt.models.pi0 import create_sinusoidal_pos_embedding
        emb = create_sinusoidal_pos_embedding(torch.tensor([0.0]), dimension=4)
        self.assertTrue(torch.allclose(emb[0, :2].float(), torch.zeros(2), atol=1e-6))
        self.assertTrue(torch.allclose(emb[0, 2:].float(), torch.ones(2), atol=1e-6))

    def test_create_sinusoidal_pos_embedding_odd_dim_raises(self):
        from sglang.srt.models.pi0 import create_sinusoidal_pos_embedding
        with self.assertRaises(ValueError):
            create_sinusoidal_pos_embedding(torch.tensor([1.0]), dimension=3)

    def test_create_sinusoidal_pos_embedding_non_1d_raises(self):
        from sglang.srt.models.pi0 import create_sinusoidal_pos_embedding
        with self.assertRaises(ValueError):
            create_sinusoidal_pos_embedding(torch.tensor([[0.0, 0.5]]), dimension=4)

    def test_create_sinusoidal_pos_embedding_period_monotonic(self):
        """Different timesteps should give different embeddings."""
        from sglang.srt.models.pi0 import create_sinusoidal_pos_embedding
        emb = create_sinusoidal_pos_embedding(torch.tensor([0.1, 0.2, 0.5, 0.9]), dimension=16)
        # Embeddings for distinct timesteps must differ.
        for i in range(emb.shape[0]):
            for j in range(i + 1, emb.shape[0]):
                self.assertFalse(
                    torch.allclose(emb[i], emb[j]),
                    f"emb[{i}] and emb[{j}] should differ",
                )

    def test_make_att_2d_masks_bidirectional(self):
        from sglang.srt.models.pi0 import make_att_2d_masks
        result = make_att_2d_masks(torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 4, dtype=torch.long))
        self.assertTrue(result.all())

    def test_make_att_2d_masks_causal(self):
        from sglang.srt.models.pi0 import make_att_2d_masks
        result = make_att_2d_masks(torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 4, dtype=torch.long))
        expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        self.assertTrue(torch.equal(result[0], expected))

    def test_make_att_2d_masks_prefix_lm(self):
        from sglang.srt.models.pi0 import make_att_2d_masks
        result = make_att_2d_masks(torch.ones(1, 5, dtype=torch.bool), torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.long))
        self.assertTrue(result[0, 0, 0]); self.assertFalse(result[0, 0, 3])
        self.assertTrue(result[0, 3, 0]); self.assertFalse(result[0, 3, 4])

    def test_make_att_2d_masks_respects_padding(self):
        """Padded positions (pad_masks == 0) must not attend or be attended to."""
        from sglang.srt.models.pi0 import make_att_2d_masks
        pad = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)   # last token is padding
        att = torch.zeros(1, 4, dtype=torch.long)               # full bidirectional
        result = make_att_2d_masks(pad, att)
        self.assertFalse(result[0, :, 3].any(), "Nothing should attend to padded key")
        self.assertFalse(result[0, 3, :].any(), "Padded query should attend to nothing")

    def test_prepare_attention_masks_4d(self):
        from sglang.srt.models.pi0 import prepare_attention_masks_4d, OPENPI_ATTENTION_MASK_VALUE
        mask_2d = torch.tensor([[[True, False], [True, True]]])
        mask_4d = prepare_attention_masks_4d(mask_2d)
        # Shape: (B, 1, S, S)
        self.assertEqual(mask_4d.shape, (1, 1, 2, 2))
        # True → exactly 0.0
        self.assertEqual(mask_4d[0, 0, 0, 0].item(), 0.0)
        self.assertEqual(mask_4d[0, 0, 1, 1].item(), 0.0)
        # False → the large negative constant. Comparing via `.item()` promotes
        # to Python double, which can round differently from the float32 literal.
        # Compare inside a tensor of the same dtype so we check the actual stored
        # value, not a re-parsed representation.
        expected = torch.tensor(OPENPI_ATTENTION_MASK_VALUE, dtype=mask_4d.dtype)
        self.assertTrue(torch.equal(mask_4d[0, 0, 0, 1], expected))

    def test_prepare_attention_masks_4d_value_is_finite_and_negative(self):
        """Sanity: the mask fill must be very negative but still finite so
        softmax doesn't produce NaN when all keys are masked."""
        from sglang.srt.models.pi0 import OPENPI_ATTENTION_MASK_VALUE
        self.assertTrue(math.isfinite(OPENPI_ATTENTION_MASK_VALUE))
        self.assertLess(OPENPI_ATTENTION_MASK_VALUE, -1e30)


class TestPi0Normalization(unittest.TestCase):
    """Cover the state/action normalization pipeline."""

    def test_build_norm_buffers_missing(self):
        from sglang.srt.models.pi0 import _build_norm_buffers
        self.assertIsNone(_build_norm_buffers(None, "state"))
        self.assertIsNone(_build_norm_buffers({}, "state"))
        self.assertIsNone(_build_norm_buffers({"state": None}, "state"))

    def test_build_norm_buffers_mean_std(self):
        from sglang.srt.models.pi0 import _build_norm_buffers
        stats = {"state": {"mode": "mean_std", "mean": [1.0, 2.0], "std": [0.5, 0.5]}}
        out = _build_norm_buffers(stats, "state")
        self.assertEqual(out["mode"], "mean_std")
        self.assertTrue(torch.equal(out["mean"], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(out["std"], torch.tensor([0.5, 0.5])))

    def test_build_norm_buffers_min_max(self):
        from sglang.srt.models.pi0 import _build_norm_buffers
        stats = {"action": {"mode": "min_max", "min": [-1.0, 0.0], "max": [1.0, 2.0]}}
        out = _build_norm_buffers(stats, "action")
        self.assertEqual(out["mode"], "min_max")
        self.assertTrue(torch.equal(out["min"], torch.tensor([-1.0, 0.0])))
        self.assertTrue(torch.equal(out["max"], torch.tensor([1.0, 2.0])))

    def test_apply_norm_mean_std_forward_inverse(self):
        """Forward then inverse must recover the original input."""
        from sglang.srt.models.pi0 import _apply_norm, _build_norm_buffers
        stats = _build_norm_buffers(
            {"s": {"mode": "mean_std", "mean": [0.5, -0.2], "std": [2.0, 0.1]}}, "s"
        )
        x = torch.tensor([[1.0, 0.3]])
        z = _apply_norm(x, stats, inverse=False)
        # (1 - 0.5) / 2 = 0.25 ;  (0.3 - -0.2) / 0.1 = 5.0
        self.assertTrue(torch.allclose(z, torch.tensor([[0.25, 5.0]]), atol=1e-5))
        # Round-trip
        x_back = _apply_norm(z, stats, inverse=True)
        self.assertTrue(torch.allclose(x_back, x, atol=1e-5))

    def test_apply_norm_min_max_forward_inverse(self):
        from sglang.srt.models.pi0 import _apply_norm, _build_norm_buffers
        stats = _build_norm_buffers(
            {"s": {"mode": "min_max", "min": [0.0, -1.0], "max": [1.0, 1.0]}}, "s"
        )
        x = torch.tensor([[0.25, 0.5]])
        z = _apply_norm(x, stats, inverse=False)
        # (0.25 - 0) / 1 * 2 - 1 = -0.5 ;  (0.5 - -1) / 2 * 2 - 1 = 0.5
        self.assertTrue(torch.allclose(z, torch.tensor([[-0.5, 0.5]]), atol=1e-5))
        x_back = _apply_norm(z, stats, inverse=True)
        self.assertTrue(torch.allclose(x_back, x, atol=1e-5))

    def test_apply_norm_preserves_padded_tail(self):
        """π0 pads state/action to max_dim. Only the first len(stats) entries
        should be transformed; padded tail must pass through untouched."""
        from sglang.srt.models.pi0 import _apply_norm, _build_norm_buffers
        stats = _build_norm_buffers(
            {"s": {"mode": "mean_std", "mean": [0.0, 0.0], "std": [1.0, 1.0]}}, "s"
        )
        # Last two dims are padding (state has 2 real entries, padded to 4).
        x = torch.tensor([[1.0, 2.0, 99.0, -99.0]])
        z = _apply_norm(x, stats, inverse=False)
        self.assertTrue(torch.allclose(z[:, :2], x[:, :2]))  # mean=0,std=1 identity
        self.assertTrue(torch.allclose(z[:, 2:], x[:, 2:]))  # padded tail untouched

    def test_apply_norm_noop_when_stats_none(self):
        from sglang.srt.models.pi0 import _apply_norm
        x = torch.randn(2, 5)
        self.assertTrue(torch.equal(_apply_norm(x, None, inverse=False), x))
        self.assertTrue(torch.equal(_apply_norm(x, None, inverse=True), x))


class TestPi0Config(unittest.TestCase):
    def test_default_config(self):
        from sglang.srt.configs.pi0 import Pi0Config
        config = Pi0Config()
        self.assertEqual(config.model_type, "pi0")
        self.assertEqual(config.architectures, ["Pi0ForActionPrediction"])
        self.assertEqual(config.chunk_size, 50)
        self.assertEqual(config.max_action_dim, 32)

    def test_custom_config(self):
        from sglang.srt.configs.pi0 import Pi0Config
        config = Pi0Config(chunk_size=100, max_action_dim=16, num_inference_steps=5)
        self.assertEqual(config.chunk_size, 100)
        self.assertEqual(config.max_action_dim, 16)


class TestGemmaVariantConfig(unittest.TestCase):
    def test_gemma_2b(self):
        from sglang.srt.models.pi0 import get_gemma_config
        c = get_gemma_config("gemma_2b")
        self.assertEqual(c.width, 2048); self.assertEqual(c.depth, 18)

    def test_gemma_300m(self):
        from sglang.srt.models.pi0 import get_gemma_config
        c = get_gemma_config("gemma_300m")
        self.assertEqual(c.width, 1024); self.assertEqual(c.depth, 18)

    def test_unknown_raises(self):
        from sglang.srt.models.pi0 import get_gemma_config
        with self.assertRaises(ValueError):
            get_gemma_config("gemma_7b")


class TestIsVlaModel(unittest.TestCase):
    def test_pi0_is_vla(self):
        from sglang.srt.configs.model_config import is_vla_model
        self.assertTrue(is_vla_model(["Pi0ForActionPrediction"]))

    def test_llama_is_not_vla(self):
        from sglang.srt.configs.model_config import is_vla_model
        self.assertFalse(is_vla_model(["LlamaForCausalLM"]))

    def test_gemma3_is_not_vla(self):
        from sglang.srt.configs.model_config import is_vla_model
        self.assertFalse(is_vla_model(["Gemma3ForConditionalGeneration"]))


class TestPi0ImageProcessor(unittest.TestCase):
    def test_resize_with_pad_square(self):
        from sglang.srt.multimodal.processors.pi0 import resize_with_pad
        out = resize_with_pad(torch.ones(1, 3, 224, 224), 224, 224)
        self.assertEqual(out.shape, (1, 3, 224, 224))

    def test_resize_with_pad_wide(self):
        """Wide source (H < W): padding is applied to top/bottom (rows)."""
        from sglang.srt.multimodal.processors.pi0 import resize_with_pad
        out = resize_with_pad(torch.zeros(1, 3, 100, 200), 224, 224)
        self.assertEqual(out.shape, (1, 3, 224, 224))
        self.assertEqual(out[0, 0, 0, 112].item(), -1.0)  # top row is padded

    def test_resize_with_pad_tall(self):
        """Tall source (H > W): padding is applied to left/right (cols)."""
        from sglang.srt.multimodal.processors.pi0 import resize_with_pad
        out = resize_with_pad(torch.zeros(1, 3, 400, 100), 224, 224)
        self.assertEqual(out.shape, (1, 3, 224, 224))
        # Left-most column must be padded to -1.
        self.assertEqual(out[0, 0, 112, 0].item(), -1.0)

    def test_resize_with_pad_rejects_non_4d(self):
        from sglang.srt.multimodal.processors.pi0 import resize_with_pad
        with self.assertRaises(ValueError):
            resize_with_pad(torch.zeros(3, 100, 200), 224, 224)

    def test_pil_to_tensor(self):
        from sglang.srt.multimodal.processors.pi0 import pil_image_to_tensor
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.full((10, 10, 3), 128, dtype=np.uint8))
        t = pil_image_to_tensor(img)
        self.assertEqual(t.shape, (1, 3, 10, 10))
        expected = 128.0 / 255.0 * 2.0 - 1.0
        self.assertAlmostEqual(t[0, 0, 5, 5].item(), expected, places=4)

    def test_pil_to_tensor_rgba_to_rgb(self):
        """Non-RGB PIL images (RGBA, L, …) must be converted to RGB."""
        from sglang.srt.multimodal.processors.pi0 import pil_image_to_tensor
        from PIL import Image
        import numpy as np
        rgba = np.full((8, 8, 4), 200, dtype=np.uint8)
        img = Image.fromarray(rgba, mode="RGBA")
        t = pil_image_to_tensor(img)
        self.assertEqual(t.shape, (1, 3, 8, 8))

    def test_make_empty_image(self):
        from sglang.srt.multimodal.processors.pi0 import Pi0ImageProcessor
        proc = Pi0ImageProcessor(image_size=224)
        empty = proc.make_empty_image()
        self.assertEqual(empty.shape, (1, 3, 224, 224))
        self.assertTrue((empty == -1.0).all())

    def test_preprocess_single_resizes_if_needed(self):
        """preprocess_single should pipe through resize_with_pad when HxW differs."""
        from sglang.srt.multimodal.processors.pi0 import Pi0ImageProcessor
        from PIL import Image
        import numpy as np
        proc = Pi0ImageProcessor(image_size=224)
        img = Image.fromarray(np.zeros((120, 120, 3), dtype=np.uint8))
        t = proc.preprocess_single(img)
        self.assertEqual(t.shape, (1, 3, 224, 224))


class TestLoadWeightsRemap(unittest.TestCase):
    """Cover the two key-rewrite rules in ``Pi0ForActionPrediction.load_weights``.

    With modern transformers ``PaliGemmaForConditionalGeneration`` nests
    ``vision_tower`` / ``multi_modal_projector`` / ``language_model`` under
    ``self.paligemma.model.*``, and ties ``lm_head.weight`` with
    ``embed_tokens.weight`` at construction time. The LeRobot checkpoint stores
    them *flat* (and only stores the tied ``lm_head.weight`` copy, not
    ``embed_tokens.weight``), so we have to remap both things on the fly.
    """

    @classmethod
    def setUpClass(cls):
        # Each test instantiates a tiny π0 model and feeds a single synthetic
        # weight into ``load_weights``; the remaining 776 parameters are
        # intentionally unloaded, so the loader's "missing params" WARNING is
        # expected noise. Silence it inside this test class only.
        cls._pi0_logger = logging.getLogger("sglang.srt.models.pi0")
        cls._saved_level = cls._pi0_logger.level
        cls._pi0_logger.setLevel(logging.ERROR)

    @classmethod
    def tearDownClass(cls):
        cls._pi0_logger.setLevel(cls._saved_level)

    def _tiny_model(self):
        from sglang.srt.configs.pi0 import Pi0Config
        from sglang.srt.models.pi0 import Pi0ForActionPrediction
        cfg = Pi0Config(max_action_dim=8, max_state_dim=8, chunk_size=2)
        return Pi0ForActionPrediction(cfg)


    def test_remap_paligemma_submodules_to_nested_layout(self):
        """``paligemma.vision_tower.*`` (flat, checkpoint layout) must route
        to ``paligemma.model.vision_tower.*`` (nested, runtime layout)."""
        model = self._tiny_model()
        params = dict(model.named_parameters())

        target = next(
            k for k in params
            if k.startswith("paligemma_with_expert.paligemma.model.vision_tower.")
        )
        # Build the matching flat checkpoint key.
        flat_key = "model." + target.replace(
            "paligemma_with_expert.paligemma.model.vision_tower.",
            "paligemma_with_expert.paligemma.vision_tower.",
        )
        payload = torch.full(params[target].shape, 0.12345)
        before = params[target].detach().clone()

        model.load_weights([(flat_key, payload)])

        after = dict(model.named_parameters())[target]
        self.assertFalse(
            torch.equal(before, after),
            "nested vision_tower param was not updated — the flat→nested "
            "remap is broken",
        )
        self.assertTrue(torch.allclose(after, payload))

    def test_lm_head_remap_to_embed_tokens(self):
        """The LeRobot ``pi0_base`` checkpoint only stores
        ``paligemma.lm_head.weight`` (the *tied* copy). Our loader must
        redirect that key into ``embed_tokens.weight``; without the redirect
        the language embedding silently stays at random init, which
        ``test_pi0_parity.py`` catches end-to-end."""
        model = self._tiny_model()
        params = dict(model.named_parameters())

        target = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        self.assertIn(target, params)

        # std ~0.5 is far from nn.Embedding's default ~0.02, so we'll be able
        # to tell the write actually landed.
        payload = torch.full(params[target].shape, 0.5)
        before = params[target].detach().clone()

        # Exactly what lerobot/pi0_base stores (leading "model." prefix, no
        # ".model." infix — it's the flat checkpoint layout).
        model.load_weights([("model.paligemma_with_expert.paligemma.lm_head.weight",
                             payload)])

        after = dict(model.named_parameters())[target]
        self.assertFalse(
            torch.equal(before, after),
            "embed_tokens.weight was not updated — lm_head→embed_tokens "
            "remap is broken",
        )
        self.assertTrue(torch.allclose(after, payload))


class TestPi0ProcessorHelpers(unittest.TestCase):
    """Tests for the pure-Python helpers on the π0 multimodal processor that
    don't require a tokenizer download."""

    def test_extract_state_from_dict(self):
        from sglang.srt.multimodal.processors.pi0 import Pi0Processor

        class _Req:
            extra_body = {"state": [1.0, 2.0, 3.0]}

        # _extract_state is a bound method; call it without a real processor
        # by grabbing the unbound function.
        state = Pi0Processor._extract_state(None, _Req())
        self.assertEqual(state, [1.0, 2.0, 3.0])

    def test_extract_state_missing(self):
        from sglang.srt.multimodal.processors.pi0 import Pi0Processor
        self.assertIsNone(Pi0Processor._extract_state(None, None))

        class _Req:
            extra_body = None

        self.assertIsNone(Pi0Processor._extract_state(None, _Req()))

    def test_extract_num_steps(self):
        from sglang.srt.multimodal.processors.pi0 import Pi0Processor

        class _Req:
            extra_body = {"num_inference_steps": 7}

        self.assertEqual(Pi0Processor._extract_num_steps(None, _Req()), 7)
        self.assertIsNone(Pi0Processor._extract_num_steps(None, None))


class TestPi0ConfigValidation(unittest.TestCase):
    """Contract checks on ``Pi0Config``."""

    def test_non_square_resolution_rejected(self):
        """π0 / SigLIP only support square inputs — non-square config must
        fail fast at construction time, not silently drop the width."""
        from sglang.srt.configs.pi0 import Pi0Config
        with self.assertRaises(ValueError):
            Pi0Config(image_resolution=(224, 320))
        with self.assertRaises(ValueError):
            Pi0Config(image_resolution=(160, 224))

    def test_square_resolution_accepted(self):
        from sglang.srt.configs.pi0 import Pi0Config
        cfg = Pi0Config(image_resolution=(224, 224))
        self.assertEqual(cfg.image_resolution, (224, 224))
        cfg = Pi0Config(image_resolution=[160, 160])
        # list gets normalized to tuple
        self.assertEqual(cfg.image_resolution, (160, 160))

    def test_malformed_resolution_rejected(self):
        from sglang.srt.configs.pi0 import Pi0Config
        with self.assertRaises(ValueError):
            Pi0Config(image_resolution=224)
        with self.assertRaises(ValueError):
            Pi0Config(image_resolution=(224, 224, 3))


class TestPi0ArchitectureRegistration(unittest.TestCase):
    """Make sure π0 is routed through the multimodal + VLA plumbing."""

    def test_is_multimodal_and_is_vla(self):
        from sglang.srt.configs.model_config import (
            is_multimodal_model,
            is_vla_model,
        )
        archs = ["Pi0ForActionPrediction"]
        self.assertTrue(is_vla_model(archs), "Pi0 must be routed through the VLA path")
        self.assertTrue(
            is_multimodal_model(archs),
            "Pi0 must also be registered as multimodal so get_processor() / "
            "get_mm_processor() fire on startup",
        )


class TestVlaOutputPromotion(unittest.TestCase):
    """Cover the ``_promote_vla_actions`` helper that surfaces the VLA
    action chunk on the TokenizerManager's success path.

    We test the helper directly rather than driving the full
    ``_handle_batch_output`` through a stubbed TokenizerManager — the
    helper *is* the invariant, and isolating it keeps the test fast (no
    asyncio / zmq / model config needed).
    """

    class _FakeRecv:
        """Minimal stand-in for ``BatchStrOutput`` / ``BatchTokenIDOutput``
        that only exposes the attribute the helper reads."""

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
        """Text-generation responses have ``vla_actions=None`` — the helper
        must not insert an ``actions`` key."""
        from sglang.srt.managers.tokenizer_manager import _promote_vla_actions
        recv = self._FakeRecv(vla_actions=None)
        out = {"text": "hello", "output_ids": [1, 2, 3], "meta_info": {}}
        _promote_vla_actions(out, recv, 0)
        self.assertNotIn("actions", out)
        self.assertEqual(out["text"], "hello")

    def test_none_slot_in_mixed_batch_leaves_out_dict_untouched(self):
        """Helper contract: a ``None`` slot in ``vla_actions`` leaves the
        corresponding ``out_dict`` untouched. The back-fill logic in
        ``scheduler_output_processor_mixin`` pads ``None`` for non-VLA
        requests, so the helper must skip those slots cleanly."""
        from sglang.srt.managers.tokenizer_manager import _promote_vla_actions
        recv = self._FakeRecv(vla_actions=[[[0.0]], None])
        out_vla = {"meta_info": {}}
        out_plain = {"meta_info": {}}
        _promote_vla_actions(out_vla, recv, 0)
        _promote_vla_actions(out_plain, recv, 1)
        self.assertIn("actions", out_vla)
        self.assertNotIn("actions", out_plain)

    def test_index_out_of_bounds_leaves_out_dict_untouched(self):
        """Defensive: if for some reason the caller asks for an index past
        the list length, we should not raise or overwrite."""
        from sglang.srt.managers.tokenizer_manager import _promote_vla_actions
        recv = self._FakeRecv(vla_actions=[[[0.0]]])
        out = {"meta_info": {}}
        _promote_vla_actions(out, recv, 5)
        self.assertNotIn("actions", out)

    def test_helper_ignores_missing_vla_actions_attr(self):
        """Backwards compat: older ``BatchEmbeddingOutput`` dataclasses may
        not even define the attribute; ``getattr(..., None)`` must handle it."""
        from sglang.srt.managers.tokenizer_manager import _promote_vla_actions

        class _NoAttr:
            pass

        out = {"embedding": [0.1, 0.2], "meta_info": {}}
        _promote_vla_actions(out, _NoAttr(), 0)
        self.assertNotIn("actions", out)


class TestFinishVlaJson(unittest.TestCase):
    """Lock down the ``FINISH_VLA.to_json()`` contract.

    The client-facing documentation (``docs/supported_models/pi0_vla.md``)
    promises a stable JSON shape — specifically
    ``{"type": "stop", "matched": "vla_done"}`` — so downstream clients can
    distinguish VLA responses from regular text-generation ``stop`` reasons.
    This test guards against accidental renames/refactors.
    """

    def test_finish_vla_to_json_shape(self):
        from sglang.srt.managers.schedule_batch import FINISH_VLA
        out = FINISH_VLA().to_json()
        self.assertEqual(out, {"type": "stop", "matched": "vla_done"})

    def test_finish_vla_is_not_error(self):
        """FINISH_VLA represents successful completion, not an abort."""
        from sglang.srt.managers.schedule_batch import FINISH_VLA
        self.assertFalse(FINISH_VLA().is_error)


if __name__ == "__main__":
    unittest.main()
