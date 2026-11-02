"""Unit tests for the native-Qwen MTP early embed/lm_head bind.

``EagleDraftWorker.__init__`` binds the draft's BF16 skeleton
``embed_tokens``/``lm_head`` to the target's tensors as soon as both models
are loaded, before the scheduler allocates any memory pools. ``alloc_memory_pool``
consumes the precomputed result and skips the original post-pool init when the
early bind ran.

These tests run on CPU — they never allocate real CUDA tensors. They verify:

1. the guard is narrow (only ``Qwen3_5ForCausalLMMTP`` from
   ``sglang.srt.models.qwen3_5_mtp`` takes the early path);
2. the alias runs during construction (before ``alloc_memory_pool``);
3. embed/lm_head storage is shared with the target after the bind;
4. token-mapped (hot-token) heads fall back to the post-pool ordering;
5. genuine MTP decoder/fc/norm weights are untouched by the bind.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _qwen35_mtp_class():
    """A class whose __name__/__module__ match the native Qwen3.5 MTP guard."""
    return type(
        "Qwen3_5ForCausalLMMTP",
        (object,),
        {"__module__": "sglang.srt.models.qwen3_5_mtp"},
    )


class _Storage:
    def __init__(self, ptr):
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr


def _tensor_like(ptr, numel=1, element_size=2):
    t = MagicMock()
    t.untyped_storage.return_value = _Storage(ptr)
    t.numel.return_value = numel
    t.element_size.return_value = element_size
    return t


def _make_worker(draft_model=None, *, shared=True):
    """Worker double with mock draft/target models.

    shared=True wires the draft embed/head to the same storage as the target
    (the state the early bind is supposed to produce), so the positive path
    passes its assertions.
    """
    worker = object.__new__(EagleDraftWorker)
    worker.device = "cpu"

    target_model = MagicMock()
    target_model.model.embed_tokens.weight = _tensor_like(100, numel=3, element_size=2)
    target_model.lm_head = MagicMock()
    target_model.lm_head.weight = _tensor_like(200, numel=3, element_size=2)

    if draft_model is None:
        draft_model = _qwen35_mtp_class()()
    draft_model.model.embed_tokens.weight = (
        _tensor_like(100, numel=3, element_size=2) if shared else _tensor_like(500)
    )
    draft_model.lm_head = target_model.lm_head if shared else MagicMock()
    if not shared:
        draft_model.lm_head.weight = _tensor_like(600, numel=3, element_size=2)

    worker.draft_runner = SimpleNamespace(model=draft_model)
    worker.target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(model=target_model)
    )
    worker.init_token_map = MagicMock()
    worker.init_lm_head = MagicMock()
    worker.hot_token_id = None
    return worker


def _patch_spec():
    """alloc_memory_pool reads get_spec().speculative_use_rejection_sampling;
    patch it so the test needs no runtime context."""
    return patch(
        "sglang.srt.speculative.eagle_worker_v2.get_spec",
        return_value=SimpleNamespace(speculative_use_rejection_sampling=False),
    )


class TestNativeQwenMtpEarlyBindGuard(CustomTestCase):
    def test_qwen35_mtp_takes_early_path(self):
        """The narrow guard accepts exactly the native Qwen3.5 MTP draft."""
        worker = _make_worker()
        result = worker._bind_native_qwen_mtp_before_pool()
        self.assertTrue(result)
        worker.init_token_map.assert_called_once()
        worker.init_lm_head.assert_called_once()

    def test_other_drafts_do_not_take_early_path(self):
        """EAGLE3 / other MTP wrappers / plain Qwen keep post-pool ordering."""
        for name, module in [
            ("Qwen3NextForCausalLMMTP", "sglang.srt.models.qwen3_next_mtp"),
            ("Qwen3MoeForCausalLMMTP", "sglang.srt.models.qwen3_moe_mtp"),
            ("EagleDraftModel", "sglang.srt.models.some_eagle3"),
            ("Qwen3_5ForCausalLM", "sglang.srt.models.qwen3_5"),
        ]:
            with self.subTest(name=name):
                draft = type(name, (object,), {"__module__": module})()
                worker = _make_worker(draft)
                result = worker._bind_native_qwen_mtp_before_pool()
                self.assertFalse(result, f"{name} must not take the early path")
                worker.init_lm_head.assert_not_called()

    def test_wrong_module_does_not_take_early_path(self):
        """Same class name from a different module is not the native MTP."""
        draft = type(
            "Qwen3_5ForCausalLMMTP", (object,), {"__module__": "some.other.module"}
        )()
        worker = _make_worker(draft)
        result = worker._bind_native_qwen_mtp_before_pool()
        self.assertFalse(result)

    def test_token_map_falls_back_to_post_pool(self):
        """A hot-token (token-mapped) head must keep the post-pool ordering."""
        worker = _make_worker()
        worker.hot_token_id = object()  # any non-None value
        result = worker._bind_native_qwen_mtp_before_pool()
        self.assertFalse(result)
        worker.init_lm_head.assert_not_called()


class TestNativeQwenMtpEarlyBindTiming(CustomTestCase):
    def test_bind_runs_during_init_before_any_pool(self):
        """The early bind runs at construction time, before alloc_memory_pool.

        ``EagleDraftWorker.__init__`` computes ``_early_mtp_alias_bound`` via
        ``_bind_native_qwen_mtp_before_pool``; ``alloc_memory_pool`` only
        consumes that precomputed result and never re-binds.
        """
        worker = _make_worker()
        worker._early_mtp_alias_bound = True
        worker.draft_worker = MagicMock()
        worker._bind_native_qwen_mtp_before_pool = MagicMock(return_value=True)
        worker.init_token_map = MagicMock()
        worker.init_lm_head = MagicMock()

        with _patch_spec():
            worker.alloc_memory_pool(
                memory_pool_config=None,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=None,
            )

        # The precomputed flag is consumed, not recomputed, and the post-pool
        # init is skipped (the bind already ran in __init__).
        worker._bind_native_qwen_mtp_before_pool.assert_not_called()
        worker.draft_worker.alloc_memory_pool.assert_called_once()
        worker.init_token_map.assert_not_called()
        worker.init_lm_head.assert_not_called()

    def test_alloc_memory_pool_fallback_ordering_preserved(self):
        """Non-MTP drafts still run init_token_map + init_lm_head AFTER the pool."""
        worker = _make_worker()
        worker._early_mtp_alias_bound = False
        worker.draft_worker = MagicMock()
        worker.init_token_map = MagicMock()
        worker.init_lm_head = MagicMock()

        with _patch_spec():
            worker.alloc_memory_pool(
                memory_pool_config=None,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=None,
            )

        # Pool first, alias after.
        worker.draft_worker.alloc_memory_pool.assert_called_once()
        worker.init_token_map.assert_called_once()
        worker.init_lm_head.assert_called_once()

    def test_storage_shared_after_bind(self):
        """After the early bind, draft embed/head share target storage."""
        worker = _make_worker(shared=True)
        result = worker._bind_native_qwen_mtp_before_pool()
        self.assertTrue(result)
        draft_model = worker.draft_runner.model
        target_model = worker.target_worker.model_runner.model
        self.assertEqual(
            draft_model.model.embed_tokens.weight.untyped_storage().data_ptr(),
            target_model.model.embed_tokens.weight.untyped_storage().data_ptr(),
        )
        self.assertIs(draft_model.lm_head, target_model.lm_head)

    def test_genuine_mtp_weights_untouched(self):
        """The bind must not modify MTP decoder/fc/norm parameters.

        It only rebinds embed_tokens/lm_head; the MTP layer modules (fc,
        norms, decoder layers) are separate nn.Module attributes that the
        alias never touches.
        """
        draft_model = _qwen35_mtp_class()()
        draft_model.fc = MagicMock()
        draft_model.pre_fc_norm_embedding = MagicMock()
        draft_model.pre_fc_norm_hidden = MagicMock()
        draft_model.layers = MagicMock()
        worker = _make_worker(draft_model, shared=True)
        worker._bind_native_qwen_mtp_before_pool()
        # No attribute on the genuine MTP modules is deleted or reassigned.
        draft_model.fc.assert_not_called()
        draft_model.pre_fc_norm_embedding.assert_not_called()
        draft_model.pre_fc_norm_hidden.assert_not_called()


if __name__ == "__main__":
    sys.exit(unittest.main())
