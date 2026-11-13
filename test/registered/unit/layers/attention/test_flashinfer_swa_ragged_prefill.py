"""FlashInfer extend metadata: sliding-window dispatch must not take the ragged prefill path.

The ragged wrapper is shared by every layer and planned without a window, so a sliding-window
layer served through it attends the whole prompt on a cold (no-prefix) prefill. The per-dispatch
paged wrappers are planned with window_left and must be used instead.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention import flashinfer_backend as fib
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _RecordingUpdater:
    def __init__(self):
        self.use_ragged = None

    def update(self, *args, use_ragged, **kwargs):
        self.use_ragged = use_ragged


def _backend(dispatch_reason):
    backend = object.__new__(fib.FlashInferAttnBackend)
    backend.use_sliding_window_kv_pool = False
    backend.is_multimodal = False
    backend.enable_mis = False
    backend.enable_deterministic = False
    backend.use_paged = False
    backend.dispatch_reason = dispatch_reason
    backend.indices_updater_prefill = _RecordingUpdater()
    backend.prefill_wrappers_paged = [object(), object()]
    backend.prefill_split_tile_size = None
    backend.dq_page_table = None
    backend._prepare_dequant_workspace_metadata_for_extend = lambda *args: None
    return backend


def _cold_extend_batch():
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        out_cache_loc=None,
        req_pool_indices=None,
        seq_lens=None,
        seq_lens_cpu=None,
        seq_lens_sum=8,
        extend_prefix_lens=None,
        extend_prefix_lens_cpu=[0],
        encoder_lens=None,
        cross_attention_custom_mask=None,
        kv_loc_plan=None,
    )


class TestFlashInferSlidingWindowPrefill(CustomTestCase):
    def test_sliding_window_dispatch_uses_paged_wrappers(self):
        backend = _backend(fib.WrapperDispatch.SLIDING_WINDOW)
        backend.init_forward_metadata(_cold_extend_batch())
        self.assertFalse(backend.indices_updater_prefill.use_ragged)
        self.assertFalse(backend.forward_metadata.use_ragged)

    def test_other_dispatch_keeps_ragged(self):
        for reason in (None, fib.WrapperDispatch.CROSS_ATTENTION):
            backend = _backend(reason)
            backend.init_forward_metadata(_cold_extend_batch())
            self.assertTrue(backend.forward_metadata.use_ragged)


if __name__ == "__main__":
    unittest.main()
