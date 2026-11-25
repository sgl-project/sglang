"""FlashInfer replays ragged (compact) target-verify CUDA graphs.

Each token tier keeps its own wrappers, the plan covers the whole tier over the
captured slots, and the host-side DFLASH fast plan (uniform arange) stays off
ragged wrappers.
"""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    WrapperDispatch,
    fast_prefill_plan,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    build_capture_verify_lens,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GRID = [4, 8, 12, 16]


def _layout(verify_lens):
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=verify_lens, device=torch.device("cpu"), grid=GRID
    )


def _backend(dispatch_reason=None):
    backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
    backend.dispatch_reason = dispatch_reason
    backend.prefill_backend = "fa2"
    backend.use_sliding_window_kv_pool = False
    backend.decode_cuda_graph_metadata = {}
    backend.prefill_cuda_graph_metadata = {}
    backend.draft_extend_cuda_graph_metadata = {}
    backend._create_prefill_wrappers = lambda bs, use_custom_mask=False: [
        SimpleNamespace(begin_forward=lambda *args, **kwargs: None)
    ]
    backend.indices_updater_prefill = Mock()
    return backend


def _verify_batch(bs, layout, *, num_tokens):
    spec_info = DFlashVerifyInput(
        draft_token=None,
        positions=None,
        draft_token_num=4,
        ragged_verify_layout=layout,
    )
    seq_lens = torch.full((bs,), 10, dtype=torch.int64)
    return SimpleNamespace(
        batch_size=bs,
        req_pool_indices=torch.arange(bs),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        seq_lens_sum=int(seq_lens.sum()),
        encoder_lens=None,
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=spec_info,
        ragged_verify_layout=layout,
        positions=torch.zeros(num_tokens, dtype=torch.int64),
        iteration_ids=None,
        out_cache_loc=None,
    )


def _planned(backend):
    kwargs = backend.indices_updater_prefill.update.call_args.kwargs
    return kwargs["prefill_wrappers"], kwargs["spec_info"]


class TestFlashInferRaggedVerifyGraph(unittest.TestCase):
    def test_tiers_sharing_slots_keep_their_own_wrappers(self):
        backend = _backend()
        self.assertTrue(backend.supports_ragged_verify_graph)
        planned = {}
        for tier, lens in ((8, [2, 2, 2, 2]), (12, [3, 3, 3, 3]), (8, [1, 3, 2, 2])):
            batch = _verify_batch(4, _layout(lens), num_tokens=tier)
            backend.init_forward_metadata_out_graph(batch, in_capture=True)
            wrappers, _ = _planned(backend)
            self.assertIs(backend.forward_metadata.prefill_wrappers, wrappers)
            self.assertIs(planned.setdefault(tier, wrappers), wrappers)
        self.assertIsNot(planned[8], planned[12])
        self.assertEqual(
            set(backend.prefill_cuda_graph_metadata), {("ragged", 8), ("ragged", 12)}
        )

    def test_plan_covers_the_tier_over_the_captured_slots(self):
        cases = (
            # (live verify lens, captured slots): fewer live rows than slots,
            # and every slot live with fewer tokens than the tier.
            ([1, 3, 2], 4),
            ([1, 3, 2], 3),
        )
        for lens, slots in cases:
            with self.subTest(lens=lens, slots=slots):
                backend = _backend()
                capture = _layout(
                    build_capture_verify_lens(
                        num_tokens=8, num_slots=slots, num_draft_tokens=4
                    )
                )
                backend.init_forward_metadata_out_graph(
                    _verify_batch(slots, capture, num_tokens=8), in_capture=True
                )
                live = _layout(lens)
                self.assertEqual(live.graph_num_tokens, 8)
                batch = _verify_batch(slots, live, num_tokens=8)
                backend.init_forward_metadata_out_graph(batch)
                _, spec_info = _planned(backend)
                padded = spec_info.ragged_verify_layout
                self.assertEqual(padded.bs, slots)
                self.assertEqual(int(padded.qo_indptr_device[-1]), 8)
                self.assertEqual(padded.total_verify_tokens, 8)
                # Real rows keep their lengths.
                self.assertEqual(
                    padded.verify_lens[: len(lens) - 1].tolist(), lens[:-1]
                )
                # The worker's spec input is left as it was.
                self.assertIs(batch.spec_info.ragged_verify_layout, live)

    def test_ragged_wrappers_keep_the_device_plan(self):
        backend = _backend()
        batch = _verify_batch(4, _layout([2, 2, 2, 2]), num_tokens=8)
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        (wrapper,) = backend.prefill_cuda_graph_metadata[("ragged", 8)]
        self.assertNotIsInstance(wrapper.begin_forward, partial)

        # A uniform DFLASH verify graph still gets the host fast plan.
        batch = _verify_batch(2, None, num_tokens=8)
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        (wrapper,) = backend.prefill_cuda_graph_metadata[2]
        self.assertIsInstance(wrapper.begin_forward, partial)
        self.assertIs(wrapper.begin_forward.func, fast_prefill_plan)

    def test_wrapper_pairs_refuse_ragged_verify(self):
        for reason in (WrapperDispatch.SLIDING_WINDOW, WrapperDispatch.CROSS_ATTENTION):
            with self.subTest(reason=reason):
                backend = _backend(reason)
                self.assertFalse(backend.supports_ragged_verify_graph)
                batch = _verify_batch(4, _layout([2, 2, 2, 2]), num_tokens=8)
                with self.assertRaises(NotImplementedError):
                    backend.init_forward_metadata_out_graph(batch, in_capture=True)


if __name__ == "__main__":
    unittest.main()
