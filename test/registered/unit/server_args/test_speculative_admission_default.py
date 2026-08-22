"""Unit tests for the speculative admission default in srt/arg_groups/speculative_hook.

``handle_speculative_decoding`` fills ``max_running_requests`` from the captured decode
CUDA-graph ladder when the user did not pass ``--max-running-requests``. These tests pin the
four branches that decide the value, plus the invariant that a non-speculative server is left
alone so the KV-capacity estimator still runs.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.arg_groups.speculative_hook import (
    _SPEC_MIN_MAX_RUNNING_REQUESTS,
    _fill_default_max_running_requests,
    handle_speculative_decoding,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import _REGISTRY
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSpeculativeAdmissionDefault(CustomTestCase):
    def _args(self, *, ladder_max_bs=256, backend=Backend.FULL, **overrides):
        # model_path="dummy" early-returns __post_init__, so fields stay writable and the hook
        # can be driven directly without resolving a real model.
        args = ServerArgs(model_path="dummy")
        args.device = "cuda"
        args.speculative_algorithm = "NGRAM"
        args.speculative_num_draft_tokens = 4
        # model_path="dummy" skips resolution, so the two fields _handle_ngram reads before this
        # hook runs are still unset; give them their ordinary values.
        args.speculative_ngram_max_bfs_breadth = 1
        args.page_size = 1
        args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=backend, max_bs=ladder_max_bs)
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_derives_from_decode_graph_ladder(self):
        args = self._args(ladder_max_bs=256)
        handle_speculative_decoding(args)
        self.assertEqual(args.max_running_requests, 256)

    def test_scales_by_attention_dp_degree(self):
        # max_running_requests is a global figure that resolve_max_num_reqs divides by the
        # attention-DP degree, so the derived global must be scaled to land on the ladder.
        # Driven through the helper rather than the hook: the algorithms whose handlers accept
        # DP attention (the EAGLE/MTP family) load the model's HF config on the way there, which
        # a dummy model path cannot satisfy. The other tests cover the hook wiring.
        args = self._args(ladder_max_bs=128, enable_dp_attention=True, dp_size=4)
        _fill_default_max_running_requests(
            args, SpeculativeAlgorithm.from_string(args.speculative_algorithm)
        )
        self.assertEqual(args.max_running_requests, 128 * 4)

    def test_externally_registered_algorithm_keeps_the_capacity_estimator(self):
        # A plugin registered through SpeculativeAlgorithm.register never received the historical
        # 48 (CustomSpecAlgo.handle_server_args is a no-op), so it resolves through the estimator.
        # Filling it here would *lower* its limit, which is the one thing this default must not do.
        args = self._args(ladder_max_bs=512, speculative_algorithm="MY_PLUGIN")

        @SpeculativeAlgorithm.register("MY_PLUGIN", supports_overlap=True)
        def _factory(server_args):
            return MagicMock

        try:
            algo = SpeculativeAlgorithm.from_string("MY_PLUGIN")
            _fill_default_max_running_requests(args, algo)
        finally:
            _REGISTRY.pop("my_plugin", None)
        self.assertIsNone(args.max_running_requests)

    def test_dp_size_ignored_without_dp_attention(self):
        args = self._args(ladder_max_bs=128, dp_size=4)
        handle_speculative_decoding(args)
        self.assertEqual(args.max_running_requests, 128)

    def test_floors_at_the_historical_value(self):
        # A small ladder must never admit fewer requests than previous releases did.
        args = self._args(ladder_max_bs=8)
        handle_speculative_decoding(args)
        self.assertEqual(args.max_running_requests, _SPEC_MIN_MAX_RUNNING_REQUESTS)

    def test_disabled_decode_graph_falls_back_to_the_floor(self):
        # No ladder to derive from, and downstream code requires the field to be set under
        # speculation, so it must land on the historical value rather than stay None.
        args = self._args(backend=Backend.DISABLED)
        handle_speculative_decoding(args)
        self.assertEqual(args.max_running_requests, _SPEC_MIN_MAX_RUNNING_REQUESTS)

    def test_explicit_flag_is_never_clobbered(self):
        args = self._args(max_running_requests=99)
        handle_speculative_decoding(args)
        self.assertEqual(args.max_running_requests, 99)

    def test_earlier_model_specific_writer_is_never_clobbered(self):
        # deepseek_v4_hook sets 256 before this hook runs; that value must survive.
        args = self._args(ladder_max_bs=512, max_running_requests=256)
        handle_speculative_decoding(args)
        self.assertEqual(args.max_running_requests, 256)

    def test_non_speculative_server_is_left_for_the_capacity_estimator(self):
        # The hook runs unconditionally. Leaving None is what lets resolve_max_num_reqs size the
        # limit from KV capacity, exactly as before this default existed.
        args = self._args(speculative_algorithm=None)
        handle_speculative_decoding(args)
        self.assertIsNone(args.max_running_requests)


if __name__ == "__main__":
    unittest.main()
