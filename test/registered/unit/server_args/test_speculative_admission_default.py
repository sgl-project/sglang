"""Unit tests for the speculative admission default in srt/arg_groups/speculative_hook.

``handle_speculative_decoding`` fills ``max_running_requests`` from the captured decode
CUDA-graph ladder when the user did not pass ``--max-running-requests``. These tests pin the
four branches that decide the value, plus the invariant that a non-speculative server is left
alone so the KV-capacity estimator still runs.

Every assertion reads ``resolving_view(args)``, never ``args.<field>``. Resolution accumulates
in the declaration stash and leaves the record pristine, so a field read reports the *input*
whatever the hook decided -- it would report ``None`` for a working derivation and ``None`` for a
broken one alike. Reading the record is how a suite stays green through a broken guarantee.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.arg_groups.speculative_hook import (
    _SPEC_MIN_MAX_RUNNING_REQUESTS,
    _speculative_admission_default,
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
    def _resolved(self, args):
        """What the engine will read: the record overlaid with the declaration stash."""
        return resolving_view(args)

    def _decode(self, args):
        cfg = self._resolved(args).cuda_graph_config
        return cfg.decode if cfg is not None else None

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
        self.assertEqual(self._resolved(args).max_running_requests, 256)

    def test_attention_dp_degree_does_not_scale_the_derived_limit(self):
        # Regression guard, and the reason it exists is a measured OOM. This default once derived
        # ladder_max_bs * attn_dp_size on the argument that resolve_max_num_reqs divides the global
        # limit by the DP degree. That multiplies the *per-worker* limit by the degree, and
        # admission costs memory of its own even though the capture ceiling is held: on
        # test/registered/models_e2e/test_dsa_glm52_dp_mtp.py (GLM-5.2-FP8, tp8/dp8, EAGLE MTP,
        # mem-fraction 0.88) the scaled limit of 4096 exhausted device memory during startup, while
        # 512 -- the unscaled ladder value -- serves. Verified on a pristine tree with the limit
        # passed explicitly, so the boundary belongs to upstream behaviour, not to this hook.
        # Driven through the helper rather than the hook: the algorithms whose handlers accept
        # DP attention (the EAGLE/MTP family) load the model's HF config on the way there, which
        # a dummy model path cannot satisfy. The other tests cover the hook wiring.
        args = self._args(ladder_max_bs=128, enable_dp_attention=True, dp_size=4)
        _speculative_admission_default(args)
        self.assertEqual(self._resolved(args).max_running_requests, 128)

    def test_externally_registered_algorithm_keeps_the_capacity_estimator(self):
        # A plugin registered through SpeculativeAlgorithm.register never received the historical
        # 48 (CustomSpecAlgo.handle_server_args is a no-op), so it resolves through the estimator.
        # Filling it here would *lower* its limit, which is the one thing this default must not do.
        args = self._args(ladder_max_bs=512, speculative_algorithm="MY_PLUGIN")

        @SpeculativeAlgorithm.register("MY_PLUGIN", supports_overlap=True)
        def _factory(server_args):
            return MagicMock

        try:
            _speculative_admission_default(args)
        finally:
            _REGISTRY.pop("my_plugin", None)
        self.assertIsNone(self._resolved(args).max_running_requests)

    def test_dp_size_ignored_without_dp_attention(self):
        args = self._args(ladder_max_bs=128, dp_size=4)
        handle_speculative_decoding(args)
        self.assertEqual(self._resolved(args).max_running_requests, 128)

    def test_floors_at_the_historical_value(self):
        # A small ladder must never admit fewer requests than previous releases did.
        args = self._args(ladder_max_bs=8)
        handle_speculative_decoding(args)
        self.assertEqual(
            self._resolved(args).max_running_requests, _SPEC_MIN_MAX_RUNNING_REQUESTS
        )

    def test_disabled_decode_graph_falls_back_to_the_floor(self):
        # No ladder to derive from, and downstream code requires the field to be set under
        # speculation, so it must land on the historical value rather than stay None.
        args = self._args(backend=Backend.DISABLED)
        handle_speculative_decoding(args)
        self.assertEqual(
            self._resolved(args).max_running_requests, _SPEC_MIN_MAX_RUNNING_REQUESTS
        )

    def test_explicit_flag_is_never_clobbered(self):
        args = self._args(max_running_requests=99)
        handle_speculative_decoding(args)
        self.assertEqual(self._resolved(args).max_running_requests, 99)

    def test_earlier_model_specific_writer_is_never_clobbered(self):
        # deepseek_v4_hook resolves 256 before this hook runs, and it does so by DECLARING, not by
        # assigning -- the record's field stays None. This hook must therefore consult the
        # resolving view; reading the field instead sees None, derives over the model's answer and
        # silently breaks the "DeepSeek-V4 is untouched" guarantee.
        #
        # Regression guard with a measured cost: it is exactly what a mechanical rebase of this
        # change onto the declaration pipeline gets wrong. The five hardcoded assignments this
        # replaces DO conflict textually and get fixed; the derivation is new code with no
        # upstream counterpart, so it merges clean, keeps its field reads, imports fine and passes
        # every assertion written against the record.
        args = self._args(ladder_max_bs=512)
        declare_resolution(args, "test_model_hook", max_running_requests=256)
        self.assertIsNone(
            args.max_running_requests,
            "a declaration must not write the record -- otherwise this test proves nothing",
        )
        handle_speculative_decoding(args)
        self.assertEqual(self._resolved(args).max_running_requests, 256)

    def test_capture_ladder_is_not_lengthened_by_the_raised_limit(self):
        # get_batch_sizes_to_capture clips capture_bs to the admission limit, so raising the
        # limit on its own lengthens the ladder. A speculative graph row is num_draft_tokens
        # wide, and the extra rungs can exhaust device memory during capture on configurations
        # that serve today. The ceiling has to stay put while the limit moves.
        args = self._args(ladder_max_bs=256)
        args.cuda_graph_config.decode.bs = [1, 2, 4, 8, 16, 32, 48, 64, 128, 256]
        handle_speculative_decoding(args)
        self.assertEqual(self._resolved(args).max_running_requests, 256)
        self.assertEqual(self._decode(args).max_bs, 48)
        self.assertEqual(self._decode(args).bs, [1, 2, 4, 8, 16, 32, 48])

    def test_captured_set_matches_the_clip_the_capture_path_applies(self):
        # The memory argument is that the captured set is the one this configuration captures
        # today -- which holds only if the ladder is narrowed by the same predicate
        # get_batch_sizes_to_capture uses (bs <= limit), not regenerated from the new max_bs.
        # Regeneration can land on different rungs, and then graph memory is no longer equal.
        ladder = [1, 2, 3, 5, 7, 11, 13, 17, 40, 44, 48, 52, 96, 200]
        args = self._args(ladder_max_bs=200)
        args.cuda_graph_config.decode.bs = list(ladder)
        handle_speculative_decoding(args)
        self.assertEqual(self._decode(args).bs, [bs for bs in ladder if bs <= 48])

    def test_small_tier_ladder_is_left_alone(self):
        # The <20GB memory tier tops the ladder out at 8, below the ceiling. The ceiling must
        # only ever narrow: assigning it unconditionally would raise max_bs from 8 to 48 there,
        # growing exactly the memory this is here to hold still. The bug is invisible on the
        # large-memory tiers CI runs on, so only this case catches it.
        args = self._args(ladder_max_bs=8)
        args.cuda_graph_config.decode.bs = [1, 2, 4, 8]
        handle_speculative_decoding(args)
        self.assertEqual(self._decode(args).max_bs, 8)
        self.assertEqual(self._decode(args).bs, [1, 2, 4, 8])

    def test_dp_attention_ceiling_is_per_worker(self):
        # The capture path clips the ladder to req_to_token_pool.size, which is the admission
        # limit divided by the attention-DP degree -- so the ceiling the historical 48 imposes
        # under DP attention is 48 // dp, not 48. Holding the ladder at the bare 48 would
        # lengthen the per-worker ladder (12 -> 48 here), growing exactly the capture memory
        # this is here to hold still.
        args = self._args(ladder_max_bs=128, enable_dp_attention=True, dp_size=4)
        args.cuda_graph_config.decode.bs = [1, 2, 4, 8, 12, 16, 32, 48, 64, 128]
        _speculative_admission_default(args)
        self.assertEqual(self._resolved(args).max_running_requests, 128)
        self.assertEqual(self._decode(args).max_bs, 12)
        self.assertEqual(self._decode(args).bs, [1, 2, 4, 8, 12])

    def test_explicit_graph_config_keeps_todays_limit_and_ladder(self):
        # --cuda-graph-max-bs-decode without --max-running-requests is clipped to the admission
        # limit by the capture path, so at today's 48 the flag is effectively inert. Deriving a
        # higher limit would silently un-clip it and grow graph memory on a configuration whose
        # author never asked for more admission; the limit has to stay at the historical value,
        # and the explicitly-set ladder fields must not be rewritten.
        for overrides in (
            {"cuda_graph_max_bs_decode": 256},
            {"cuda_graph_bs_decode": [1, 16, 64, 256]},
        ):
            with self.subTest(**overrides):
                args = self._args(ladder_max_bs=256, **overrides)
                args.cuda_graph_config.decode.bs = [1, 16, 64, 256]
                handle_speculative_decoding(args)
                self.assertEqual(
                    self._resolved(args).max_running_requests,
                    _SPEC_MIN_MAX_RUNNING_REQUESTS,
                )
                self.assertEqual(self._decode(args).max_bs, 256)
                self.assertEqual(self._decode(args).bs, [1, 16, 64, 256])

    def test_explicit_limit_leaves_the_ladder_untouched(self):
        # A server that passed --max-running-requests already sized its ladder against the limit
        # it chose. Narrowing it would silently push its large batches onto the eager path.
        args = self._args(ladder_max_bs=256, max_running_requests=512)
        args.cuda_graph_config.decode.bs = [1, 16, 64, 256]
        handle_speculative_decoding(args)
        self.assertEqual(self._decode(args).max_bs, 256)
        self.assertEqual(self._decode(args).bs, [1, 16, 64, 256])

    def test_non_speculative_server_is_left_for_the_capacity_estimator(self):
        # The hook runs unconditionally. Leaving None is what lets resolve_max_num_reqs size the
        # limit from KV capacity, exactly as before this default existed.
        args = self._args(speculative_algorithm=None)
        handle_speculative_decoding(args)
        self.assertIsNone(self._resolved(args).max_running_requests)


if __name__ == "__main__":
    unittest.main()
