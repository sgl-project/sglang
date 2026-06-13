import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.gdn_attention import (
    GDNAttentionCase,
    make_gdn_cases,
    run_gdn_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_gdn_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_extend_runner import (
    run_gdn_eagle_draft_extend_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_gdn_eagle_verify_case,
    run_gdn_eagle_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_gdn_split_op_extend_case,
)

register_cuda_ci(est_time=20, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-large-amd")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonGDNBackendCorrectness(CustomTestCase):
    CASES = make_gdn_cases("triton")
    CUDA_GRAPH_CASES = (
        GDNAttentionCase(
            name="runner_cuda_graph_gdn_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_k_heads=2,
            num_v_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )
    SPLIT_OP_CASES = (
        (
            GDNAttentionCase(
                name="runner_split_op_gdn_extend_ragged_page_boundary",
                backend="triton",
                forward_mode=ForwardMode.EXTEND,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(0, 8, 16),
                extend_lens=(15, 8, 1),
            ),
            32,
        ),
    )
    # GDN verify covers EAGLE chain/tree plus the non-EAGLE chain spec
    # kinds (frozen_kv_mtp, dflash, ngram). All three pass against the
    # pure-PyTorch gated-delta recurrence reference; the GDN backend
    # treats them uniformly via the spec_info custom/tree mask.
    EAGLE_VERIFY_CASES = (
        (
            GDNAttentionCase(
                name="runner_eagle_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_eagle_verify_gdn_tree",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_frozen_kv_mtp_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            GDNAttentionCase(
                name="runner_dflash_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            GDNAttentionCase(
                name="runner_ngram_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )
    EAGLE_VERIFY_CUDA_GRAPH_CASES = (
        (
            GDNAttentionCase(
                name="runner_cuda_graph_eagle_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_eagle_verify_gdn_tree",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_frozen_kv_mtp_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_dflash_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_ngram_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )

    def test_projected_gdn_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_gdn_attention_case(self, case)

    # Layout-robustness. See dense/test_triton.py for the rationale.
    # shuffled_pages is the default for all tests; this method opts
    # into the more aggressive interleaved_pages + non_monotonic_extend.
    # GDN Triton handles all non-tidy layouts cleanly.
    LAYOUT_ROBUSTNESS_CASES = (
        GDNAttentionCase(
            name="layout_gdn_extend_two_request",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_k_heads=4,
            num_v_heads=4,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(16, 16),
        ),
        GDNAttentionCase(
            name="layout_gdn_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_k_heads=4,
            num_v_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_gdn_attention_case(self, case, loc_layout=layout)

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_gdn_cuda_graph_decode_case(self, case)

    @unittest.skipIf(
        is_hip(),
        "split-op extend runner exercises the piecewise-CUDA-graph path "
        "(TcPiecewiseForwardContext.num_tokens), which is not wired on ROCm.",
    )
    def test_runner_mode_split_op_extend_cases(self):
        for case, static_num_tokens in self.SPLIT_OP_CASES:
            for breakable in (False, True):
                runner = "bcg" if breakable else "pcg"
                with self.subTest(
                    case=case.name,
                    backend=case.backend,
                    runner=runner,
                ):
                    run_gdn_split_op_extend_case(
                        self,
                        case,
                        breakable=breakable,
                        static_num_tokens=static_num_tokens,
                    )

    def test_runner_mode_eagle_verify_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_gdn_eagle_verify_case(self, case, topk=topk, spec_kind=spec_kind)

    def test_runner_mode_eagle_verify_cuda_graph_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_gdn_eagle_verify_cuda_graph_case(
                    self, case, topk=topk, spec_kind=spec_kind
                )

    # EAGLE / Frozen-KV MTP DRAFT_EXTEND eager — `HybridLinearAttnBackend`
    # raises `ValueError("Invalid forward mode")` for DRAFT_EXTEND CG
    # capture (`hybrid_linear_attn_backend.py:509,572`), so CG is
    # structurally blocked across the family (GDN/KDA/Lightning/Mamba2).
    # The EXTEND-style gated-delta recurrence reference doubles as the
    # DRAFT_EXTEND reference across both spec kinds.
    EAGLE_DRAFT_EXTEND_CASES = (
        (
            GDNAttentionCase(
                name="runner_eagle_draft_extend_gdn",
                backend="triton",
                forward_mode=ForwardMode.DRAFT_EXTEND,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_frozen_kv_mtp_draft_extend_gdn",
                backend="triton",
                forward_mode=ForwardMode.DRAFT_EXTEND,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            "frozen_kv_mtp",
        ),
    )

    def test_runner_mode_eagle_draft_extend_cases(self):
        for case, spec_kind in self.EAGLE_DRAFT_EXTEND_CASES:
            with self.subTest(
                case=case.name, backend=case.backend, spec_kind=spec_kind
            ):
                run_gdn_eagle_draft_extend_case(self, case, spec_kind=spec_kind)

    # Spy directly on each sub-backend's `init_forward_metadata*` so
    # dispatch-layer slice mutations show up as a missing call, which
    # forward-output assertions can miss when the fixture happens to
    # use identical capture/replay metadata.

    def _make_dispatch_spy_backend(self):
        full_attn_backend = MagicMock(name="full_attn_backend")
        # `HybridLinearAttnBackend.__init__` aliases these buffer refs.
        full_attn_backend.token_to_kv_pool = object()
        full_attn_backend.req_to_token_pool = object()

        linear_attn_backend = MagicMock(
            spec=MambaAttnBackendBase, name="linear_attn_backend"
        )

        backend = HybridLinearAttnBackend(
            full_attn_backend,
            linear_attn_backend,
            full_attn_layers=[],
        )
        return backend, full_attn_backend, linear_attn_backend

    @staticmethod
    def _assert_fanout_forwarded(method_mock, *sentinels):
        """Assert `method_mock` was called exactly once and that each sentinel
        object identity is present in the call's positional or keyword args.
        Tolerates production switching between positional / keyword arg
        forwarding (the previous `assert_called_once_with(*positional)` form
        would silently break on such a refactor)."""
        method_mock.assert_called_once()
        call = method_mock.call_args
        forwarded = list(call.args) + list(call.kwargs.values())
        for sentinel in sentinels:
            if not any(v is sentinel for v in forwarded):
                raise AssertionError(
                    f"sentinel {sentinel!r} not forwarded by "
                    f"{method_mock._mock_name or method_mock}; call_args={call}"
                )

    def test_hybrid_dispatch_eager_init_forward_metadata_fan_out(self):
        backend, full_attn_backend, linear_attn_backend = (
            self._make_dispatch_spy_backend()
        )
        # Sentinel exposes the attribute production reads at the dispatch
        # gate (`forward_mode.is_draft_extend_v2()`); returns False so the
        # fan-out path that delegates to both children is exercised, which
        # is what these spy tests assert.
        sentinel_forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_draft_extend_v2=lambda: False)
        )
        backend.init_forward_metadata(sentinel_forward_batch)
        self._assert_fanout_forwarded(
            full_attn_backend.init_forward_metadata, sentinel_forward_batch
        )
        self._assert_fanout_forwarded(
            linear_attn_backend.init_forward_metadata, sentinel_forward_batch
        )

    def _make_sentinel_fb(self):
        return SimpleNamespace(
            batch_size=3,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=object(),
            seq_lens=object(),
            seq_lens_cpu=object(),
            seq_lens_sum=42,
            spec_info=object(),
            encoder_lens=None,
            positions=object(),
            input_ids=object(),
            out_cache_loc=None,
        )

    def test_hybrid_dispatch_replay_init_forward_metadata_fan_out(self):
        backend, full_attn_backend, linear_attn_backend = (
            self._make_dispatch_spy_backend()
        )

        fb = self._make_sentinel_fb()
        backend.init_forward_metadata_out_graph(fb)

        # We assert sentinel identity rather than exact (args, kwargs) shape
        # so a positional↔keyword refactor inside `HybridLinearAttnBackend`
        # doesn't trip the test as long as the fb still flows through.
        for sub_backend in (full_attn_backend, linear_attn_backend):
            self._assert_fanout_forwarded(
                sub_backend.init_forward_metadata_out_graph, fb
            )

    def test_hybrid_dispatch_capture_init_forward_metadata_fan_out(self):
        # Capture mirrors the eager/replay loop shape; a slice mutation
        # there would silently miss without a spy.
        backend, full_attn_backend, linear_attn_backend = (
            self._make_dispatch_spy_backend()
        )
        fb = self._make_sentinel_fb()
        backend.init_forward_metadata_out_graph(fb, in_capture=True)

        for sub_backend in (full_attn_backend, linear_attn_backend):
            self._assert_fanout_forwarded(
                sub_backend.init_forward_metadata_out_graph, fb
            )


if __name__ == "__main__":
    unittest.main()
