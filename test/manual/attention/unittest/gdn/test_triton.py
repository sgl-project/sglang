import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.gdn_attention import (
    GDNAttentionCase,
    make_gdn_cases,
    run_gdn_attention_case,
)
from common.runner_modes.cuda_graph_decode_runner import run_gdn_cuda_graph_decode_case
from common.runner_modes.speculative_target_verify_runner import (
    run_gdn_eagle_verify_case,
    run_gdn_eagle_verify_cuda_graph_case,
)
from common.runner_modes.split_op_runner import run_gdn_split_op_extend_case


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
        ),
    )

    def test_projected_gdn_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_gdn_attention_case(self, case)

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_gdn_cuda_graph_decode_case(self, case)

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
        for case, topk in self.EAGLE_VERIFY_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_gdn_eagle_verify_case(self, case, topk=topk)

    def test_runner_mode_eagle_verify_cuda_graph_cases(self):
        for case, topk in self.EAGLE_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_gdn_eagle_verify_cuda_graph_case(self, case, topk=topk)

    # ------------------------------------------------------------------
    # Focused HybridLinearAttnBackend dispatch tests (M19, M20).
    #
    # The GDN fixture already wraps the linear backend in
    # `HybridLinearAttnBackend`, but the dispatch-layer mutations
    # `attn_backend_list[1:]` (M20, eager) and `attn_backend_list[:1]`
    # (M19, replay) escaped detection because:
    #   - The GDN fixture sets `full_attn_layers=[]`, so all real
    #     forward calls go through the linear backend; skipping the
    #     full backend's `init_forward_metadata` is therefore a no-op
    #     for the actual attention output.
    #   - The cuda-graph capture batch and replay batch use the same
    #     `req_pool_indices` shape and arange, so the linear backend's
    #     `state_indices_list[bs - 1]` ends up holding the same values
    #     whether or not the replay path runs. Skipping the linear
    #     backend's replay therefore leaves a metadata buffer that
    #     *coincidentally* matches the replay-time mamba indices.
    # These two test methods avoid the coincidence-driven invisibility
    # by spying directly on each sub-backend's
    # `init_forward_metadata*` methods. Any dispatch-layer slice
    # mutation will reduce the call count for at least one sub-backend
    # and the assertion will trip.
    # ------------------------------------------------------------------

    def _make_dispatch_spy_backend(self):
        # Use plain MagicMock subclasses of the production
        # AttentionBackend / MambaAttnBackendBase types so the
        # HybridLinearAttnBackend constructor accepts them. We only
        # need spy semantics on the `init_forward_metadata*` family;
        # everything else can stay as the default mock.
        full_attn_backend = MagicMock(name="full_attn_backend")
        # Use plain attributes for the buffer refs the wrapper aliases
        # in `__init__`.
        full_attn_backend.token_to_kv_pool = object()
        full_attn_backend.req_to_token_pool = object()

        # MambaAttnBackendBase isinstance check is loose; a MagicMock
        # passes through the wrapper unchanged.
        linear_attn_backend = MagicMock(
            spec=MambaAttnBackendBase, name="linear_attn_backend"
        )

        backend = HybridLinearAttnBackend(
            full_attn_backend,
            linear_attn_backend,
            full_attn_layers=[],
        )
        return backend, full_attn_backend, linear_attn_backend

    def test_hybrid_dispatch_eager_init_forward_metadata_fan_out(self):
        backend, full_attn_backend, linear_attn_backend = (
            self._make_dispatch_spy_backend()
        )

        sentinel_forward_batch = object()
        backend.init_forward_metadata(sentinel_forward_batch)

        full_attn_backend.init_forward_metadata.assert_called_once_with(
            sentinel_forward_batch
        )
        linear_attn_backend.init_forward_metadata.assert_called_once_with(
            sentinel_forward_batch
        )

    def test_hybrid_dispatch_replay_init_forward_metadata_fan_out(self):
        backend, full_attn_backend, linear_attn_backend = (
            self._make_dispatch_spy_backend()
        )

        sentinel_req_pool = object()
        sentinel_seq_lens = object()
        sentinel_seq_lens_cpu = object()
        sentinel_spec_info = object()

        backend.init_forward_metadata_replay_cuda_graph(
            bs=3,
            req_pool_indices=sentinel_req_pool,
            seq_lens=sentinel_seq_lens,
            seq_lens_sum=42,
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=sentinel_spec_info,
            seq_lens_cpu=sentinel_seq_lens_cpu,
        )

        expected_args = (
            3,
            sentinel_req_pool,
            sentinel_seq_lens,
            42,
            None,
            ForwardMode.DECODE,
            sentinel_spec_info,
            sentinel_seq_lens_cpu,
        )
        full_attn_backend.init_forward_metadata_replay_cuda_graph.assert_called_once_with(
            *expected_args
        )
        linear_attn_backend.init_forward_metadata_replay_cuda_graph.assert_called_once_with(
            *expected_args
        )

    def test_hybrid_dispatch_capture_init_forward_metadata_fan_out(self):
        # Capture isn't on the mutation list directly, but the wrapper
        # mirrors the same `for ... in attn_backend_list` shape, so a
        # similar slice mutation would silently miss without a spy.
        backend, full_attn_backend, linear_attn_backend = (
            self._make_dispatch_spy_backend()
        )
        sentinel_req_pool = object()
        sentinel_seq_lens = object()
        sentinel_spec_info = object()

        backend.init_forward_metadata_capture_cuda_graph(
            bs=3,
            num_tokens=3,
            req_pool_indices=sentinel_req_pool,
            seq_lens=sentinel_seq_lens,
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=sentinel_spec_info,
        )

        expected_args = (
            3,
            3,
            sentinel_req_pool,
            sentinel_seq_lens,
            None,
            ForwardMode.DECODE,
            sentinel_spec_info,
        )
        full_attn_backend.init_forward_metadata_capture_cuda_graph.assert_called_once_with(
            *expected_args
        )
        linear_attn_backend.init_forward_metadata_capture_cuda_graph.assert_called_once_with(
            *expected_args
        )


if __name__ == "__main__":
    unittest.main()
