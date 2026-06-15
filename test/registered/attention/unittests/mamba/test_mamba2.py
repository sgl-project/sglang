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
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.mamba2_attention import (
    DEFAULT_CONV_KERNEL,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MAMBA_CHUNK_SIZE,
    DEFAULT_N_GROUPS,
    DEFAULT_NUM_HEADS,
    DEFAULT_STATE_SIZE,
    Mamba2AttentionCase,
    build_mamba2_attention_fixture,
    make_mamba2_cases,
    run_mamba2_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_mamba2_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_mamba2_eagle_verify_case,
    run_mamba2_eagle_verify_cuda_graph_case,
)

register_cuda_ci(est_time=10, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonMamba2BackendCorrectness(CustomTestCase):
    CASES = make_mamba2_cases("triton")
    # `seq_lens_cpu=[5, 1, 1]` mixes a live row with two cuda-graph
    # fill-value rows so the replay padding count is non-trivial.
    REPLAY_METADATA_CASE = Mamba2AttentionCase(
        name="mamba2_decode_replay_metadata_padding",
        backend="triton",
        forward_mode=ForwardMode.DECODE,
        num_heads=DEFAULT_NUM_HEADS,
        head_dim=DEFAULT_HEAD_DIM,
        state_size=DEFAULT_STATE_SIZE,
        n_groups=DEFAULT_N_GROUPS,
        conv_kernel=DEFAULT_CONV_KERNEL,
        mamba_chunk_size=DEFAULT_MAMBA_CHUNK_SIZE,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        page_size=16,
        prefix_lens=(4, 0, 0),
    )

    CUDA_GRAPH_CASES = (
        Mamba2AttentionCase(
            name="runner_cuda_graph_mamba2_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_heads=DEFAULT_NUM_HEADS,
            head_dim=DEFAULT_HEAD_DIM,
            state_size=DEFAULT_STATE_SIZE,
            n_groups=DEFAULT_N_GROUPS,
            conv_kernel=DEFAULT_CONV_KERNEL,
            mamba_chunk_size=DEFAULT_MAMBA_CHUNK_SIZE,
            hidden_size=DEFAULT_HIDDEN_SIZE,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )
    # Chain verify (topk=1) across EAGLE plus the three non-EAGLE chain
    # spec kinds (frozen_kv_mtp / dflash / ngram). Mamba2's SSM kernel
    # processes draft tokens linearly regardless of the spec_info tree
    # mask, so the EXTEND-style recurrence reference doubles as the
    # chain verify reference across all kinds. Tree verify (topk>1) is
    # structurally unsupported and skip-gated at the runner.
    EAGLE_VERIFY_CASES = tuple(
        (
            Mamba2AttentionCase(
                name=f"runner_{spec_kind}_verify_mamba2_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=DEFAULT_NUM_HEADS,
                head_dim=DEFAULT_HEAD_DIM,
                state_size=DEFAULT_STATE_SIZE,
                n_groups=DEFAULT_N_GROUPS,
                conv_kernel=DEFAULT_CONV_KERNEL,
                mamba_chunk_size=DEFAULT_MAMBA_CHUNK_SIZE,
                hidden_size=DEFAULT_HIDDEN_SIZE,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            spec_kind,
        )
        for spec_kind in ("eagle", "frozen_kv_mtp", "dflash", "ngram")
    )
    EAGLE_VERIFY_CUDA_GRAPH_CASES = (
        (
            Mamba2AttentionCase(
                name="runner_cuda_graph_eagle_verify_mamba2_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=DEFAULT_NUM_HEADS,
                head_dim=DEFAULT_HEAD_DIM,
                state_size=DEFAULT_STATE_SIZE,
                n_groups=DEFAULT_N_GROUPS,
                conv_kernel=DEFAULT_CONV_KERNEL,
                mamba_chunk_size=DEFAULT_MAMBA_CHUNK_SIZE,
                hidden_size=DEFAULT_HIDDEN_SIZE,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
        ),
    )

    def test_projected_mamba2_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mamba2_attention_case(self, case)

    # Layout-robustness. See dense/test_triton.py for the rationale.
    # Reuse the case generator's first two cases to avoid duplicating
    # all the Mamba2-specific config fields.
    def test_layout_robustness_cases(self):
        cases = [
            self.CASES[0],  # extend exact-page (zero-prefix, multi-token)
            self.CASES[3],  # extend with prefix (`prefix=16, extend=16`)
        ]
        for case in cases:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                with self.subTest(case=case.name, layout=layout):
                    run_mamba2_attention_case(self, case, loc_layout=layout)

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mamba2_cuda_graph_decode_case(self, case)

    def test_runner_mode_eagle_verify_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_mamba2_eagle_verify_case(self, case, topk=topk, spec_kind=spec_kind)

    def test_runner_mode_eagle_verify_cuda_graph_cases(self):
        for case, topk in self.EAGLE_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_mamba2_eagle_verify_cuda_graph_case(self, case, topk=topk)

    # PCG/BCG split-op extend is deliberately NOT covered. The
    # `MambaMixer2.forward` asserts `num_actual_tokens ==
    # projected_states.shape[0]` (`mamba.py:467`) — the projection step
    # requires `hidden_states.shape[0]` to equal the LIVE token count
    # exactly, with no padding tolerance. The shared split-op runner
    # pads `hidden_states` to a fixed `static_num_tokens` upper bound
    # and then relies on the backend's per-layer slicing contract via
    # `num_token_non_padded_cpu`. Mamba2 doesn't support this padding
    # because its mixer projects BEFORE the attention dispatch sees
    # `num_token_non_padded_cpu`. Landing this needs either a Mamba2
    # mixer change to accept padded `hidden_states`, or a split-op
    # runner variant that passes unpadded `hidden_states` while still
    # padding the `forward_batch.input_ids` / `out_cache_loc`.

    def test_mamba2_replay_metadata_padding_indices(self):
        # Drive `init_forward_metadata_out_graph` (replay path) directly with
        # `seq_lens_cpu=[5, 1, 1]` (two trailing rows at the cuda-graph
        # fill value 1) so the padding-row count is observable in
        # `state_indices_list[bs - 1]`.
        case = self.REPLAY_METADATA_CASE
        fixture = build_mamba2_attention_fixture(
            self,
            case,
            disable_cuda_graph=False,
            runner_batch_size=case.batch_size,
        )
        backend = fixture.backend
        bs = case.batch_size

        backend.init_cuda_graph_state(max_bs=bs, max_num_tokens=bs)

        # Sentinel distinguishes "never written" from "overwritten with -1".
        backend.state_indices_list[bs - 1].fill_(99)

        device = fixture.runner.device
        req_pool_indices = torch.arange(bs, dtype=torch.int32, device=device)
        seq_lens_cpu = torch.tensor([5, 1, 1], dtype=torch.int32, device="cpu")
        seq_lens = seq_lens_cpu.to(device=device)

        # Slot 7 on req 0 must survive; the trailing two rows must be -1.
        fixture.runner.req_to_token_pool.req_index_to_mamba_index_mapping[
            req_pool_indices
        ] = torch.tensor([7, 0, 0], dtype=torch.int32, device=device)

        fb = SimpleNamespace(
            batch_size=bs,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=int(seq_lens_cpu.sum().item()),
            spec_info=None,
            encoder_lens=None,
        )
        backend.init_forward_metadata_out_graph(fb)

        state_indices = backend.state_indices_list[bs - 1].cpu().tolist()
        self.assertEqual(
            state_indices,
            [7, -1, -1],
            "`MambaAttnBackendBase._replay_metadata` must use the "
            "unmutated `seq_lens_cpu` to count cuda-graph padding rows "
            "(== fill value 1). With `seq_lens_cpu - 1` (M21) the "
            "padding count for `[5, 1, 1]` drops from 2 to 0, leaving "
            "the trailing rows holding the real mamba indices instead "
            "of -1.",
        )

    # Hybrid dispatch fan-out tests (MagicMock-based) — same pattern as
    # GDN. `Mamba2AttnBackend` inherits the `MambaAttnBackendBase`
    # capture/replay contract through `HybridLinearAttnBackend`, so a
    # dispatch-layer slice mutation (e.g. `attn_backend_list[1:]` vs
    # `[:1]`) would silently break Mamba2 dispatch without these spies.

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
        """Assert `method_mock` was called exactly once and that each
        sentinel object identity appears in the call's positional or
        keyword args (tolerates positional↔keyword refactors inside
        `HybridLinearAttnBackend`)."""
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

        for sub_backend in (full_attn_backend, linear_attn_backend):
            self._assert_fanout_forwarded(
                sub_backend.init_forward_metadata_out_graph, fb
            )

    def test_hybrid_dispatch_capture_init_forward_metadata_fan_out(self):
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
