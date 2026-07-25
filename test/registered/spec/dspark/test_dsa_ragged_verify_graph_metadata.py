"""DSA ragged verify: graph-recorded prep must match the eager build.

`DeepseekSparseAttnBackend.init_forward_metadata_in_graph` re-derives every
target-verify metadata tensor from the staged verify lens with one fused triton
kernel, replacing the eager torch chain. Nothing else pins the two against each
other, so this drives both against the same batch and diffs the tensors the
attention/indexer kernels actually read.

Also covers the two failure modes that are invisible in end-to-end output:
  * a consumer refreshed OUTSIDE the graph reads the pre-replay (stale) buffer
    -- checked here for the flashmla_kv schedule;
  * a width-capped padded layout leaves the tier's tail rows claimed by no
    request; they must come out length 0, not carry a neighbour's kv length.
"""

import types
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    build_capture_verify_lens,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

NEXT_N = 4
# Ragged widths, each <= NEXT_N. Sum 10 < TIER, so the padded layout also leaves
# uncovered tail rows (padded_bs == bs leaves no padding row to absorb slack).
VERIFY_LENS = [1, 4, 2, 3]
# Keep every seq_len well under the indexer's top-k width: `compute_dsa_seqlens`
# clamps to dsa_index_topk, and a saturated vector hides per-row differences --
# with 128-token prefixes the flashmla schedule comes out identical for the
# capture and replay layouts, which silently defuses the staleness check below.
PREFIX_LENS = (8, 20, 5, 12)
BS = len(VERIFY_LENS)
TIER = BS * NEXT_N
COVERED = sum(VERIFY_LENS)

# Tensors the fused in-graph rebuild owns, with the row count each is valid for.
_PER_REQ_FIELDS = ("cache_seqlens_int32",)
_PER_ROW_FIELDS = ("dsa_seqlens_expanded", "dsa_cache_seqlens_int32")


def _make_case():
    from sglang.test.kits.attention_unittest.attention_methods.dsa_attention import (
        DSA_PAGE_SIZE,
        DSAAttentionCase,
    )

    return DSAAttentionCase(
        name="dsa_ragged_verify_graph_metadata",
        backend="dsa",
        forward_mode=ForwardMode.TARGET_VERIFY,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        prefix_lens=PREFIX_LENS,
        # DSAMockModelRunner derives speculative_num_draft_tokens from this.
        extend_lens=(NEXT_N,) * BS,
    )


def _layout(verify_lens_cpu, device):
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=list(verify_lens_cpu), device=device, grid=[TIER]
    )


def _spec_info(layout):
    return types.SimpleNamespace(ragged_verify_layout=layout)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDsaRaggedVerifyGraphMetadata(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.device("cuda")

    def _build(self):
        from sglang.test.kits.attention_unittest.attention_methods.dsa_attention import (
            build_dsa_sparse_attention_fixture,
        )

        return build_dsa_sparse_attention_fixture(
            self,
            _make_case(),
            device=str(self.device),
            dsa_decode_backend="flashmla_kv",
        )

    def _run_both_paths(self, *, flashmla_calls=None):
        """Return (eager_metadata_snapshot, graph_metadata, backend).

        `flashmla_calls`: optional list; the `cache_seqlens` handed to
        `_compute_flashmla_metadata` during the replay step is appended to it.
        """
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        fixture = self._build()
        backend = fixture.backend
        fb = fixture.forward_batch
        raw = _layout(VERIFY_LENS, self.device)

        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"), torch.no_grad():
            with forward_context(ForwardContext(attn_backend=backend)):
                # --- eager reference (raw layout, no padding) ---
                fb.spec_info = _spec_info(raw)
                backend.init_forward_metadata(fb)
                eager = {
                    name: getattr(backend.forward_metadata, name).clone()
                    for name in _PER_REQ_FIELDS + _PER_ROW_FIELDS
                }

                # --- graph path: capture at the tier, then replay the raw layout ---
                backend.init_cuda_graph_state(max_bs=BS, max_num_tokens=TIER)
                capture_lens = build_capture_verify_lens(
                    num_tokens=TIER, num_slots=BS, num_draft_tokens=NEXT_N
                )
                fb.spec_info = _spec_info(_layout(capture_lens, self.device))
                backend.init_forward_metadata_out_graph(fb, in_capture=True)
                backend.init_forward_metadata_in_graph(fb)

                if flashmla_calls is not None:
                    original = backend._compute_flashmla_metadata

                    def _spy(*, cache_seqlens, seq_len_q):
                        flashmla_calls.append(cache_seqlens.detach().clone())
                        return original(
                            cache_seqlens=cache_seqlens, seq_len_q=seq_len_q
                        )

                    backend._compute_flashmla_metadata = _spy

                fb.spec_info = _spec_info(raw)
                backend.init_forward_metadata_out_graph(fb)
                backend.init_forward_metadata_in_graph(fb)
                graph = backend.forward_metadata

                if flashmla_calls is not None:
                    del backend._compute_flashmla_metadata

        return eager, graph, backend

    def test_graph_prep_matches_eager_metadata(self):
        eager, graph, _ = self._run_both_paths()

        for name in _PER_REQ_FIELDS:
            torch.testing.assert_close(
                getattr(graph, name)[:BS],
                eager[name][:BS],
                rtol=0,
                atol=0,
                msg=f"{name} (per request)",
            )
        for name in _PER_ROW_FIELDS:
            torch.testing.assert_close(
                getattr(graph, name)[:COVERED],
                eager[name][:COVERED],
                rtol=0,
                atol=0,
                msg=f"{name} (claimed rows)",
            )
        # cu_seqlens_k is a prefix sum over requests; compare the live prefix.
        torch.testing.assert_close(
            graph.cu_seqlens_k[: BS + 1],
            torch.nn.functional.pad(
                torch.cumsum(graph.cache_seqlens_int32[:BS], dim=0, dtype=torch.int32),
                (1, 0),
            ),
            rtol=0,
            atol=0,
            msg="cu_seqlens_k",
        )

    def test_uncovered_tail_rows_are_zero(self):
        _, graph, _ = self._run_both_paths()
        self.assertLess(COVERED, TIER, "fixture must leave unclaimed tail rows")
        tail = torch.zeros(TIER - COVERED, dtype=torch.int32, device=self.device)
        for name in _PER_ROW_FIELDS:
            torch.testing.assert_close(
                getattr(graph, name)[COVERED:TIER],
                tail,
                rtol=0,
                atol=0,
                msg=f"{name} (unclaimed rows)",
            )

    def test_flashmla_schedule_sees_rebuilt_seqlens(self):
        """The flashmla_kv schedule derives from dsa_cache_seqlens, which only the
        in-graph rebuild fills, so its refresh must run after that rebuild -- doing
        it in the out-graph step reads the previous step's buffer.

        Asserted on the refresh's INPUT rather than on the resulting schedule:
        `get_mla_metadata` leaves one column of its output uninitialized, so two
        calls on identical seqlens are not bitwise equal and diffing the schedule
        tensor would compare scratch memory.
        """
        calls = []
        _, graph, backend = self._run_both_paths(flashmla_calls=calls)
        if backend.dsa_decode_impl != "flashmla_kv":
            self.skipTest(f"dsa_decode_impl={backend.dsa_decode_impl}, not flashmla_kv")
        self.assertIsNotNone(graph.flashmla_metadata)
        self.assertTrue(calls, "flashmla schedule was never refreshed on replay")

        torch.testing.assert_close(
            calls[-1],
            graph.dsa_cache_seqlens_int32[: calls[-1].numel()],
            rtol=0,
            atol=0,
            msg="flashmla refresh ran on pre-rebuild (stale) dsa_cache_seqlens",
        )


if __name__ == "__main__":
    unittest.main()
