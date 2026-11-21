"""DeepSeek-V4.1 decoder SWA bounded replay in ROCm breakable prefill graphs: the HIP
backend's graph tail block (real tail rows first, padding rows dropped by the scatter),
the exact tail the break switches onto, the attention-DP refusal, and the HIP tail
layer stack the break and the tail graphs run."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

SWA_LOC_OFFSET = 5000


def _batch(extend_lens, graph_rows):
    """A prefill batch whose rows are padded to the captured shape graph_rows."""
    num_tokens = sum(extend_lens)
    pad = graph_rows - num_tokens
    return SimpleNamespace(
        extend_seq_lens_cpu=list(extend_lens),
        seq_lens_cpu=torch.tensor(extend_lens),
        seq_lens=torch.tensor(extend_lens),
        req_pool_indices=torch.arange(len(extend_lens)) + 3,
        positions=torch.cat(
            [torch.arange(n) for n in extend_lens]
            + [torch.zeros(pad, dtype=torch.int64)]
        ),
        out_cache_loc=torch.cat(
            [100 + torch.arange(num_tokens), torch.zeros(pad, dtype=torch.int64)]
        ),
    )


def _tail_rows(extend_lens):
    rows, start = [], 0
    for n in extend_lens:
        rows.append(torch.arange(start + max(n - 128, 0), start + n))
        start += n
    return torch.cat(rows)


@unittest.skipUnless(is_hip(), "ROCm attention backend")
class TestHipPrefillGraphTail(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention import deepseek_v4_backend as cuda_backend
        from sglang.srt.layers.attention import deepseek_v4_backend_hip_radix as hip

        cls.hip = hip
        cls.cuda_backend = cuda_backend

    def _backend(self, max_running_requests=4):
        backend = object.__new__(self.hip.DeepseekV4HipRadixBackend)
        backend.is_dspark = False
        backend.enable_decoder_swa_bounded_replay = True
        backend.enable_hip_prefill_graph_tail = True
        backend.model_runner = SimpleNamespace(
            server_args=SimpleNamespace(max_running_requests=max_running_requests),
            req_to_token_pool=SimpleNamespace(size=8),
        )
        backend._bcg_tail_capacity = None
        backend._bcg_tail_buffers = None
        backend._bcg_tail_metadata = {}
        backend.token_to_kv_pool = SimpleNamespace(
            translate_loc_from_full_to_swa=lambda loc: loc + SWA_LOC_OFFSET
        )
        return backend

    def _build(self, backend, batch, **kwargs):
        calls = []

        def init_forward_metadata_prefill(self, **kw):
            calls.append(kw)
            return SimpleNamespace(
                core_attn_metadata=SimpleNamespace(
                    request_window_layout=None, swa_out_cache_loc=None
                ),
                low_ratio_req_indices=None,
                low_ratio_pos_i64=None,
                late_layer_tail=None,
            )

        cls = self.hip.DeepseekV4HipRadixBackend
        with (
            patch.object(
                cls, "init_forward_metadata_prefill", init_forward_metadata_prefill
            ),
            patch.object(
                cls, "_refresh_fp4_prefill_workspace", lambda self, *a, **k: None
            ),
            patch.object(self.hip, "late_layer_whole_extend", return_value=None),
        ):
            metadata = backend._build_late_layer_tail_metadata(batch, **kwargs)
        (kw,) = calls
        return metadata, kw

    def test_graph_block_puts_the_real_tail_rows_first(self):
        backend = self._backend()  # capacity 4 x 128 = 512 rows
        lens = [300, 50, 1000]
        batch = _batch(lens, 2048)
        metadata, kw = self._build(
            backend, batch, for_prefill_graph=True, max_seq_len_override=16384
        )
        tail = metadata.late_layer_tail
        want = _tail_rows(lens)
        real = want.shape[0]
        self.assertEqual(real, 128 + 50 + 128)
        self.assertEqual(tail.real_rows_count, real)
        self.assertEqual(tail.token_indices.shape[0], 512)
        self.assertEqual(tail.extend_seq_lens_cpu, [128, 50, 128])
        self.assertIsNone(tail.contiguous_start)
        torch.testing.assert_close(tail.token_indices[:real], want)
        torch.testing.assert_close(tail.scatter_indices[:real], want)
        torch.testing.assert_close(tail.positions[:real], batch.positions[want])
        self.assertTrue(bool((tail.token_indices[real:] == 0).all()))
        self.assertTrue(bool((tail.positions[real:] == 0).all()))
        # Padding rows scatter past the shape's rows, which the outer cut drops.
        self.assertTrue(bool((tail.scatter_indices[real:] == 2048).all()))
        loc = kw["out_cache_loc"]
        torch.testing.assert_close(loc[:real], batch.out_cache_loc[want])
        self.assertTrue(bool((loc[real:] == 0).all()))
        torch.testing.assert_close(
            tail.swa_out_cache_loc, (loc + SWA_LOC_OFFSET).to(torch.int32)
        )
        self.assertEqual(kw["num_tokens"], real)
        self.assertEqual(kw["max_seq_len"], 16384)
        self.assertTrue(kw["use_prefill_cuda_graph"])
        self.assertFalse(kw["exact_num_tokens"])
        self.assertEqual(kw["swa_replay_start"].shape[0], 512)
        req = torch.repeat_interleave(
            batch.req_pool_indices, torch.tensor([128, 50, 128])
        )
        torch.testing.assert_close(
            metadata.low_ratio_req_indices,
            torch.cat([req, req.new_zeros(512 - real)]),
        )
        torch.testing.assert_close(metadata.low_ratio_pos_i64, tail.positions.long())

    def test_graph_blocks_share_one_storage_across_shapes(self):
        backend = self._backend()
        big, _ = self._build(backend, _batch([1000], 1024), for_prefill_graph=True)
        big = big.late_layer_tail
        self.assertEqual(big.token_indices.shape[0], 512)
        # One request: its tail is a contiguous slice, written as indices.
        torch.testing.assert_close(big.token_indices[:128], torch.arange(872, 1000))
        small, _ = self._build(backend, _batch([200, 100], 384), for_prefill_graph=True)
        small = small.late_layer_tail
        self.assertEqual(small.token_indices.shape[0], 384)
        # Every block is a view of the same buffers, refreshed by each build.
        self.assertEqual(small.token_indices.data_ptr(), big.token_indices.data_ptr())
        torch.testing.assert_close(big.token_indices[:228], _tail_rows([200, 100]))
        self.assertEqual(small.real_rows_count, 228)
        self.assertTrue(bool((small.scatter_indices[228:] == 384).all()))

    def test_exact_tail_for_the_break(self):
        backend = self._backend()
        lens = [300, 50]
        metadata, kw = self._build(backend, _batch(lens, 350))
        tail = metadata.late_layer_tail
        want = _tail_rows(lens)
        self.assertIsNone(tail.scatter_indices)
        self.assertFalse(kw["use_prefill_cuda_graph"])
        self.assertTrue(kw["exact_num_tokens"])
        self.assertEqual(kw["num_tokens"], want.shape[0])
        torch.testing.assert_close(tail.token_indices, want)
        self.assertEqual(metadata.low_ratio_req_indices.shape[0], want.shape[0])
        self.assertIsNone(backend._bcg_tail_buffers)

    def _can_run(self, backend, attn_dp_size):
        batch = object()
        with (
            patch.object(
                self.hip,
                "get_parallel",
                return_value=SimpleNamespace(attn_dp_size=attn_dp_size),
            ),
            patch.object(
                self.cuda_backend.DeepseekV4AttnBackend,
                "can_run_prefill_cuda_graph",
                return_value="shared",
            ) as shared,
        ):
            result = backend.can_run_prefill_cuda_graph(batch)
        return result, shared, batch

    def test_attention_dp_keeps_bounded_replay_eager(self):
        backend = self._backend()
        result, shared, _ = self._can_run(backend, attn_dp_size=2)
        self.assertFalse(result)
        shared.assert_not_called()
        result, shared, batch = self._can_run(backend, attn_dp_size=1)
        self.assertEqual(result, "shared")
        shared.assert_called_once_with(backend, batch)

    def test_off_by_default_leaves_the_shared_gate(self):
        backend = self._backend()
        backend.enable_hip_prefill_graph_tail = False
        result, shared, batch = self._can_run(backend, attn_dp_size=2)
        self.assertEqual(result, "shared")
        shared.assert_called_once_with(backend, batch)


class _State:
    def __init__(self, tag, calls):
        self.tag, self.calls = tag, calls
        self.residual, self.pre = f"residual{tag}", f"pre{tag}"

    def materialized(self, cfg):
        self.calls.append(("materialized", self.tag, cfg))
        return self


@unittest.skipUnless(is_hip(), "ROCm tail layer stack")
class TestHipTailLayerStack(CustomTestCase):
    def _run(self, *, capture_dspark):
        from sglang.kernels.ops.layernorm import mhc as mhc_ops
        from sglang.srt.models import deepseek_v4_platform_stage as stage
        from sglang.srt.models import deepseek_v4_replay_tail as replay_tail

        calls = []
        kwargs = dict(
            positions="positions",
            input_ids="ids",
            input_ids_global="ids_global",
            forward_batch="batch",
        )

        def layer(i, fused):
            def forward_hc_pre_from_prev(**kw):
                calls.append(("layer", i, kw.pop("state").tag))
                self.assertEqual(kw, kwargs)
                return _State(i, calls)

            return SimpleNamespace(
                engram=None,
                hc_boundary_fused=fused,
                hc_cfg=f"cfg{i}",
                forward_hc_pre_from_prev=forward_hc_pre_from_prev,
            )

        def fused_boundary(model, i, state, **kw):
            calls.append(("fused", i, state.tag))
            self.assertEqual(kw, dict(kwargs, capture_dspark=capture_dspark))
            return _State(i, calls)

        model = SimpleNamespace(
            layers={20: layer(20, False), 21: layer(21, True), 22: layer(22, False)},
            dspark_layers_to_capture={21},
        )
        tail = object.__new__(replay_tail.DecoderReplayTail)
        tail._model, tail._layer_ids = model, range(20, 23)
        with (
            patch.object(replay_tail, "check_cuda_graph_backend", return_value=True),
            patch.object(stage, "forward_layer_fused_boundary", fused_boundary),
            patch.object(mhc_ops, "mhc_stream_mean", lambda r: f"mean({r})"),
        ):
            out = tail._layer_stack_hip(
                _State(19, calls), capture_dspark=capture_dspark, **kwargs
            )
        return out, calls

    def test_fused_boundary_layers_and_draft_inputs(self):
        (residual, pre, aux), calls = self._run(capture_dspark=True)
        self.assertEqual(
            calls,
            [
                ("layer", 20, 19),
                ("fused", 21, 20),
                ("layer", 22, 21),
                ("materialized", 22, "cfg22"),
            ],
        )
        # The draft reads layer 21's input: the residual layer 20 left.
        self.assertEqual(aux, ["mean(residual20)"])
        self.assertEqual((residual, pre), ("residual22", "pre22"))

    def test_no_draft_inputs_without_dspark(self):
        (_, _, aux), calls = self._run(capture_dspark=False)
        self.assertEqual(
            [c[:2] for c in calls[:3]], [("layer", 20), ("fused", 21), ("layer", 22)]
        )
        self.assertEqual(aux, [])


if __name__ == "__main__":
    unittest.main()
