"""DeepSeek-V4 unified_kv fp8: the packed pairs handed to the two readers.

Decode only needs Q packed -- its K is already in the ring. Prefill is a KV
source of its own, so it gets a packed K pair beside the Q one, and the same
buffers have to reach both attention and the ring write after it. Verify retains
caller-owned K outputs but writes the ring in the fused prepare kernel, then
skips the backend scatter. Verify tests run the real prepare and backend control
flow with CPU substitutes for the projection, fused kernel, and attention reader.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.kernels.ops.attention import fused_qk_norm_rope_store as fused_store
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import env_gate, runtime
from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# deliberately != head_dim below: the row width has to come off the pool, since
# that is the stride the kernel reads Q with. Sharing head_dim's value would let
# a regression that reads self.head_dim pass.
NOPE_ROW_BYTES = 16
ROPE_DIM = 2
HEAD_DIM = 8
N_LOCAL_HEADS = 16
TOKENS = 3


class _RecordingBackend:
    def __init__(self, pool):
        self.calls = []
        self.token_to_kv_pool = pool
        self.softmax_scale = 512**-0.5
        self.speculative_num_steps = 1
        # Two requests, with two verify tokens in one and one in the other.
        # Positions 7, 8 cross the first request's ring boundary.
        unified = SimpleNamespace(
            swa_loc=torch.tensor([15, 8, 19], dtype=torch.int32),
            verify_store_state_slot=torch.tensor([1, 1, 2], dtype=torch.int32),
            swa_indices=torch.tensor([15, 15, 8, 19], dtype=torch.int32),
            swa_indptr=torch.tensor([0, 1, 3, 4], dtype=torch.int32),
        )
        self.forward_metadata = SimpleNamespace(
            core_attn_metadata=SimpleNamespace(unified=unified)
        )

    get_unified_swa_loc = DeepseekV4HipRadixBackend.get_unified_swa_loc

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["forward_batch"].forward_mode.is_target_verify():
            args = {
                key: value for key, value in kwargs.items() if key not in ("k", "v")
            }
            return DeepseekV4HipRadixBackend._forward_unified_kv(
                self,
                kv=kwargs["k"],
                core_attn_metadata=self.forward_metadata.core_attn_metadata,
                **args,
            )
        query = kwargs["q"]
        # bf16 regardless of the q layout -- attention output is never fp8
        return torch.zeros(
            query.shape[0], query.shape[1], ROPE_DIM, dtype=torch.bfloat16
        )


class _Pool:
    def __init__(self, fp8):
        rows = 32
        self.unified_swa_window = 6
        self.unified_swa_ring_size = 8
        self.unified_swa_pages = rows
        self.nope = torch.zeros(
            rows, NOPE_ROW_BYTES, dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16
        )
        self.rope = torch.zeros(rows, ROPE_DIM, dtype=torch.bfloat16)

    def get_unified_kv(self, layer_id):
        return self.nope

    def get_unified_kv_rope(self, layer_id):
        return self.rope


class _Harness(deepseek_v4.MQALayer):
    def __init__(self, rank=3):
        torch.nn.Module.__init__(self)
        self.layer_id = 0
        self.attn_tp_rank = rank
        self.attn_tp_size = 8
        self.n_heads = 128
        self.n_local_heads = N_LOCAL_HEADS
        self.head_dim = HEAD_DIM
        self.n_local_groups = 1
        self.o_lora_rank = 3
        self.qk_rope_head_dim = ROPE_DIM
        self.freqs_cis = torch.empty(0)
        self.compress_ratio = 4
        self.attn_mqa = SimpleNamespace(layer_id=0, v_head_dim=ROPE_DIM)
        self.attn_sink = torch.nn.Parameter(torch.arange(128, dtype=torch.float32))
        self._attn_sink_local = None
        self.alt_streams = None
        self.dsa_enable_prefill_cp = False
        self.use_fused_wo_a = False
        self.use_npu_arch35_mxfp8_wo_a = False
        self.compressor = object()
        self.wo_a = SimpleNamespace(
            weight=torch.ones(
                self.n_local_groups,
                self.o_lora_rank,
                self.n_local_heads * ROPE_DIM,
                dtype=torch.bfloat16,
            )
        )
        self.wo_b = lambda value, skip_all_reduce=False: (value, None)
        self.prepare_kwargs = None
        self.kernel_kwargs = None
        self.fuse_wqa_wkv = True
        self.q_lora_rank = 4
        self.eps = 1e-6
        self.cos_cache = self.sin_cache = torch.empty(0)
        self.kv_norm = SimpleNamespace(weight=torch.ones(HEAD_DIM))
        self.wqkv_a = lambda value: (
            value.new_ones(value.shape[0], self.q_lora_rank + HEAD_DIM),
            None,
        )
        self.wq_b = lambda value: (
            value.new_ones(value.shape[0], N_LOCAL_HEADS * HEAD_DIM),
            None,
        )

    def _normalize_q_lora(self, q):
        return q, q

    def _fake_fused_store(self, **kwargs):
        # Numerics are covered by the GPU kernel tests. Distinct values expose
        # missing stores, swapped halves, or use of the temporary pair as Q.
        self.kernel_kwargs = kwargs
        kwargs["q_out"].zero_()
        if kwargs["q_rope_out"] is not None:
            kwargs["q_rope_out"].zero_()
        rows = kwargs["swa_loc"].long()
        pool = kwargs["swa_cache"]
        if kwargs["fp8_2buff"]:
            kwargs["k_nope_out"].view(torch.uint8).fill_(7)
            kwargs["k_rope_out"].fill_(3)
            pool.view(torch.uint8)[rows] = kwargs["k_nope_out"].view(torch.uint8)
            kwargs["swa_rope_cache"][rows] = kwargs["k_rope_out"]
        else:
            pool[rows] = 5
        return kwargs["q_out"]

    def _forward_prepare(
        self,
        x,
        positions,
        forward_batch,
        attn_backend,
        q_out=None,
        x_quant=None,
        q_rope_out=None,
        k_nope_out=None,
        k_rope_out=None,
    ):
        self.prepare_kwargs = dict(
            q_out=q_out,
            q_rope_out=q_rope_out,
            k_nope_out=k_nope_out,
            k_rope_out=k_rope_out,
        )
        if forward_batch.forward_mode.is_target_verify():
            return super()._forward_prepare(
                x,
                positions,
                forward_batch,
                attn_backend,
                x_quant=x_quant,
                **self.prepare_kwargs,
            )
        q_out.zero_()
        # mirrors the prefill arm: the packed nope half leaves on the kv slot,
        # which is what turns save_kv_cache on in the caller
        return q_out, k_nope_out


def _run(fp8, mode=ForwardMode.DECODE, cp=False, fused_verify=True):
    layer = _Harness()
    layer.dsa_enable_prefill_cp = cp
    pool = _Pool(fp8)
    backend = _RecordingBackend(pool)
    positions = (
        torch.tensor([7, 8, 3]) if mode.is_target_verify() else torch.arange(TOKENS)
    )
    forward_batch = SimpleNamespace(
        forward_mode=mode,
        positions=positions,
        req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
    )
    if mode.is_target_verify():
        layer.compress_ratio = 0
        layer.compressor = layer.indexer = None

    def reader(**kwargs):
        return torch.zeros(TOKENS, N_LOCAL_HEADS, ROPE_DIM, dtype=torch.bfloat16)

    with (
        envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.override(False),
        envs.SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY.override(fused_verify),
        patch.object(env_gate, "is_unified_kv_triton", return_value=True),
        patch.object(env_gate, "is_unified_kv_fp8", return_value=fp8),
        patch.object(deepseek_v4, "get_token_to_kv_pool", return_value=pool),
        patch.object(
            deepseek_v4,
            "get_attn_tp_context",
            return_value=SimpleNamespace(input_scattered=True),
        ),
        patch.object(
            deepseek_v4, "get_parallel", return_value=SimpleNamespace(tp_size=8)
        ),
        patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
        patch.object(deepseek_v4, "dsa_use_prefill_cp", return_value=cp),
        patch.object(deepseek_v4, "fused_rope_inplace", return_value=None),
        patch.object(deepseek_v4, "_FP8_WO_A_GEMM", False),
        patch.object(deepseek_v4, "_is_gfx942_supported", False),
        patch.object(deepseek_v4, "_is_gfx95_supported", False),
        patch.object(deepseek_v4, "_is_gfx1250_supported", False),
        patch.object(
            fused_store, "fused_qk_norm_rope_swa_store", layer._fake_fused_store
        ),
        patch.object(runtime, "decode_fp8_2buff", side_effect=reader) as fp8_reader,
        patch.object(runtime, "decode", side_effect=reader),
        patch.object(runtime, "store_swa_into_unified") as scatter,
        patch.object(deepseek_v4, "_is_hip", True),
        patch.object(deepseek_v4, "_is_npu", False),
    ):
        layer.forward(
            torch.zeros(TOKENS, 4, dtype=torch.bfloat16),
            positions,
            forward_batch,
        )
        if mode.is_target_verify():
            scatter.assert_not_called()
            if fp8:
                fp8_reader.assert_called_once()
                assert fp8_reader.call_args.kwargs["unified_kv"] is pool.nope
                assert fp8_reader.call_args.kwargs["unified_kv_rope"] is pool.rope

    return layer, backend.calls[0]


class TestUnifiedFp8QPair(unittest.TestCase):
    def test_fp8_decode_hands_the_backend_a_packed_pair(self):
        layer, call = _run(fp8=True)

        q, q_rope = call["q"], call["q_rope"]
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        # width off the pool, not off head_dim
        self.assertEqual(tuple(q.shape), (TOKENS, N_LOCAL_HEADS, NOPE_ROW_BYTES))
        self.assertEqual(tuple(q_rope.shape), (TOKENS, N_LOCAL_HEADS, ROPE_DIM))
        self.assertEqual(q_rope.dtype, torch.bfloat16)
        # the asm kernel walks both as flat buffers, no stride arguments
        self.assertTrue(q.is_contiguous())
        self.assertTrue(q_rope.is_contiguous())
        # same pair reached the store, or nothing would have written them
        self.assertIs(layer.prepare_kwargs["q_out"], q)
        self.assertIs(layer.prepare_kwargs["q_rope_out"], q_rope)

    def test_bf16_decode_still_gets_one_plain_tensor(self):
        layer, call = _run(fp8=False)

        # q_rope absent is what routes the backend back to the Triton reader
        self.assertNotIn("q_rope", call)
        self.assertIsNone(layer.prepare_kwargs["q_rope_out"])
        self.assertEqual(call["q"].dtype, torch.bfloat16)
        self.assertEqual(tuple(call["q"].shape), (TOKENS, N_LOCAL_HEADS, HEAD_DIM))

    def test_fp8_prefill_also_gets_a_packed_k_pair(self):
        layer, call = _run(fp8=True, mode=ForwardMode.EXTEND)

        k, k_rope = call["k"], call["k_rope"]
        self.assertEqual(k.dtype, torch.float8_e4m3fn)
        # one row per token, width off the pool like Q
        self.assertEqual(tuple(k.shape), (TOKENS, NOPE_ROW_BYTES))
        self.assertEqual(tuple(k_rope.shape), (TOKENS, ROPE_DIM))
        self.assertEqual(k_rope.dtype, torch.bfloat16)
        self.assertTrue(k.is_contiguous())
        self.assertTrue(k_rope.is_contiguous())
        # the buffers the fused store filled are the ones attention reads, and
        # the ring write after it consumes the same rows
        self.assertIs(layer.prepare_kwargs["k_nope_out"], k)
        self.assertIs(layer.prepare_kwargs["k_rope_out"], k_rope)
        self.assertTrue(call["save_kv_cache"])
        # Q is packed here too, that is what picks the fp8 prefill kernel
        self.assertEqual(call["q"].dtype, torch.float8_e4m3fn)
        self.assertIsNotNone(call["q_rope"])

    def test_fp8_decode_gets_no_k_pair(self):
        """decode attends over rows the ring already holds, so it has no extend"""
        layer, call = _run(fp8=True, mode=ForwardMode.DECODE)

        self.assertNotIn("k_rope", call)
        self.assertIsNone(layer.prepare_kwargs["k_nope_out"])
        self.assertIsNone(layer.prepare_kwargs["k_rope_out"])

    def test_bf16_prefill_keeps_one_plain_tensor(self):
        layer, call = _run(fp8=False, mode=ForwardMode.EXTEND)

        self.assertNotIn("q_rope", call)
        self.assertNotIn("k_rope", call)
        self.assertIsNone(layer.prepare_kwargs["k_nope_out"])
        self.assertEqual(call["q"].dtype, torch.bfloat16)

    def test_fp8_target_verify_writes_ring_without_backend_scatter(self):
        layer, call = _run(fp8=True, mode=ForwardMode.TARGET_VERIFY)

        self.assertEqual(call["q"].dtype, torch.float8_e4m3fn)
        self.assertIsNotNone(call["q_rope"])
        k = layer.prepare_kwargs["k_nope_out"]
        k_rope = layer.prepare_kwargs["k_rope_out"]
        self.assertEqual(tuple(k.shape), (TOKENS, NOPE_ROW_BYTES))
        self.assertEqual(tuple(k_rope.shape), (TOKENS, ROPE_DIM))
        self.assertIs(layer.kernel_kwargs["k_nope_out"], k)
        self.assertIs(layer.kernel_kwargs["k_rope_out"], k_rope)
        # Retaining the dense outputs must not turn the backend store back on.
        self.assertFalse(call["save_kv_cache"])
        self.assertIs(call["k"], call["q"])
        self.assertIs(call["v"], call["q"])
        self.assertIs(call["k_rope"], k_rope)
        expected_nope = torch.zeros(32, NOPE_ROW_BYTES, dtype=torch.uint8)
        expected_rope = torch.zeros(32, ROPE_DIM, dtype=torch.bfloat16)
        expected_nope[[15, 8, 19]] = 7
        expected_rope[[15, 8, 19]] = 3
        torch.testing.assert_close(
            layer.kernel_kwargs["swa_cache"].view(torch.uint8), expected_nope
        )
        torch.testing.assert_close(layer.kernel_kwargs["swa_rope_cache"], expected_rope)

    def test_fp8_target_verify_needs_the_fused_store(self):
        """nothing else packs the pair, so the unfused arm would hand over bf16"""
        with self.assertRaisesRegex(
            NotImplementedError, "SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY"
        ):
            _run(fp8=True, mode=ForwardMode.TARGET_VERIFY, fused_verify=False)

    def test_bf16_target_verify_is_left_alone(self):
        """the packing is fp8-only; bf16 verify keeps working as it always did"""
        layer, call = _run(fp8=False, mode=ForwardMode.TARGET_VERIFY)

        self.assertNotIn("q_rope", call)
        self.assertNotIn("k_rope", call)
        self.assertIsNone(layer.prepare_kwargs["k_nope_out"])
        self.assertFalse(call["save_kv_cache"])
        expected = torch.zeros(32, NOPE_ROW_BYTES, dtype=torch.bfloat16)
        expected[[15, 8, 19]] = 5
        torch.testing.assert_close(layer.kernel_kwargs["swa_cache"], expected)

    def test_fp8_prefill_cp_is_refused_with_a_reason(self):
        """the gather hands kv back in global token order after norm+RoPE, so
        packing would have to move ahead of it -- refuse rather than guess"""
        with self.assertRaisesRegex(NotImplementedError, "cp_size"):
            _run(fp8=True, mode=ForwardMode.EXTEND, cp=True)

    def test_bf16_prefill_cp_is_left_alone(self):
        """the refusal is fp8-only, CP prefill without it keeps working"""
        _, call = _run(fp8=False, mode=ForwardMode.EXTEND, cp=True)

        self.assertNotIn("q_rope", call)
        self.assertNotIn("k_rope", call)

    def test_fp8_decode_under_cp_is_not_refused(self):
        """only prefill packs this chunk; decode reads rows the ring already has"""
        _, call = _run(fp8=True, mode=ForwardMode.DECODE, cp=True)

        self.assertEqual(call["q"].dtype, torch.float8_e4m3fn)

    def test_sink_is_sliced_to_this_rank(self):
        _, call = _run(fp8=True)

        sink = call["attn_sink"]
        self.assertEqual(tuple(sink.shape), (N_LOCAL_HEADS,))
        torch.testing.assert_close(
            sink, torch.arange(3 * N_LOCAL_HEADS, 4 * N_LOCAL_HEADS).float()
        )


if __name__ == "__main__":
    unittest.main()
