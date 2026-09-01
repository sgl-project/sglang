"""DeepSeek-V4 unified_kv fp8: the packed pairs handed to the two readers.

Decode only needs Q packed -- its K is already in the ring. Prefill is a KV
source of its own, so it gets a packed K pair beside the Q one, and the same
buffers have to reach both attention and the ring write after it. Verify wants
both halves: it reads the ring the way decode does and fills it the way prefill
does, only the write lands before attention instead of after.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import env_gate
from sglang.srt.environ import envs
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
    def __init__(self):
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        query = kwargs["q"]
        # bf16 regardless of the q layout -- attention output is never fp8
        return torch.zeros(
            query.shape[0], query.shape[1], ROPE_DIM, dtype=torch.bfloat16
        )


class _Pool:
    def __init__(self, fp8):
        rows = 32
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
        self.compressor = object()
        self.wo_a = SimpleNamespace(
            weight=torch.ones(
                self.n_local_groups,
                self.o_lora_rank,
                self.n_local_heads * ROPE_DIM,
                dtype=torch.bfloat16,
            )
        )
        self.wo_b = lambda value: (value, None)
        self.prepare_kwargs = None

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
        q_out.zero_()
        # mirrors the prefill arm: the packed nope half leaves on the kv slot,
        # which is what turns save_kv_cache on in the caller
        return q_out, k_nope_out


def _run(fp8, mode=ForwardMode.DECODE, cp=False, fused_verify=True):
    layer = _Harness()
    layer.dsa_enable_prefill_cp = cp
    backend = _RecordingBackend()
    forward_batch = SimpleNamespace(forward_mode=mode)

    with (
        envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.override(False),
        envs.SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY.override(fused_verify),
        patch.object(env_gate, "is_unified_kv_triton", return_value=True),
        patch.object(env_gate, "is_unified_kv_fp8", return_value=fp8),
        patch.object(deepseek_v4, "get_token_to_kv_pool", return_value=_Pool(fp8)),
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
        patch.object(deepseek_v4, "_is_hip", True),
        patch.object(deepseek_v4, "_is_npu", False),
    ):
        layer.forward(
            torch.zeros(TOKENS, 4, dtype=torch.bfloat16),
            torch.arange(TOKENS),
            forward_batch,
        )

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

    def test_fp8_target_verify_gets_the_packed_pair(self):
        """verify reads the ring like decode, but it also feeds it like prefill"""
        layer, call = _run(fp8=True, mode=ForwardMode.TARGET_VERIFY)

        # packed Q is what picks the decode reader over the Triton one
        self.assertEqual(call["q"].dtype, torch.float8_e4m3fn)
        self.assertIsNotNone(call["q_rope"])
        k, k_rope = call["k"], call["k_rope"]
        self.assertEqual(k.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(k.shape), (TOKENS, NOPE_ROW_BYTES))
        self.assertEqual(tuple(k_rope.shape), (TOKENS, ROPE_DIM))
        self.assertIs(layer.prepare_kwargs["k_nope_out"], k)
        self.assertIs(layer.prepare_kwargs["k_rope_out"], k_rope)
        # unlike prefill the ring write happens before attention, but it is the
        # same flag and the same pair
        self.assertTrue(call["save_kv_cache"])

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
