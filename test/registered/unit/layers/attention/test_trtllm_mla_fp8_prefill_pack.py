"""trtllm_mla builds the fp8 Q/K/V of MLA chunked prefill in fewer launches.

With an fp8 KV cache, the ragged prefill kernel reads fp8 Q/K/V. The stock path
built K = [k_nope | k_pe] in bf16, then cast K and V to fp8 with strided copies
in every ragged launch, and re-cast the same bf16 Q once per launch (the causal
pass and every prefix chunk). The backend now implements forward_mha's hooks:

- ``pack_extend_kv`` / ``pack_prefix_chunk_kv``: one ``mla_kv_pack_quantize_fp8``
  launch writes contiguous unit-scale fp8 K and V;
- ``prepare_chunked_prefill_qkv``: Q is cast once per layer.

Covered here:
- hook gating: hidden without fp8 KV, inactive under the flashinfer fallback;
- ``_quantize_fp8_qkv`` passes packed fp8 K/V through at unit scale;
- forward_normal_prepare returns the packed K/V when the backend packs;
- bit-exactness of pack + ``_quantize_fp8_qkv`` vs the stock concat + cast, and
  of the ragged prefill output (CUDA; the ragged kernel needs SM100).

    python3 test/registered/unit/layers/attention/test_trtllm_mla_fp8_prefill_pack.py
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLAPrefillMetadata,
    _quantize_fp8_qkv,
)
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha
from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="4-gpu-b200")

FP8 = torch.float8_e4m3fn
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
KV_LORA_RANK = 512


def _is_sm100() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and is_flashinfer_available()
    )


def _backend(fallback_to_flashinfer_impl=None) -> TRTLLMMLABackend:
    """A backend with only the prefill state the hooks read; None = no
    extend metadata (e.g. after a verify forward)."""
    backend = TRTLLMMLABackend.__new__(TRTLLMMLABackend)
    backend.data_type = FP8
    backend.forward_prefill_metadata = (
        None
        if fallback_to_flashinfer_impl is None
        else TRTLLMMLAPrefillMetadata(
            max_seq_len=4,
            cum_seq_lens=torch.tensor([0, 4], dtype=torch.int32),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            fallback_to_flashinfer_impl=fallback_to_flashinfer_impl,
        )
    )
    return backend


def _layer(k_scale_float=None, v_scale_float=None):
    return SimpleNamespace(
        k_scale_float=k_scale_float,
        v_scale_float=v_scale_float,
        scaling=(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5,
    )


def _mla_inputs(num_tokens, num_heads, device, seed=0):
    """Strided k_nope / v views of a kv_b_proj output and the k_pe view of the
    latent cache row, as forward_normal_prepare produces them."""
    gen = torch.Generator(device=device).manual_seed(seed)
    kv = torch.randn(
        num_tokens,
        num_heads,
        QK_NOPE_HEAD_DIM + V_HEAD_DIM,
        generator=gen,
        device=device,
    ).to(torch.bfloat16)
    latent = torch.randn(
        num_tokens, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM, generator=gen, device=device
    ).to(torch.bfloat16)
    q = torch.randn(
        num_tokens,
        num_heads,
        QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
        generator=gen,
        device=device,
    ).to(torch.bfloat16)
    k_nope = kv[..., :QK_NOPE_HEAD_DIM]
    v = kv[..., QK_NOPE_HEAD_DIM:]
    k_pe = latent[..., KV_LORA_RANK:]
    return q, k_nope, k_pe, v


def _stock_k(k_nope, k_pe):
    """The stock bf16 K = [k_nope | broadcast k_pe]."""
    return torch.cat([k_nope, k_pe.expand(-1, k_nope.shape[1], -1)], dim=-1)


class TestHookGating(CustomTestCase):
    def test_runs_only_on_the_ragged_fp8_kernel(self):
        self.assertTrue(_backend(False)._runs_ragged_fp8_prefill())
        self.assertFalse(_backend(True)._runs_ragged_fp8_prefill())
        self.assertFalse(_backend(None)._runs_ragged_fp8_prefill())

    def test_flashinfer_fallback_keeps_bf16(self):
        # The flashinfer fallback reads bf16 Q/K/V: no packing, no Q cast.
        backend = _backend(True)
        q, k_nope, k_pe, v = _mla_inputs(4, 2, "cpu")
        self.assertIsNone(backend.pack_extend_kv(k_nope=k_nope, k_pe=k_pe, v=v))
        q_out, k_out, v_out = backend.prepare_chunked_prefill_qkv(q, k_nope, v, None)
        self.assertIs(q_out, q)
        self.assertIs(k_out, k_nope)
        self.assertIs(v_out, v)

    def test_q_cast_once(self):
        backend = _backend(False)
        q, k_nope, _, v = _mla_inputs(4, 2, "cpu")
        q_out, k_out, v_out = backend.prepare_chunked_prefill_qkv(q, k_nope, v, None)
        self.assertEqual(q_out.dtype, FP8)
        torch.testing.assert_close(q_out.float(), q.to(FP8).float(), rtol=0, atol=0)
        self.assertIs(k_out, k_nope)
        self.assertIs(v_out, v)
        # A pre-cast Q is passed through untouched by every ragged launch.
        q_again, _, _, k_scale, v_scale = _quantize_fp8_qkv(
            q_out, k_out.contiguous(), v_out.contiguous(), _layer()
        )
        self.assertEqual(q_again.data_ptr(), q_out.data_ptr())
        self.assertEqual((k_scale, v_scale), (1.0, 1.0))


class TestQuantizePackedKV(CustomTestCase):
    def test_packed_kv_pass_through_at_unit_scale(self):
        q, k_nope, k_pe, v = _mla_inputs(4, 2, "cpu")
        k_fp8 = _stock_k(k_nope, k_pe).to(FP8)
        v_fp8 = v.contiguous().to(FP8)
        # Layer scales do not apply to K/V already packed at unit scale.
        q_out, k_out, v_out, k_scale, v_scale = _quantize_fp8_qkv(
            q, k_fp8, v_fp8, _layer(k_scale_float=2.0, v_scale_float=4.0)
        )
        self.assertEqual(q_out.dtype, FP8)
        self.assertEqual(k_out.data_ptr(), k_fp8.data_ptr())
        self.assertEqual(v_out.data_ptr(), v_fp8.data_ptr())
        self.assertEqual((k_scale, v_scale), (1.0, 1.0))

    def test_bf16_kv_unchanged(self):
        q, k_nope, k_pe, v = _mla_inputs(4, 2, "cpu")
        k = _stock_k(k_nope, k_pe)
        _, k_out, v_out, k_scale, v_scale = _quantize_fp8_qkv(q, k, v, _layer())
        self.assertEqual((k_scale, v_scale), (1.0, 1.0))
        torch.testing.assert_close(k_out.float(), k.to(FP8).float(), rtol=0, atol=0)
        torch.testing.assert_close(v_out.float(), v.to(FP8).float(), rtol=0, atol=0)


class _FakeAttention:
    """The slice of DeepseekV2AttentionMLA that forward_normal_prepare reads on
    the non-DSA, no-q_lora, no-RoPE path."""

    num_local_heads = 2
    qk_nope_head_dim = QK_NOPE_HEAD_DIM
    qk_rope_head_dim = QK_ROPE_HEAD_DIM
    qk_head_dim = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
    v_head_dim = V_HEAD_DIM
    kv_lora_rank = KV_LORA_RANK
    q_lora_rank = None
    use_dsa = False
    rotary_emb = None

    def __init__(self, num_tokens):
        _, self.k_nope, self.k_pe, self.v = _mla_inputs(
            num_tokens, self.num_local_heads, "cpu"
        )
        self.concat_calls = 0
        heads = self.num_local_heads
        self.q_proj = lambda x: (torch.zeros(x.shape[0], heads * self.qk_head_dim),)
        self.kv_a_proj_with_mqa = lambda x: (
            torch.cat([torch.zeros(x.shape[0], KV_LORA_RANK), self.k_pe[:, 0]], -1),
        )
        self.kv_a_layernorm = lambda x: x
        kv = torch.cat([self.k_nope, self.v], dim=-1)
        self.kv_b_proj = lambda x: (kv.reshape(x.shape[0], -1),)

    def _set_mla_kv_buffer(self, *args):
        pass

    def _concat_and_cast_mha_k(self, k_nope, k_pe, forward_batch):
        self.concat_calls += 1
        return _stock_k(k_nope, k_pe)


class TestForwardNormalPreparePack(CustomTestCase):
    def _prepare(self, backend):
        attn = _FakeAttention(num_tokens=3)
        forward_batch = SimpleNamespace(mha_one_shot=False)
        with mock.patch.object(
            forward_mha, "resolve_attn_backend", return_value=backend
        ):
            _, k, v, _ = forward_mha.DeepseekMHAForwardMixin.forward_normal_prepare(
                attn,
                positions=torch.arange(3),
                hidden_states=torch.zeros(3, 8),
                forward_batch=forward_batch,
                zero_allocator=None,
            )
        return attn, k, v

    def test_uses_backend_pack(self):
        packed = (torch.empty(0), torch.empty(0))
        calls = []

        def pack_extend_kv(*, k_nope, k_pe, v):
            calls.append((k_nope.shape, k_pe.shape, v.shape))
            return packed

        attn, k, v = self._prepare(SimpleNamespace(pack_extend_kv=pack_extend_kv))
        self.assertIs(k, packed[0])
        self.assertIs(v, packed[1])
        self.assertEqual(attn.concat_calls, 0)
        self.assertEqual(
            calls,
            [
                (
                    torch.Size([3, 2, QK_NOPE_HEAD_DIM]),
                    torch.Size([3, 1, QK_ROPE_HEAD_DIM]),
                    torch.Size([3, 2, V_HEAD_DIM]),
                )
            ],
        )

    def test_falls_back_to_concat(self):
        for backend in (
            SimpleNamespace(),  # backend without the hook
            SimpleNamespace(pack_extend_kv=None),  # hook hidden (bf16 KV)
            SimpleNamespace(pack_extend_kv=lambda **_: None),  # declined
        ):
            with self.subTest(backend=backend):
                attn, k, v = self._prepare(backend)
                self.assertEqual(attn.concat_calls, 1)
                self.assertTrue(torch.equal(k, _stock_k(attn.k_nope, attn.k_pe)))
                self.assertTrue(torch.equal(v, attn.v))


@unittest.skipUnless(torch.cuda.is_available(), "the pack kernel is Triton (CUDA)")
class TestPackBitExact(CustomTestCase):
    """pack + _quantize_fp8_qkv == stock concat + _quantize_fp8_qkv, bit for
    bit, over head counts (12 = Kimi-K3 / TP8, 16 = DeepSeek-V2-Lite, 128 =
    DeepSeek-V3 / TP1) and token counts."""

    def test_bit_exact(self):
        backend = _backend(False)
        for num_heads in (12, 16, 128):
            for num_tokens in (1, 17, 1024, 16384):
                with self.subTest(num_heads=num_heads, num_tokens=num_tokens):
                    q, k_nope, k_pe, v = _mla_inputs(num_tokens, num_heads, "cuda")
                    ref = _quantize_fp8_qkv(q, _stock_k(k_nope, k_pe), v, _layer())
                    for k, v_packed in (
                        backend.pack_extend_kv(k_nope=k_nope, k_pe=k_pe, v=v),
                        backend.pack_prefix_chunk_kv(k_nope, k_pe, v),
                    ):
                        self.assertTrue(k.is_contiguous() and v_packed.is_contiguous())
                        out = _quantize_fp8_qkv(q, k, v_packed, _layer())
                        self.assertEqual(out[3:], ref[3:])
                        for got, want in zip(out[:3], ref[:3]):
                            self.assertEqual(got.dtype, FP8)
                            self.assertTrue(
                                torch.equal(
                                    got.view(torch.uint8), want.view(torch.uint8)
                                )
                            )


@unittest.skipUnless(_is_sm100(), "the trtllm-gen ragged prefill needs SM100")
class TestRaggedPrefillBitExact(CustomTestCase):
    """A causal pass plus one prefix pass through _run_prefill_kernel give the
    same output and LSE with packed fp8 inputs and a pre-cast Q as with the
    stock bf16 inputs."""

    NUM_HEADS = 16
    SEQ_LENS = [1, 300, 2049]
    # Every row has prefix KV: zero-KV rows need fixup_zero_kv_rows in both arms.
    PREFIX_LENS = [70, 1, 4096]

    def _run(self, backend, layer, q, k, v, q_lens, kv_lens, is_causal):
        to_cu = lambda lens: torch.nn.functional.pad(  # noqa: E731
            torch.tensor(lens, device="cuda").cumsum(0), (1, 0)
        ).int()
        out = torch.empty(
            q.shape[0], self.NUM_HEADS, V_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        return backend._run_prefill_kernel(
            q=q,
            k=k,
            v=v,
            layer=layer,
            batch_size=len(q_lens),
            cum_seq_lens_q=to_cu(q_lens),
            max_q_len=max(q_lens),
            seq_lens_kv=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
            cum_seq_lens_kv=to_cu(kv_lens),
            max_kv_len=max(kv_lens),
            is_causal=is_causal,
            return_lse=True,
            out_buffer=out,
            o_sf_scale=1.0 if is_causal else -1.0,
        )

    def test_causal_and_prefix_pass(self):
        backend = _backend(False)
        backend.workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.int8, device="cuda"
        )
        layer = _layer()
        num_q, num_kv = sum(self.SEQ_LENS), sum(self.PREFIX_LENS)
        q, k_nope, k_pe, v = _mla_inputs(num_q, self.NUM_HEADS, "cuda", seed=1)
        _, pk_nope, pk_pe, pv = _mla_inputs(num_kv, self.NUM_HEADS, "cuda", seed=2)
        q_fp8, _, _ = backend.prepare_chunked_prefill_qkv(q, None, None, None)
        cases = (
            # (stock q/k/v, packed q/k/v, kv lens, causal)
            (
                (q, _stock_k(k_nope, k_pe), v),
                (q_fp8, *backend.pack_extend_kv(k_nope=k_nope, k_pe=k_pe, v=v)),
                self.SEQ_LENS,
                True,
            ),
            (
                (q, _stock_k(pk_nope, pk_pe), pv),
                (q_fp8, *backend.pack_prefix_chunk_kv(pk_nope, pk_pe, pv)),
                self.PREFIX_LENS,
                False,
            ),
        )
        for stock, packed, kv_lens, is_causal in cases:
            with self.subTest(is_causal=is_causal):
                ref_out, ref_lse = self._run(
                    backend, layer, *stock, self.SEQ_LENS, kv_lens, is_causal
                )
                out, lse = self._run(
                    backend, layer, *packed, self.SEQ_LENS, kv_lens, is_causal
                )
                self.assertTrue(torch.equal(out, ref_out))
                self.assertTrue(torch.equal(lse, ref_lse))


if __name__ == "__main__":
    unittest.main()
