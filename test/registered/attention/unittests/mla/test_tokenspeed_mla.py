import importlib.util
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    MLAAttentionCase,
    run_mla_attention_case,
)
from sglang.test.test_utils import CustomTestCase

# tokenspeed_mla is a CuTe DSL backend for Blackwell (SM100). It additionally
# enforces:
#   - kv_cache_dtype == torch.float8_e4m3fn (kv_cache_dtype=fp8_e4m3)
#   - page_size in {32, 64}
# See python/sglang/srt/layers/attention/tokenspeed_mla_backend.py and
# is_tokenspeed_mla_available() in python/sglang/srt/utils/common.py.
#
# The shared MLAAttentionCase fixture now supports `fp8_kv_cache=True`:
# `MockMLAModelRunner` decouples `kv_cache_dtype` from the model `dtype`
# and routes K writes through the FP8 quantize path. The reference still
# computes against BF16 K (independent of the cache bytes) and tolerates
# FP8 quant noise via a looser tolerance.
_MIN_SM = 100


def _supported() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is required"
    if importlib.util.find_spec("tokenspeed_mla") is None:
        return False, "tokenspeed_mla python package is not installed"
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm < _MIN_SM:
        return (
            False,
            f"tokenspeed_mla requires SM {_MIN_SM // 10}.{_MIN_SM % 10}+ (Blackwell), "
            f"got SM {major}.{minor}",
        )
    return True, ""


_SUPPORTED, _SKIP_REASON = _supported()


MLA_SHAPE_KWARGS = dict(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    hidden_size=1024,
    max_context_len=256,
)


from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)
class TestTokenspeedMLAAttentionBackendCorrectness(CustomTestCase):
    # tokenspeed_mla allows page_size in {32, 64} (server_args.py:2809-2813)
    # and requires kv_cache_dtype==fp8_e4m3 (server_args.py:2814-2818).
    # Cover both page sizes, with extend + decode + ragged + page-boundary.
    CASES = (
        # ----- page_size=64 -----
        MLAAttentionCase(
            name="mla_extend_tokenspeed_zero_prefix_exact_page_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(64,),
        ),
        MLAAttentionCase(
            name="mla_extend_tokenspeed_zero_prefix_below_page_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(63,),
        ),
        MLAAttentionCase(
            name="mla_extend_tokenspeed_zero_prefix_above_page_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(65,),
        ),
        MLAAttentionCase(
            name="mla_extend_tokenspeed_prefix_exact_page_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(64,),
            extend_lens=(4,),
        ),
        MLAAttentionCase(
            name="mla_extend_tokenspeed_cross_page_boundary_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(63,),
            extend_lens=(2,),
        ),
        MLAAttentionCase(
            name="mla_extend_tokenspeed_ragged_page_boundary_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0, 32, 64),
            extend_lens=(63, 32, 1),
        ),
        MLAAttentionCase(
            name="mla_decode_tokenspeed_page_boundary_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=64,
            prefix_lens=(62, 63, 64),
        ),
        MLAAttentionCase(
            name="mla_decode_tokenspeed_bsz1_nonzero_prefix_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=64,
            prefix_lens=(31,),
        ),
        # ----- page_size=32 -----
        MLAAttentionCase(
            name="mla_extend_tokenspeed_zero_prefix_exact_page_32",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=32,
            prefix_lens=(0,),
            extend_lens=(32,),
        ),
        MLAAttentionCase(
            name="mla_extend_tokenspeed_cross_page_boundary_32",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=32,
            prefix_lens=(31,),
            extend_lens=(2,),
        ),
        MLAAttentionCase(
            name="mla_decode_tokenspeed_page_boundary_32",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=32,
            prefix_lens=(30, 31, 32),
        ),
    )

    def test_projected_mla_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                # Looser tolerance to absorb FP8 quant noise (the reference
                # reads BF16 K independent of the FP8 cache, so per-element
                # drift from the BF16->FP8 cast accumulates through the
                # attention reduction).
                run_mla_attention_case(
                    self,
                    case,
                    fp8_kv_cache=True,
                    atol=2e-1,
                    rtol=2e-1,
                    **MLA_SHAPE_KWARGS,
                )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTokenspeedMLANoPEPrefill(CustomTestCase):
    """prepare_prefill_qkv with rotary_emb=None (skip_rope, e.g. Kimi Linear)."""

    T = 37  # not a multiple of any kernel block size
    NUM_HEADS = 4
    QK_NOPE_HEAD_DIM = 128
    QK_ROPE_HEAD_DIM = 64
    V_HEAD_DIM = 128
    KV_LORA_RANK = 512

    def _assert_fp8_equal(self, out: torch.Tensor, ref_bf16: torch.Tensor):
        fp8 = torch.float8_e4m3fn
        self.assertEqual(out.dtype, fp8)
        self.assertEqual(out.shape, ref_bf16.shape)
        self.assertTrue(
            torch.equal(out.view(torch.uint8), ref_bf16.to(fp8).view(torch.uint8))
        )

    def test_nope_prefill_skips_rope_and_packs_fp8(self):
        device = torch.device("cuda")
        bf16 = torch.bfloat16
        T, H = self.T, self.NUM_HEADS
        nope, rope = self.QK_NOPE_HEAD_DIM, self.QK_ROPE_HEAD_DIM
        v_dim, lora = self.V_HEAD_DIM, self.KV_LORA_RANK
        torch.manual_seed(0)

        # Same shapes and views forward_normal_prepare passes to the hook.
        q = torch.randn(T, H, nope + rope, dtype=bf16, device=device)
        q_pe = q[..., nope:]
        latent_cache = torch.randn(T, 1, lora + rope, dtype=bf16, device=device)
        kv_a = latent_cache[:, 0, :lora].contiguous()
        k_pe = latent_cache[:, :, lora:]
        kv_b_weight = 0.05 * torch.randn(
            H * (nope + v_dim), lora, dtype=bf16, device=device
        )
        positions = torch.arange(T, device=device)
        out_cache_loc = torch.randperm(4 * T, device=device)[:T]

        layer = SimpleNamespace(
            kv_b_proj=lambda x: (x @ kv_b_weight.t(), None),
            num_local_heads=H,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope,
            v_head_dim=v_dim,
            rotary_emb=None,
            attn_mha=SimpleNamespace(layer_id=0),
        )

        kv_writes = []
        backend = TokenspeedMLABackend.__new__(TokenspeedMLABackend)
        backend.token_to_kv_pool = SimpleNamespace(
            set_mla_kv_buffer=lambda *args: kv_writes.append(args)
        )

        def _no_rope(**_):
            self.fail("NoPE layer must not enter the fused RoPE path")

        backend._fused_rope_fp8_quantize = _no_rope

        q_fp8, k_fp8, v_fp8 = backend.prepare_prefill_qkv(
            q=q,
            q_pe=q_pe,
            kv_a=kv_a,
            k_pe=k_pe,
            positions=positions,
            layer=layer,
            forward_batch=SimpleNamespace(out_cache_loc=out_cache_loc),
        )

        kv = layer.kv_b_proj(kv_a)[0].view(T, H, nope + v_dim)
        k_ref = torch.cat([kv[..., :nope], k_pe.expand(-1, H, -1)], dim=-1)
        v_ref = kv[..., nope:]
        for name, out, ref in (
            ("q", q_fp8, q),
            ("k", k_fp8, k_ref),
            ("v", v_fp8, v_ref),
        ):
            with self.subTest(tensor=name):
                self.assertTrue(out.is_contiguous())
                self._assert_fp8_equal(out, ref)

        # KV cache receives the FP8 latent and the unrotated k_pe.
        self.assertEqual(len(kv_writes), 1)
        attn_layer, loc, cache_k_nope, cache_k_rope = kv_writes[0]
        self.assertIs(attn_layer, layer.attn_mha)
        self.assertIs(loc, out_cache_loc)
        self._assert_fp8_equal(cache_k_nope, kv_a.unsqueeze(1))
        self._assert_fp8_equal(cache_k_rope, k_pe)


if __name__ == "__main__":
    unittest.main()
