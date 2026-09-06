import importlib.util
import unittest

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


class TestTokenspeedMLANoPEPrefillQuantize(CustomTestCase):
    """NoPE MLA layers must still get packed FP8 prefill Q/K.

    Layers built with ``skip_rope`` carry no rotary embedding, so the prefill
    quantize runs with ``cos_sin_cache=None``. It must return the same packed
    ``[nope | pe]`` FP8 layout as the RoPE path, and the head-0 pe slice it
    writes to the KV cache must equal the plain FP8 cast of the unroped
    ``k_pe`` that the decode path reads back.
    """

    T = 7
    NUM_HEADS = 4
    QK_NOPE_HEAD_DIM = 128
    QK_ROPE_HEAD_DIM = 64
    KV_LORA_RANK = 512

    def test_nope_prefill_quantize_packs_without_rope(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = self.QK_NOPE_HEAD_DIM + self.QK_ROPE_HEAD_DIM

        q_nope = torch.randn(
            self.T,
            self.NUM_HEADS,
            self.QK_NOPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        q_pe = torch.randn(
            self.T,
            self.NUM_HEADS,
            self.QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        k_nope = torch.randn(
            self.T,
            self.NUM_HEADS,
            self.QK_NOPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        # k_pe reaches the backend as the strided tail of the latent cache.
        latent_cache = torch.randn(
            self.T,
            1,
            self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        k_pe = latent_cache[:, :, self.KV_LORA_RANK :]

        q_fp8, k_fp8 = TokenspeedMLABackend._fused_rope_fp8_quantize(
            q_nope=q_nope,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            cos_sin_cache=None,
            positions=torch.arange(self.T, device=device),
            is_neox=True,
            qk_nope_head_dim=self.QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=self.QK_ROPE_HEAD_DIM,
        )

        for name, out in (("q", q_fp8), ("k", k_fp8)):
            with self.subTest(tensor=name):
                self.assertEqual(out.shape, (self.T, self.NUM_HEADS, head_dim))
                self.assertEqual(out.dtype, torch.float8_e4m3fn)
                self.assertTrue(out.is_contiguous())

        fp8 = torch.float8_e4m3fn
        nope = self.QK_NOPE_HEAD_DIM
        self.assertTrue(torch.equal(q_fp8[..., :nope], q_nope.to(fp8)))
        self.assertTrue(torch.equal(q_fp8[..., nope:], q_pe.to(fp8)))
        self.assertTrue(torch.equal(k_fp8[..., :nope], k_nope.to(fp8)))
        # The slice prepare_prefill_qkv writes into the KV cache; the decode
        # NoPE path reads it back as a plain cast of the unroped k_pe.
        self.assertTrue(torch.equal(k_fp8[:, 0:1, nope:], k_pe.to(fp8)))


if __name__ == "__main__":
    unittest.main()
