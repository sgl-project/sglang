import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
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


register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9,
    "FP8 prefill preparation requires Hopper or newer",
)
class TestTokenspeedMLAPrefillPreparation(CustomTestCase):
    def test_prefill_preserves_position_components_and_cache(self):
        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLABackend,
        )

        torch.manual_seed(0)
        q = torch.randn(3, 2, 192, device="cuda", dtype=torch.bfloat16)
        kv_a = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(3, 2, 256, device="cuda", dtype=torch.bfloat16)
        k_pe = torch.randn(3, 1, 64, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(3, device="cuda")
        locations = torch.tensor([2, 0, 1], device="cuda")
        backend = TokenspeedMLABackend.__new__(TokenspeedMLABackend)
        backend.token_to_kv_pool = Mock()
        identity_rope = SimpleNamespace(
            cos_sin_cache=torch.cat(
                [torch.ones(3, 32, device="cuda"), torch.zeros(3, 32, device="cuda")],
                dim=-1,
            ),
            is_neox_style=True,
        )
        for rotary_emb in (None, identity_rope):
            with self.subTest(uses_rope=rotary_emb is not None):
                layer = SimpleNamespace(
                    rotary_emb=rotary_emb,
                    num_local_heads=2,
                    qk_nope_head_dim=128,
                    qk_rope_head_dim=64,
                    v_head_dim=128,
                    kv_b_proj=lambda _: (kv, None),
                    attn_mha=object(),
                )
                actual = backend.prepare_prefill_qkv(
                    q=q,
                    q_pe=q[..., 128:],
                    kv_a=kv_a,
                    k_pe=k_pe,
                    positions=positions,
                    layer=layer,
                    forward_batch=SimpleNamespace(out_cache_loc=locations),
                )
                expected = (
                    q,
                    torch.cat([kv[..., :128], k_pe.expand(-1, 2, -1)], -1),
                    kv[..., 128:],
                )
                for output, reference in zip(actual, expected):
                    self.assertEqual(output.dtype, torch.float8_e4m3fn)
                    torch.testing.assert_close(
                        output.float(), reference.to(output.dtype).float()
                    )
                write = backend.token_to_kv_pool.set_mla_kv_buffer.call_args.args
                self.assertIs(write[0], layer.attn_mha)
                self.assertIs(write[1], locations)
                torch.testing.assert_close(
                    write[2].float(), kv_a.to(torch.float8_e4m3fn).unsqueeze(1).float()
                )
                torch.testing.assert_close(
                    write[3].float(), k_pe.to(torch.float8_e4m3fn).float()
                )


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


if __name__ == "__main__":
    unittest.main()
