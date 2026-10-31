import unittest

import torch

from sglang.kernels.ops.attention import extend_attention as ea
from sglang.srt.environ import envs
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")


@unittest.skipUnless(is_xpu(), "requires an Intel XPU")
class TestExtendAttentionTensorDescriptor(CustomTestCase):
    def test_nonfinite_values_in_adjacent_request(self):
        """A packed request's NaN/Inf V must not contaminate another request."""
        self.assertTrue(ea._support_tensor_descriptor)
        qo_indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device="xpu")
        kv_indptr = torch.zeros(3, dtype=torch.int32, device="xpu")
        kv_indices = torch.empty(0, dtype=torch.int32, device="xpu")

        for dtype in (torch.float16, torch.bfloat16):
            q = torch.zeros((2, 1, 64), dtype=dtype, device="xpu")
            k = torch.zeros_like(q)
            v = torch.ones_like(q)
            self.assertTrue(ea._descriptor_fits(tensor=v, head_dim=64, tile_width=64))
            for nonfinite in (float("nan"), float("inf"), float("-inf")):
                v[1].fill_(nonfinite)
                for use_descriptor in (False, True):
                    with self.subTest(
                        dtype=dtype,
                        nonfinite=nonfinite,
                        use_descriptor=use_descriptor,
                    ):
                        output = torch.empty_like(q)
                        with envs.SGLANG_USE_TRITON_ATTN_TENSOR_DESC.override(
                            use_descriptor
                        ):
                            ea.extend_attention_fwd(
                                q,
                                k,
                                v,
                                output,
                                k,
                                v,
                                qo_indptr,
                                kv_indptr,
                                kv_indices,
                                custom_mask=None,
                                is_causal=True,
                                mask_indptr=None,
                                max_len_extend=1,
                                k_scale=1.0,
                                v_scale=1.0,
                            )
                        # With one token and no prefix, attention returns V.
                        torch.testing.assert_close(output[0], v[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
