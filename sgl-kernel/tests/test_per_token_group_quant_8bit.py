import itertools

import pytest
import torch

from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_8bit as triton_per_token_group_quant_8bit,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_8bit
from sglang.srt.layers.quantization.utils import assert_fp8_all_close
from sglang.srt.utils import is_hip

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


@pytest.mark.parametrize(
    "num_tokens, hidden_dim, group_size, dst_dtype, flags",
    list(
        itertools.product(
            [127, 128, 512, 1024, 4096, 8192],  # num_tokens
            [256, 512, 1024, 2048, 4096],  # hidden_dim
            [8, 16, 32, 64, 128],  # group_size
            # TODO test int8
            [fp8_type_],  # dtype
            [
                dict(
                    column_major_scales=False,
                    scale_tma_aligned=False,
                    scale_ue8m0=False,
                ),
                dict(
                    column_major_scales=True,
                    scale_tma_aligned=False,
                    scale_ue8m0=False,
                ),
                dict(
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=False,
                ),
                dict(
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                ),
            ],
        )
    ),
)
def test_per_token_group_quant_with_column_major(
    num_tokens,
    hidden_dim,
    group_size,
    dst_dtype,
    flags,
):
    if flags["scale_ue8m0"] and ((group_size != 128) or (hidden_dim % 512 != 0)):
        pytest.skip()
        return
    if flags["scale_ue8m0"] and not deep_gemm_wrapper.DEEPGEMM_BLACKWELL:
        pytest.skip("scale_ue8m0 only supported on Blackwell")
        return

    x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)

    execute_kwargs = dict(
        x=x,
        group_size=group_size,
        eps=1e-10,
        dst_dtype=dst_dtype,
        **flags,
    )

    x_q_triton, x_s_triton = triton_per_token_group_quant_8bit(**execute_kwargs)
    x_q_sglang, x_s_sglang = sglang_per_token_group_quant_8bit(**execute_kwargs)

    # torch.set_printoptions(profile="full")
    # print(f"{x_q_triton=}")
    # print(f"{x_s_triton=}")
    # print(f"{x_q_sglang=}")
    # print(f"{x_s_sglang=}")
    # torch.set_printoptions(profile="default")

    assert_fp8_all_close(x_q_triton, x_q_sglang)
    torch.testing.assert_close(
        x_s_triton.contiguous(),
        x_s_sglang.contiguous(),
        rtol=1e-3,
        atol=1e-5,
        msg=lambda message: message + f" {x_s_triton=} {x_s_sglang=}",
    )


if __name__ == "__main__":
    pytest.main([__file__])
