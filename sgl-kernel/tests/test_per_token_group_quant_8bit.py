import itertools

import pytest
import torch

from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_8bit as triton_per_token_group_quant_8bit,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_8bit
from sglang.srt.layers.quantization.utils import assert_fp8_all_close
from sglang.srt.utils import is_hip
from sgl_kernel.test_utils import create_per_token_group_quant_test_data

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn

configs = list(
    itertools.product(
        [1, 4, 16, 64, 127, 128, 512, 1024, 4096, 8192],  # num_tokens
        [256, 512, 1024, 1536, 2048, 4096, 7168, 16384],  # hidden_dim
        # TODO support group size != 128
        # [8, 16, 32, 64, 128],  # group_size
        [128],  # group_size
        [fp8_type_, torch.int8],  # dtype
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
) + list(
    itertools.product(
        [1, 4, 1 * 8, 4 * 8, 64 * 8, 256 * 8, 768 * 8],
        [256, 1024, 2048, 4096],
        [128],
        [fp8_type_],
        [
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                fuse_silu_and_mul=True,
            ),
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                fuse_silu_and_mul=True,
                masked_layout=True,
            ),
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                fuse_silu_and_mul=True,
                masked_layout=True,
                masked_data_generation_mode="imbalanced",
            ),
        ],
    )
)

@pytest.mark.parametrize("num_tokens, hidden_dim, group_size, dst_dtype, flags", configs)
def test_per_token_group_quant_with_column_major(
    num_tokens,
    hidden_dim,
    group_size,
    dst_dtype,
    flags,
):
    if flags["scale_ue8m0"] and (group_size != 128):
        pytest.skip()
        return

    x = create_per_token_group_quant_test_data()

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
