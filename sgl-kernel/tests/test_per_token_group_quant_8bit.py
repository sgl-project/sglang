import itertools
import os
import time
from pathlib import Path

import pytest
import torch
from sgl_kernel.test_utils import (
    assert_all_close_or_tiny_diff,
    create_per_token_group_quant_test_data,
)

from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_8bit as triton_per_token_group_quant_8bit,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_8bit
from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn

configs = list(
    itertools.product(
        [1, 4, 16, 64, 127, 128, 512, 1024, 4096, 8192],  # num_tokens
        [128, 256, 384, 512, 1024, 1536, 1664, 2048, 4096, 7168, 16384],  # hidden_dim
        [16, 32, 64, 128],  # group_size
        [None],  # num_ranks
        [fp8_type_, torch.int8],  # dtype
        [
            dict(
                column_major_scales=False,
                scale_tma_aligned=False,
                scale_ue8m0=False,
                fuse_silu_and_mul=False,
                masked_layout_mode=None,
            ),
            dict(
                column_major_scales=True,
                scale_tma_aligned=False,
                scale_ue8m0=False,
                fuse_silu_and_mul=False,
                masked_layout_mode=None,
            ),
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
                fuse_silu_and_mul=False,
                masked_layout_mode=None,
            ),
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                fuse_silu_and_mul=False,
                masked_layout_mode=None,
            ),
        ],
    )
) + list(
    itertools.product(
        [1, 4, 1 * 8, 4 * 8, 64 * 8, 256 * 8, 768 * 8],
        # TODO support more
        [2048],
        [128],
        [8, 16, 32, 48],
        [fp8_type_],
        [
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                fuse_silu_and_mul=True,
                masked_layout_mode=None,
            ),
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                fuse_silu_and_mul=True,
                masked_layout_mode="balanced",
            ),
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                fuse_silu_and_mul=True,
                masked_layout_mode="imbalanced",
            ),
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                fuse_silu_and_mul=True,
                masked_layout_mode="extreme",
            ),
        ],
    )
)


@pytest.mark.parametrize(
    "num_tokens, hidden_dim, group_size, num_ranks, dst_dtype, flags", configs
)
def test_per_token_group_quant_with_column_major(
    num_tokens,
    hidden_dim,
    group_size,
    num_ranks,
    dst_dtype,
    flags,
):
    print(
        f"{num_tokens=} {hidden_dim=} {group_size=} {num_ranks=} {dst_dtype=} {flags=}"
    )

    arch_major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if flags["scale_ue8m0"] and (arch_major <= 9):
        pytest.skip("Only Blackwell need ue8m0 fusion")
        return

    if (flags["scale_ue8m0"] and (group_size != 128)) or (
        (dst_dtype == torch.int8) and flags["column_major_scales"]
    ):
        pytest.skip()
        return

    x, masked_m = create_per_token_group_quant_test_data(
        num_tokens=num_tokens, hidden_dim=hidden_dim, num_ranks=num_ranks, flags=flags
    )

    # print("hack data!!!")
    # x = torch.full_like(x, fill_value=100)

    execute_kwargs = dict(
        x=x,
        masked_m=masked_m,
        group_size=group_size,
        eps=1e-10,
        dst_dtype=dst_dtype,
        **{k: v for k, v in flags.items() if k not in ["masked_layout_mode"]},
    )

    def _postprocess(x_q, x_s):
        if masked_m is not None:
            print(f"Mask tokens after {masked_m} to be zero")
            for i in range(len(masked_m)):
                x_q[i, masked_m[i] :, :] = 0
                x_s[i, masked_m[i] :, :] = 0
        return x_q, x_s

    x_q_triton, x_s_triton = _postprocess(
        *triton_per_token_group_quant_8bit(**execute_kwargs)
    )
    x_q_sglang, x_s_sglang = _postprocess(
        *sglang_per_token_group_quant_8bit(**execute_kwargs, enable_v2=True)
    )

    try:
        assert_all_close_or_tiny_diff(x_q_triton, x_q_sglang)
        torch.testing.assert_close(
            x_s_triton.contiguous(),
            x_s_sglang.contiguous(),
            rtol=1e-3,
            atol=1e-5,
            msg=lambda message: message + f" {x_s_triton=} {x_s_sglang=}",
        )
    except AssertionError:
        print(
            f"{x.shape=} {x_q_triton.shape=} {x_s_triton.shape=} {x_q_sglang.shape=} {x_s_sglang.shape=}"
        )
        print(f"{x=}")
        print(f"{masked_m=}")
        print(f"{x_q_triton=}")
        print(f"{x_s_triton=}")
        print(f"{x_q_sglang=}")
        print(f"{x_s_sglang=}")

        raise


if __name__ == "__main__":
    pytest.main([__file__])
