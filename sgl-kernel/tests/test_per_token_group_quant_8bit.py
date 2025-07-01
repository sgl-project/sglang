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
from sglang.srt.utils import is_hip
from sglang.srt.utils import get_bool_env_var

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn

configs = list(
    # itertools.product(
    #     [1, 4, 16, 64, 127, 128, 512, 1024, 4096, 8192],  # num_tokens
    #     # TODO
    #     # [256, 512, 1024, 1536, 2048, 4096, 7168, 16384],  # hidden_dim
    #     [512, 1024, 1536, 2048, 4096, 7168, 16384],  # hidden_dim
    #     # TODO support group size != 128
    #     # [8, 16, 32, 64, 128],  # group_size
    #     [128],  # group_size
    #     [fp8_type_, torch.int8],  # dtype
    #     [
    #         dict(
    #             column_major_scales=False,
    #             scale_tma_aligned=False,
    #             scale_ue8m0=False,
    #             fuse_silu_and_mul=False,
    #             masked_layout_mode=None,
    #         ),
    #         dict(
    #             column_major_scales=True,
    #             scale_tma_aligned=False,
    #             scale_ue8m0=False,
    #             fuse_silu_and_mul=False,
    #             masked_layout_mode=None,
    #         ),
    #         dict(
    #             column_major_scales=True,
    #             scale_tma_aligned=True,
    #             scale_ue8m0=False,
    #             fuse_silu_and_mul=False,
    #             masked_layout_mode=None,
    #         ),
    #         dict(
    #             column_major_scales=True,
    #             scale_tma_aligned=True,
    #             scale_ue8m0=True,
    #             fuse_silu_and_mul=False,
    #             masked_layout_mode=None,
    #         ),
    #     ],
    # )
) + list(
    itertools.product(
        [1, 4, 1 * 8, 4 * 8, 64 * 8, 256 * 8, 768 * 8],
        # # [256, 512, 1024, 2048, 4096], # TODO
        [512, 1024, 2048, 4096],
        [128],
        [fp8_type_],
        [
            # dict(
            #     column_major_scales=True,
            #     scale_tma_aligned=True,
            #     scale_ue8m0=True,
            #     fuse_silu_and_mul=True,
            #     masked_layout_mode=None,
            # ),
            # TODO
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                fuse_silu_and_mul=True,
                masked_layout_mode="balanced",
            ),
            # dict(
            #     column_major_scales=True,
            #     scale_tma_aligned=True,
            #     scale_ue8m0=True,
            #     fuse_silu_and_mul=True,
            #     masked_layout=True,
            #     masked_layout_mode="imbalanced",
            # ),
        ],
    )
)


@pytest.mark.parametrize(
    "num_tokens, hidden_dim, group_size, dst_dtype, flags", configs
)
def test_per_token_group_quant_with_column_major(
    num_tokens,
    hidden_dim,
    group_size,
    dst_dtype,
    flags,
):
    if (flags["scale_ue8m0"] and (group_size != 128)) or (
        (dst_dtype == torch.int8) and flags["column_major_scales"]
    ):
        pytest.skip()
        return

    x, masked_m = create_per_token_group_quant_test_data(
        num_tokens=num_tokens, hidden_dim=hidden_dim, flags=flags
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

    x_q_triton, x_s_triton = triton_per_token_group_quant_8bit(**execute_kwargs)
    x_q_sglang, x_s_sglang = sglang_per_token_group_quant_8bit(**execute_kwargs)

    # torch.set_printoptions(profile="full")
    # print(f"{x.shape=} {x_q_triton.shape=} {x_s_triton.shape=} {x_q_sglang.shape=} {x_s_sglang.shape=}")
    # print(f"{x=}")
    # print(f"{masked_m=}")
    # print(f"{x_q_triton=}")
    # print(f"{x_s_triton=}")
    # print(f"{x_q_sglang=}")
    # print(f"{x_s_sglang=}")
    # torch.set_printoptions(profile="default")

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
        if (d := os.environ.get("SGLANG_DUMP_TEST_ERROR_DIR", "")) != "":
            import matplotlib.pyplot as plt
            base_stem = time.time()
            for name, value in [
                ("x_q", x_q_triton != x_q_sglang),
                ("x_s", x_s_triton != x_s_sglang),
            ]:
                plt.figure(figsize=(20, 20))
                plt.imshow((value * 1.0).cpu().numpy())
                p = Path(d) / f"{base_stem}_{name}.png"
                print(f"Write diff to {p}", flush=True)
                plt.savefig(p)

        raise


if __name__ == "__main__":
    pytest.main([__file__])
