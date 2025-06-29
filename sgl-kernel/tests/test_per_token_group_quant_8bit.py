import pickle
import time
import itertools
from pathlib import Path

import pytest
import torch

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
            # [127, 128, 512, 1024, 4096, 8192],  # num_tokens
            # [256, 512, 1024, 2048, 4096],  # hidden_dim
            [768],
            [16384],
            # TODO support group size 8 (change 16 to 8 thread causes this error?)
            # TODO support group size != 128
            # [8, 16, 32, 64, 128],  # group_size
            [128],  # group_size
            # TODO test int8
            [fp8_type_],  # dtype
            [
                # dict(
                #     column_major_scales=False,
                #     scale_tma_aligned=False,
                #     scale_ue8m0=False,
                # ),
                # dict(
                #     column_major_scales=True,
                #     scale_tma_aligned=False,
                #     scale_ue8m0=False,
                # ),
                # dict(
                #     column_major_scales=True,
                #     scale_tma_aligned=True,
                #     scale_ue8m0=False,
                # ),
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

    try:
        assert_fp8_all_close(x_q_triton, x_q_sglang)
        torch.testing.assert_close(
            x_s_triton.contiguous(),
            x_s_sglang.contiguous(),
            rtol=1e-3,
            atol=1e-5,
        )
    except AssertionError:
        data = dict(
            x_q_triton=x_q_triton.cpu().numpy(), x_s_triton=x_s_triton.cpu().numpy(),
            x_q_sglang=x_q_sglang.cpu().numpy(), x_s_sglang=x_s_sglang.cpu().numpy(),
        )
        path = f'/data/numa0/tom/temp_sglang_server2local/test_dump_{time.time()}.pickle'
        print(f"Dump error data to {path=}")
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        raise


if __name__ == "__main__":
    pytest.main([__file__])
