"""Host-only coverage of the CUDA eligibility boundary and separate reference."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import fp4_indexer, fp4_indexer_cuda
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels.fp4_candidate import make_inputs

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def metadata_inputs(**kwargs):
    x = make_inputs(device="cpu", **kwargs)
    # Preserve real Torch shapes/strides without initializing a CUDA context.
    for name in ("q", "weights", "req", "req_table", "lens", "table", "blocks"):
        t = getattr(x, name)
        setattr(
            x,
            name,
            SimpleNamespace(
                shape=t.shape,
                ndim=t.ndim,
                dtype=t.dtype,
                stride=t.stride,
                device=torch.device("cuda:0"),
                is_cuda=True,
            ),
        )
    return x


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("page_size", [1, 8, 16, 64, 128, 257])
def test_supported_metadata_preserves_generic_pages_and_strides(ratio, page_size):
    x = metadata_inputs(ratio=ratio, padded=True, strided=True)
    x.page_size = page_size
    x.table.shape = (3, page_size * 68)
    with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
        assert fp4_indexer_cuda.is_fp4_index_logits_candidates_cuda_supported(*x.args())


@pytest.mark.parametrize(
    "field,attribute,value",
    [
        ("q", "is_cuda", False),
        ("q", "dtype", torch.float16),
        ("q", "shape", (4, 16, 128)),
        ("q", "ndim", 2),
        ("weights", "shape", (4, 16)),
        ("req", "dtype", torch.int16),
        ("req", "shape", (3,)),
        ("lens", "dtype", torch.float32),
        ("req_table", "dtype", torch.float32),
        ("req_table", "ndim", 1),
        ("req_table", "stride", lambda _: 2),
        ("table", "dtype", torch.int8),
        ("table", "shape", (6, 123)),
        ("table", "stride", lambda _: 2),
        ("blocks", "ndim", 1),
        ("blocks", "dtype", torch.int16),
        ("blocks", "device", torch.device("cuda:1")),
    ],
)
def test_unsupported_metadata(field, attribute, value):
    x = metadata_inputs()
    setattr(getattr(x, field), attribute, value)
    with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
        assert not fp4_indexer_cuda.is_fp4_index_logits_candidates_cuda_supported(
            *x.args()
        )


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (10, 0), (12, 0)])
def test_other_architectures_use_triton(capability):
    x = metadata_inputs()
    with patch.object(torch.cuda, "get_device_capability", return_value=capability):
        assert not fp4_indexer_cuda.is_fp4_index_logits_candidates_cuda_supported(
            *x.args()
        )


@pytest.mark.parametrize("use_cuda", [False, True])
def test_outer_dispatch_chooses_independent_entry(use_cuda):
    args = make_inputs(device="cpu").args()
    with (
        patch.object(
            fp4_indexer_cuda,
            "is_fp4_index_logits_candidates_cuda_supported",
            return_value=use_cuda,
        ),
        patch.object(fp4_indexer_cuda, "fp4_index_logits_candidates_cuda") as cuda,
        patch.object(fp4_indexer, "fp4_index_logits_candidates_triton") as triton,
    ):
        selected, other = (cuda, triton) if use_cuda else (triton, cuda)
        assert fp4_indexer.fp4_index_logits_candidates(*args) is selected.return_value
        selected.assert_called_once_with(*args)
        other.assert_not_called()


def test_explicit_reference_does_not_reenter_dispatch():
    x = make_inputs(requests=0, device="cpu")
    with patch.object(
        fp4_indexer_cuda,
        "is_fp4_index_logits_candidates_cuda_supported",
        side_effect=AssertionError("reference reentered CUDA dispatch"),
    ):
        out = fp4_indexer.fp4_index_logits_candidates_triton(*x.args())
    assert out.shape == (0, 72) and out.dtype == torch.float32


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
