import pytest
import torch

from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize("dtype_strings", [False, True])
def test_receive_bucket_preserves_mixed_dtype_wire_bytes(dtype_strings):
    scale_bits = torch.tensor(
        [0, -2147483648, 2143289345, 2139095040], dtype=torch.int32
    )
    values = [
        ("packed_weight", torch.arange(16, dtype=torch.uint8).reshape(2, 8)),
        ("block_scales", torch.arange(8, dtype=torch.uint8).view(torch.float8_e4m3fn)),
        ("global_scale", scale_bits.view(torch.float32)[2].reshape(())),
        ("other_scales", scale_bits.view(torch.float32)),
        ("bf16", torch.tensor([1.5, -2.0], dtype=torch.bfloat16)),
        ("empty", torch.empty((0, 4), dtype=torch.float32)),
    ]
    sent = FlattenedTensorBucket(named_tensors=values)
    dtypes = [value.dtype for _, value in values]
    if dtype_strings:
        dtypes = [str(dtype).removeprefix("torch.") for dtype in dtypes]
    received = FlattenedTensorBucket.empty(
        [name for name, _ in values],
        dtypes,
        [value.shape for _, value in values],
        "cpu",
    )
    assert received.metadata == sent.metadata
    received.flattened_tensor.copy_(sent.flattened_tensor)
    for (name, original), (received_name, actual), meta in zip(
        values, received.reconstruct_tensors(), sent.metadata, strict=True
    ):
        assert name == received_name
        assert original.shape == actual.shape and original.dtype == actual.dtype
        assert torch.equal(
            original.reshape(-1).view(torch.uint8), actual.reshape(-1).view(torch.uint8)
        )
        if actual.numel():
            assert (
                actual.data_ptr()
                == received.flattened_tensor.data_ptr() + meta.start_idx
            )


@pytest.mark.parametrize(
    "names,dtypes,shapes",
    [
        ([], [], []),
        (["a"], [], [[4]]),
        (["a"], [torch.float32], [[-1, -1]]),
        (["a", "b"], [torch.uint8, torch.float32], [[3], []]),
    ],
)
def test_receive_bucket_rejects_invalid_metadata(names, dtypes, shapes):
    with pytest.raises(ValueError):
        FlattenedTensorBucket.empty(names, dtypes, shapes, "cpu")
