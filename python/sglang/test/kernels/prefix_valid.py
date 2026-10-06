"""Shared eager oracle and byte assertions for prefix-valid KV-cache tests."""

from contextlib import contextmanager

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype


def make_kv_cache(total_slots, row_dim, dtype=fp8_dtype, device="cuda"):
    store_dtype = torch.uint8 if dtype == fp8_dtype else dtype
    return tuple(
        torch.full(
            (total_slots, 1, row_dim), fill, dtype=store_dtype, device=device
        ).view(dtype)
        for fill in (0x5A, 0xA5)
    )


def assert_bytes_equal(actual, expected, label):
    # Integer comparison preserves NaN encodings and the sign of zero.
    torch.testing.assert_close(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
        msg=lambda message: f"{label}: {message}",
    )


@contextmanager
def assert_prefix_commit(
    k, v, k_cache, v_cache, loc, lengths, k_scale, v_scale, *, inputs_mutated=False
):
    """Check the whole cache (including untouched slots) against eager division."""
    rows = torch.arange(loc.shape[1], device=loc.device)
    valid = (rows[None, :] < lengths[:, None]).reshape(-1)
    src = torch.nonzero(valid, as_tuple=False).flatten()
    dst = loc.reshape(-1)[src].long()
    before = (k.clone(), v.clone())
    scaled, expected = [], []
    for source, cache, scale in zip(before, (k_cache, v_cache), (k_scale, v_scale)):
        value = source.clone()
        if scale is not None:
            value.div_(scale)
        scaled.append(value)
        target = cache.contiguous().view(torch.uint8).clone()
        target[dst] = value.to(cache.dtype).contiguous().view(torch.uint8)[src]
        expected.append(target)

    yield

    for label, cache, reference in zip(
        ("K cache", "V cache"), (k_cache, v_cache), expected
    ):
        assert_bytes_equal(cache, reference, label)
    for label, source, reference in zip(
        ("K input", "V input"), (k, v), scaled if inputs_mutated else before
    ):
        assert_bytes_equal(source, reference, label)
