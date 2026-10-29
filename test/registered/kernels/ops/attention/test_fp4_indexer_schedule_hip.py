"""JIT specializations of the DeepSeek-V4 FP4 prefill schedule kernels on HIP.

Every prefill batch, and every row chunk the indexer splits a long-context request into,
builds the schedule with its own row count, page-table width and row offset. The prep
kernel must compile one variant per BLOCK_T for all of them, the engine-init warmup must
compile exactly those, and the schedule must stay AITER's.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
import triton

from sglang.kernels.ops.attention.dsv4 import fp4_indexer_schedule_hip as schedule
from sglang.kernels.ops.attention.dsv4.fp4_indexer_schedule_hip import (
    MAX_FUSED_ROWS,
    build_prefill_schedule,
    pad_page_table,
    padded_page_table_shape,
    warmup_prefill_schedule,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd-mi35x")

requires_gfx95 = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_hip() and is_gfx95_supported()),
    reason="The FP4 prefill schedule wraps AITER CDNA4 (gfx95x) kernels.",
)

_AITER_PREFILL = "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill"
_CTA_INFO_WIDTH = 6


def _ratio2_row_chunks(context_len: int, rows: int = 16384):
    """(rows, page-table width, row offset) of the launches a `rows`-row chunk of one
    request at `context_len` makes on the DeepSeek-V4.1 ratio-2 indexer: rows split to fit
    the 2 GiB logits block over its 2 * ceil(L / 256)-page table."""
    width = 2 * -(-context_len // 256)
    per_chunk = 131072 // -(-context_len // 8192)
    return [
        (min(per_chunk, rows - lo), width, lo)
        for lo in range(0, rows, per_chunk)
        if min(per_chunk, rows - lo) <= MAX_FUSED_ROWS
    ]


# (rows, page-table width, row offset, page_table_bucket): whole small batches over short
# and long contexts, then the row chunks of 16K-row prefill chunks at 74K, 256K and 409K.
_LAUNCHES = [
    (1, 1, 0, 4),
    (2, 3, 1, 4),
    (15, 5, 2, 4),
    (16, 33, 3, 4),
    (17, 2, 0, 64),
    (100, 128, 5, 64),
    (255, 810, 0, 64),
    (1000, 3194, 2, 64),
    (1025, 64, 1, 64),
    (2049, 2048, 3, 64),
    (MAX_FUSED_ROWS, 160, 0, 64),
    *(c + (64,) for c in _ratio2_row_chunks(74_000)),
    *(c + (64,) for c in _ratio2_row_chunks(262_144)[:2]),
    *(c + (64,) for c in _ratio2_row_chunks(409_000)[-2:]),
]


def _block_t(rows: int) -> int:
    return max(16, triton.next_power_of_2(rows))


def _schedule_inputs(rows, width, offset, device, seed=0):
    """A row slice of a taller page table and its local ends, as the indexer passes."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    table = torch.empty((offset + rows, width), dtype=torch.int32, device=device)
    ends = torch.empty(offset + rows, dtype=torch.int32, device=device)
    table[offset:] = torch.randint(
        0, 1 << 20, (rows, width), generator=gen, dtype=torch.int32
    )
    ends[offset:] = torch.randint(
        0, width * 64 + 1, (rows,), generator=gen, dtype=torch.int32
    )
    ends[offset::7] = 0
    return table[offset:], ends[offset:]


def _build(rows, width, offset, bucket, device, seed=0):
    page_table, local_ends = _schedule_inputs(rows, width, offset, device, seed)
    _, _, padded_width = padded_page_table_shape(page_table, bucket)
    cta_count = max(1024, rows)
    cta_info = torch.empty(
        (cta_count, _CTA_INFO_WIDTH), dtype=torch.int32, device=device
    )
    guarded, buffers = build_prefill_schedule(
        page_table=page_table,
        local_ends=local_ends,
        cta_info_out=cta_info,
        parallel_unit_num=cta_count,
        max_seq_len=padded_width * 64,
        page_table_bucket=bucket,
    )
    return page_table, local_ends, guarded, buffers, cta_info, padded_width


def test_schedule_kernels_specialize_on_no_runtime_argument() -> None:
    for kernel in (
        schedule._prefill_schedule_prep_kernel,
        schedule._pad_page_table_kernel,
    ):
        for param in kernel.params:
            if param.is_constexpr:
                continue
            if param.name.endswith("_ptr"):
                assert param.do_not_specialize_on_alignment, param.name
            else:
                assert param.do_not_specialize, param.name


def test_prefill_launches_share_one_specialization_per_block_t(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Triton's launch key for gfx950, computed on the host from the arguments
    build_prefill_schedule passes: one per BLOCK_T whatever the batch looks like."""
    try:
        from triton.backends.compiler import GPUTarget
        from triton.compiler import make_backend
        from triton.runtime.jit import (
            compute_cache_key,
            create_function_from_signature,
        )

        backend = make_backend(GPUTarget("hip", "gfx950", 64))
    except Exception as e:  # Triton internals or its HIP backend are missing
        pytest.skip(f"no host-side Triton HIP specialization: {e!r}")

    kernel = schedule._prefill_schedule_prep_kernel
    binder = create_function_from_signature(kernel.signature, kernel.params, backend)
    keys = {}

    class _Key:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                _, specialization, options = binder(*args, **kwargs)
                key = compute_cache_key({}, specialization, options)
                keys.setdefault(kwargs["BLOCK_T"], set()).add(key)

            return launch

    class _NoLaunch:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: None

    monkeypatch.setitem(
        sys.modules,
        _AITER_PREFILL,
        types.SimpleNamespace(
            _prefill_cta_info_kernel=_NoLaunch(), compute_prefill_schedule=None
        ),
    )
    monkeypatch.setattr(schedule, "_prefill_schedule_prep_kernel", _Key())

    launches = list(_LAUNCHES)
    for rows in (3, 31, 48, 129, 640, 1823, 2730, 4000):
        for offset in range(4):
            launches.append((rows, 5 + 16 * offset, offset * rows, 4))
    for rows, width, offset, bucket in launches:
        _build(rows, width, offset, bucket, torch.device("cpu"))

    assert set(keys) == {_block_t(launch[0]) for launch in launches}
    assert all(len(variants) == 1 for variants in keys.values()), {
        block_t: len(variants) for block_t, variants in keys.items()
    }


@requires_gfx95
def test_warmup_compiles_every_prefill_schedule_variant() -> None:
    device = torch.device("cuda")
    caches = {
        kernel.fn.__name__: kernel.device_caches[torch.cuda.current_device()][0]
        for kernel in (
            schedule._prefill_schedule_prep_kernel,
            schedule._pad_page_table_kernel,
        )
    }

    warmup_prefill_schedule(device)
    warm = {name: len(cache) for name, cache in caches.items()}
    assert warm["_prefill_schedule_prep_kernel"] == len(
        {_block_t(rows) for rows in range(1, MAX_FUSED_ROWS + 1)}
    )

    for seed, (rows, width, offset, bucket) in enumerate(_LAUNCHES):
        _build(rows, width, offset, bucket, device, seed)
        page_table, _ = _schedule_inputs(rows, width, offset, device, seed)
        pad_page_table(page_table, page_table_bucket=bucket)
    assert {name: len(cache) for name, cache in caches.items()} == warm


@requires_gfx95
@pytest.mark.parametrize("launch", _LAUNCHES, ids=lambda c: "x".join(map(str, c)))
def test_prefill_schedule_matches_aiter(launch) -> None:
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        compute_prefill_schedule,
    )

    rows, width, offset, bucket = launch
    device = torch.device("cuda")
    page_table, local_ends, guarded, buffers, cta_info, padded_width = _build(
        rows, width, offset, bucket, device, seed=rows
    )

    ref_safe, ref_cta_info, _ = compute_prefill_schedule(
        torch.arange(rows, dtype=torch.int32, device=device),
        torch.zeros(rows, dtype=torch.int32, device=device),
        local_ends.clone(),
        block_k=256,
        parallel_unit_num=cta_info.shape[0],
        max_seq_len=padded_width * 64,
    )
    ref_guarded = torch.zeros_like(guarded)
    ref_guarded[:, :width] = page_table

    assert torch.equal(buffers.safe, ref_safe)
    assert torch.equal(cta_info, ref_cta_info)
    assert torch.equal(guarded, ref_guarded)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
