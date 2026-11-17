"""NVFP4 host descriptors, packed staging and target/draft scale contracts."""

from unittest.mock import patch

import msgspec
import pytest
import torch

from sglang.srt.mem_cache.device_pool_info import (
    KVBufferLayout,
    KVBuffers,
    KVScaleBuffers,
    KVScaleLayout,
    MHABufferInfo,
)
from sglang.srt.mem_cache.pool_host import mha, mha_fp4
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def info(layers=2, native=True, page_size=16):
    tokens, heads, dim = 64, 2, 64
    k = tuple(
        torch.zeros(tokens, heads, dim // 2, dtype=torch.uint8) for _ in range(layers)
    )
    v = tuple(torch.zeros_like(t) for t in k)
    scales = lambda: tuple(
        torch.ones(tokens, heads, dim // 16, dtype=torch.uint8) for _ in range(layers)
    )
    native_scales = lambda: tuple(
        torch.ones(tokens // page_size, heads, page_size, dim // 16, dtype=torch.uint8)
        for _ in range(layers)
    )
    return MHABufferInfo(
        buffers=KVBuffers(
            k=k,
            v=v,
            layout=KVBufferLayout.NHD,
            k_data_ptrs=torch.tensor([t.data_ptr() for t in k], dtype=torch.uint64),
            v_data_ptrs=torch.tensor([t.data_ptr() for t in v], dtype=torch.uint64),
        ),
        page_size=page_size,
        head_num=heads,
        head_dim=dim,
        v_head_dim=dim,
        dtype=torch.uint8,
        device=torch.device("cpu"),
        scales=KVScaleBuffers(
            k=scales(),
            v=scales(),
            layout=KVScaleLayout.NVFP4,
            native_k=native_scales() if native else (),
            native_v=native_scales() if native else (),
        ),
    )


@pytest.mark.parametrize("native", [False, True])
def test_explicit_layout_selects_fp4_and_payload_view_preserves_allocations(native):
    source = info(native=native)
    assert mha.get_mha_host_pool_cls(source) is mha_fp4.MHATokenToKVPoolFP4Host
    plain = msgspec.structs.replace(source, scales=None)
    assert mha.get_mha_host_pool_cls(plain) is MHATokenToKVPoolHost
    payload = mha_fp4.nvfp4_payload_info(source)
    assert payload.buffers is source.buffers
    assert payload.head_dim == payload.v_head_dim == source.head_dim // 2
    assert payload.dtype == torch.uint8 and payload.scales is None
    assert source.scales is not None and source.head_dim == 64


def test_prebudget_staging_uses_packed_byte_width_and_reuses_existing_storage():
    source = info()
    with (
        patch.object(mha, "_is_cuda", True),
        patch.object(mha, "can_use_write_back_jit_kernel", return_value=True),
    ):
        staging = mha.prepare_mha_write_back_staging(
            source, layer_num=4, page_size=16, page_capacity=2
        )
        assert all(t.shape == (32, 4, 2, 32) for t in staging)
        again = mha.prepare_mha_write_back_staging(
            source, layer_num=4, page_size=16, page_capacity=2, staging=staging
        )
    assert again is staging


def make_constructor(source, drafts=(), **kwargs):
    return mha_fp4.MHATokenToKVPoolFP4Host(
        source,
        2,
        0,
        16,
        "page_first",
        pin_memory=False,
        device_capacity_tokens=48,
        start_layer=5,
        end_layer=7,
        draft_buffer_infos=drafts,
        **kwargs,
    )


@pytest.mark.parametrize("native", [False, True])
def test_constructor_uses_new_descriptor_api_and_keeps_two_distinct_drafts(native):
    target, draft0, draft1 = info(native=native), info(1, native), info(1, native)
    calls = []

    def parent_init(host, payload, *args, **kwargs):
        calls.append((payload, kwargs))
        host.can_use_jit = host.can_use_write_back_jit = True

    with (
        patch.object(MHATokenToKVPoolHost, "__init__", new=parent_init),
        patch.object(mha_fp4.MHATokenToKVPoolFP4Host, "_init_scale_buffers"),
        patch.object(mha_fp4, "can_use_write_back_jit_kernel", return_value=True),
    ):
        host = make_constructor(target, (draft0, draft1))
    payload, kwargs = calls[0]
    assert payload.buffers is target.buffers and payload.scales is None
    assert kwargs["start_layer"] == 5 and kwargs["end_layer"] == 7
    assert kwargs["draft_buffer_infos"][0].buffers is draft0.buffers
    assert kwargs["draft_buffer_infos"][1].buffers is draft1.buffers
    assert host._scale_dev_k_scale_buffer == [
        *target.scales.k,
        *draft0.scales.k,
        *draft1.scales.k,
    ]
    assert len(host._scale_names) == (4 if native else 2)
    if native:
        assert host._scale_dev_native_v_scale_buffer[-2] is draft0.scales.native_v[0]
        assert host._scale_dev_native_v_scale_buffer[-1] is draft1.scales.native_v[0]


@pytest.mark.parametrize(
    "bad",
    [
        "layout",
        "missing_native_v",
        "wrong_native_shape",
        "partial_payload",
        "wrong_payload_width",
        "draft_page",
        "draft_layouts",
        "multi_layer_draft",
    ],
)
def test_invalid_descriptor_fails_before_host_allocation(bad):
    target, drafts = info(), (info(1),)
    if bad == "layout":
        target = msgspec.structs.replace(
            target,
            scales=msgspec.structs.replace(target.scales, layout=KVScaleLayout.TOKEN),
        )
    elif bad == "missing_native_v":
        target = msgspec.structs.replace(
            target, scales=msgspec.structs.replace(target.scales, native_v=())
        )
    elif bad == "wrong_native_shape":
        target = msgspec.structs.replace(
            target,
            scales=msgspec.structs.replace(
                target.scales,
                native_k=(torch.ones(2, 2, 16, 4, dtype=torch.uint8),) * 2,
            ),
        )
    elif bad in ("partial_payload", "wrong_payload_width"):
        replacement = tuple(
            t[:-1] if bad == "partial_payload" else t[..., :-1].contiguous()
            for t in target.buffers.k
        )
        target = msgspec.structs.replace(
            target, buffers=msgspec.structs.replace(target.buffers, k=replacement)
        )
    elif bad == "draft_page":
        drafts = (info(1, page_size=32),)
    elif bad == "draft_layouts":
        drafts = (info(1, native=False),)
    else:
        drafts = (info(2),)
    with patch.object(MHATokenToKVPoolHost, "__init__") as allocated:
        with pytest.raises((ValueError, NotImplementedError)):
            make_constructor(target, drafts)
        allocated.assert_not_called()


def transfer_host():
    host = object.__new__(mha_fp4.MHATokenToKVPoolFP4Host)
    host.page_size, host.page_num, host.target_layer_num, host.layer_num = 4, 4, 2, 4
    host._scale_names = [
        "k_scale_buffer",
        "v_scale_buffer",
        "native_k_scale_buffer",
        "native_v_scale_buffer",
    ]
    host._scale_page_bytes = {name: 8 for name in host._scale_names}
    for kind, name in enumerate(host._scale_names):
        rows = [
            torch.full((4, 8), 20 * kind + layer + 1, dtype=torch.uint8)
            for layer in range(4)
        ]
        backing = torch.zeros(4, 4, 8, dtype=torch.uint8)
        setattr(host, f"_scale_rows_{name}", rows)
        setattr(host, f"_scale_host_{name}", backing)
        setattr(host, f"_scale_host_layers_{name}", list(backing.transpose(0, 1)))
        setattr(
            host,
            f"_scale_ptrs_{name}",
            torch.tensor([row.data_ptr() for row in rows], dtype=torch.uint64),
        )
        setattr(host, f"_scale_staging_{name}", torch.empty(4, 4, 8, dtype=torch.uint8))
    return host


def test_target_and_two_packed_drafts_restore_every_scale_layout_independently():
    host = transfer_host()
    registry = {
        tuple(getattr(host, f"_scale_ptrs_{name}").tolist()): getattr(
            host, f"_scale_rows_{name}"
        )
        for name in host._scale_names
    }
    expected = {
        name: [row[1:3].clone() for row in getattr(host, f"_scale_rows_{name}")]
        for name in host._scale_names
    }

    def backup(**kw):
        for layer, row in enumerate(registry[tuple(kw["ptr_src"].tolist())]):
            kw["dst"][kw["dst_indices"], layer] = row[kw["src_indices"]]

    def restore(**kw):
        kw["cache_dst"][kw["indices_dst"]] = kw["cache_src"][kw["indices_src"]]

    device_indices = torch.arange(4, 12)
    host_indices = torch.cat((torch.arange(12, 16), torch.arange(0, 4)))
    with (
        patch.object(
            MHATokenToKVPoolHost, "backup_from_device_all_layer"
        ) as payload_backup,
        patch.object(MHATokenToKVPoolHost, "load_to_device_per_layer") as payload_load,
        patch.object(
            mha_fp4,
            "jit_transfer_hicache_all_layer_mla_staged_lf_pf",
            side_effect=backup,
        ),
        patch.object(
            mha_fp4, "jit_transfer_hicache_one_layer_mla", side_effect=restore
        ),
    ):
        host.backup_from_device_all_layer(
            object(), host_indices, device_indices, "kernel"
        )
        for name in host._scale_names:
            for row in getattr(host, f"_scale_rows_{name}"):
                row.zero_()
        for layer in (2, 0, 3, 1):
            host.load_to_device_per_layer(
                object(),
                host_indices,
                device_indices,
                layer,
                "kernel",
                is_draft=layer >= 2,
            )
        assert payload_backup.call_count == 1 and payload_load.call_count == 4
    for name in host._scale_names:
        for layer, row in enumerate(getattr(host, f"_scale_rows_{name}")):
            assert torch.equal(row[1:3], expected[name][layer])
            assert torch.count_nonzero(row[0]) == torch.count_nonzero(row[3]) == 0


@pytest.mark.parametrize("direction", ["backup", "load"])
@pytest.mark.parametrize(
    "host_count,device_count,backend",
    [(3, 3, "kernel"), (4, 8, "kernel"), (4, 4, "direct")],
)
def test_invalid_transfer_is_rejected_before_payload_or_scale_copy(
    direction, host_count, device_count, backend
):
    host = transfer_host()
    name = (
        "backup_from_device_all_layer"
        if direction == "backup"
        else "load_to_device_per_layer"
    )
    with patch.object(MHATokenToKVPoolHost, name) as payload:
        with pytest.raises((ValueError, NotImplementedError)):
            args = [object(), torch.arange(host_count), torch.arange(device_count)]
            if direction == "load":
                args.append(2)
            getattr(host, name)(*args, backend)
        payload.assert_not_called()


def test_scale_memory_accounting_and_l3_rejection():
    host = transfer_host()
    host.head_dim, host.v_head_dim, host.head_num, host.dtype = 32, 32, 2, torch.uint8
    assert host.get_size_per_token() == (32 + 32) * 2 * 4 + (8 * 4 // 4) * 4
    with pytest.raises(NotImplementedError):
        host.get_data_page(0)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("host_token_capacity,expected_tokens", [(None, 112), (80, 80)])
def test_host_token_capacity_sizes_payload_and_all_scale_buffers(
    native, host_token_capacity, expected_tokens
):
    target, draft0, draft1 = info(native=native), info(1, native), info(1, native)

    def init_transfer_views(host, *args):
        # Keep real host capacity and allocations; skip device transfer setup.
        host.can_use_jit = host.can_use_write_back_jit = True
        host.staging_page_capacity = 2

    with (
        patch.object(
            MHATokenToKVPoolHost,
            "_initialize_transfer_views",
            new=init_transfer_views,
        ),
        patch.object(mha_fp4, "can_use_write_back_jit_kernel", return_value=True),
    ):
        host = make_constructor(
            target, (draft0, draft1), host_token_capacity=host_token_capacity
        )
    try:
        assert host.size == expected_tokens
        assert host.page_num == expected_tokens // 16
        assert host.kv_buffer.shape == (2, expected_tokens, 4, 2, 32)
        total_bytes = host.kv_buffer.numel() * host.kv_buffer.element_size()
        for name in host._scale_names:
            scales = getattr(host, f"_scale_host_{name}")
            assert scales.shape == (expected_tokens // 16, 4, 128)
            total_bytes += scales.numel() * scales.element_size()
        assert total_bytes == expected_tokens * host.get_size_per_token()
    finally:
        host.destroy()
