"""Regression coverage for NPU metadata with optional CPU length mirrors."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from test_npu_flash_mla import mod

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.fixture
def backend():
    obj = mod.AscendAttnBackend.__new__(mod.AscendAttnBackend)
    obj.use_mojo_mtp = obj.use_flash_mla = False
    obj.needs_cpu_seq_lens = False
    obj.is_hybrid_swa = obj.use_sliding_window_kv_pool = False
    obj.is_draft_worker = False
    obj.page_size = 128
    obj.speculative_num_draft_tokens = 8
    obj.speculative_step_id = 1
    obj.speculative_step_offset_npu = torch.tensor(2)
    obj.req_to_token = torch.arange(3 * 384).reshape(3, 384)
    obj.graph_metadata = {
        2: SimpleNamespace(
            block_tables=torch.full((3, 4), -99, dtype=torch.int32),
            seq_lens=torch.zeros(3, dtype=torch.int64),
        )
    }
    return obj


def apply(backend, mode, lengths, *, host_bound=None, mirror=None, speculative=False):
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                mod, "get_parallel", return_value=SimpleNamespace(dcp_enabled=False)
            )
        )
        stack.enter_context(patch.object(mod, "_is_dflash_verify", return_value=False))
        for name in ("cpu", "item", "tolist") if mirror is None else ("cpu",):
            stack.enter_context(
                patch.object(
                    torch.Tensor, name, side_effect=AssertionError("unexpected D2H")
                )
            )
        backend._apply_cuda_graph_metadata(
            bs=2,
            req_pool_indices=torch.tensor([2, 0]),
            seq_lens=lengths,
            seq_lens_cpu=mirror,
            forward_mode=mode,
            spec_info=SimpleNamespace(draft_token_num=8) if speculative else None,
            max_seq_len_host=host_bound,
        )
    return backend.forward_metadata


@pytest.mark.parametrize(
    "mode,speculative,bound,pages,offset",
    [
        (ForwardMode.DECODE, False, 0, 0, 0),
        (ForwardMode.DECODE, False, 128, 1, 0),
        (ForwardMode.DECODE, False, 384, 3, 0),
        (ForwardMode.DECODE, True, 127, 2, 2),
        (ForwardMode.TARGET_VERIFY, True, 128, 2, 8),
    ],
)
def test_mirrorless_replay_uses_host_bound_without_d2h(
    backend, mode, speculative, bound, pages, offset
):
    if not speculative:
        backend.speculative_num_draft_tokens = None
    prefix = torch.tensor([bound, min(bound, 2)])
    original = prefix.clone()
    tables = backend.graph_metadata[2].block_tables
    address = tables.data_ptr()
    metadata = apply(backend, mode, prefix, host_bound=bound, speculative=speculative)
    expected = backend.req_to_token[[2, 0], : bound + offset : 128] // 128
    torch.testing.assert_close(tables[:2, :pages], expected.int())
    assert torch.count_nonzero(tables[:2, pages:]) == 0
    assert torch.count_nonzero(tables[2:]) == 0
    torch.testing.assert_close(metadata.seq_lens[:2], prefix + offset)
    torch.testing.assert_close(prefix, original)
    assert tables.data_ptr() == address

    apply(backend, ForwardMode.DECODE, torch.tensor([32, 2]), host_bound=64)
    assert tables.data_ptr() == address
    assert torch.count_nonzero(tables[:2, 1:]) == 0


def test_mirrorless_static_fallback_clamps_to_request_table_capacity(backend):
    tables = backend.graph_metadata[2].block_tables
    apply(backend, ForwardMode.DECODE, torch.tensor([384, 2]))
    expected = backend.req_to_token[[2, 0], ::128] // 128
    torch.testing.assert_close(tables[:2, :3], expected.int())
    assert torch.count_nonzero(tables[2:]) == 0


def test_supplied_mirror_controls_verify_page_boundary_even_with_opt_out(backend):
    mirror = torch.tensor([121, 1])
    prefix = torch.tensor([120, 1])
    apply(
        backend,
        ForwardMode.TARGET_VERIFY,
        prefix,
        host_bound=384,
        mirror=mirror,
        speculative=True,
    )
    tables = backend.graph_metadata[2].block_tables
    expected = backend.req_to_token[[2, 0], :129:128] // 128
    torch.testing.assert_close(tables[:2, :2], expected.int())
    assert torch.count_nonzero(tables[:2, 2:]) == 0
    torch.testing.assert_close(mirror, torch.tensor([121, 1]))
    torch.testing.assert_close(prefix, torch.tensor([120, 1]))
