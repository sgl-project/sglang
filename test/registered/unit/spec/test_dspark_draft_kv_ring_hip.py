"""ROCm GQA DSpark draft KV ring: where the DSpark worker places draft KV is where the
window reads it (a placement and readback test; the KV writes themselves are stubbed)."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.kernels.ops.attention.dspark_ring_window_hip import (
    fill_dspark_ring_window_buffer_hip,
)
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_draft_ring import (
    DFlashDraftKVRing,
    draft_kv_ring_rows_per_req,
)
from sglang.srt.speculative.dspark_components.dspark_draft_ring_hip import (
    resolve_rocm_dspark_draft_kv_ring,
    rocm_dspark_gqa_ring_unsupported_reason,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    TargetVerifyExecutor,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="base-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(not is_hip(), reason="ROCm-only draft KV ring")

POOL, WINDOW, BLOCK = 16, 64, 6
GQA_DRAFT = SimpleNamespace(
    layer_types=["sliding_attention"] * 5,
    sliding_window=2048,
    num_hidden_layers=5,
    dflash_config={
        "mask_token_id": 163839,
        "target_layer_ids": [75, 79, 83, 87, 91],
        "aux_hidden_stream": "prefix",
    },
)


def _value(slots: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return slots.to(torch.float64) * 1e5 + positions.to(torch.float64)


def test_window_reads_what_the_dspark_worker_wrote():
    """Prefill through place_prefill and a partially accepted verify block through
    commit_hidden write the draft KV; the ring window must read back exactly those
    keys, across ring wraps, as an independent per-position store would hold them."""
    device = torch.device("cuda")
    rows = draft_kv_ring_rows_per_req(window_left=WINDOW, block_size=BLOCK, dcp_size=1)
    ring = DFlashDraftKVRing(
        rows_per_req=rows,
        dcp_size=1,
        req_to_token_rows=POOL,
        device=device,
        window_left=WINDOW,
        dcp_rank=0,
        pool=SimpleNamespace(maybe_get_custom_mem_pool=lambda: None),
    )
    kv = torch.full((1 + POOL * rows,), -1.0, dtype=torch.float64, device=device)

    def pool_ids(num_tokens):
        # Valid rows outside the two requests' ring rows: a path that writes here
        # instead of through the ring fails on the values read back.
        return lambda iteration_ids: torch.arange(
            1, 1 + num_tokens, dtype=torch.int64, device=device
        )

    slots = torch.tensor([3, 9], dtype=torch.int64, device=device)
    # Request 3's prefill wraps the ring twice; request 9's does not wrap.
    prefill_lens = [2 * rows + 17, 40]

    injector = SimpleNamespace(
        model_runner=SimpleNamespace(
            device=device,
            prefill_attention_backend_str="triton",
            runner_ids=SimpleNamespace(own_plan=lambda batch: MagicMock()),
        ),
        draft_model_runner=SimpleNamespace(token_to_kv_pool=object(), runner_ids=None),
        swa_ring_size=None,
        draft_kv_ring=ring,
        ids_for=pool_ids(sum(prefill_lens)),
    )
    prefill = SimpleNamespace(
        extend_lens=prefill_lens,
        prefix_lens=[0, 0],
        out_cache_loc=torch.zeros(1, dtype=torch.int64, device=device),
        req_pool_indices=slots,
    )
    placement = TargetHiddenKvInjector.place_prefill(injector, prefill)
    token_slots = torch.repeat_interleave(
        slots, torch.tensor(prefill_lens, device=device)
    )
    kv[placement.cache_loc] = _value(token_slots, placement.positions)

    # A verify block past each prefill, of which the first 2 tokens commit.
    seq_lens = torch.tensor(prefill_lens, dtype=torch.int64, device=device)
    positions_2d = seq_lens[:, None] + torch.arange(BLOCK, device=device)
    commit_lens = torch.tensor([2, 2], dtype=torch.int64, device=device)

    def inject_target_hidden(*, target_hidden, cache_loc_2d, commit_lens, **_):
        hidden = target_hidden.view(2, BLOCK)
        for b in range(2):
            kv[cache_loc_2d[b, : commit_lens[b]]] = hidden[b, : commit_lens[b]]

    executor = SimpleNamespace(
        kv_injector=SimpleNamespace(
            swa_ring_size=None,
            draft_kv_ring=ring,
            inject_target_hidden=inject_target_hidden,
            ids_for=pool_ids(2 * BLOCK),
        ),
        verify_num_draft_tokens=BLOCK,
    )
    hidden = _value(slots[:, None].expand(2, BLOCK), positions_2d)
    TargetVerifyExecutor.commit_hidden(
        executor,
        batch=SimpleNamespace(req_pool_indices=slots),
        layout=None,
        hidden_strided=None,
        verify_window=SimpleNamespace(
            positions_2d=positions_2d,
            verify_cache_loc_2d=torch.empty(2, BLOCK, device=device),
            iteration_ids=None,
        ),
        logits_output=SimpleNamespace(hidden_states=hidden.reshape(-1, 1)),
        commit_lens=commit_lens,
        bs=2,
        run_compact=False,
    )

    seq_lens = seq_lens + commit_lens
    indptr, indices, _, starts = fill_dspark_ring_window_buffer_hip(
        min_pos=ring.min_pos,
        rows_per_req=ring.rows_per_req,
        dcp_size=1,
        window_kv_indptr=torch.zeros(3, dtype=torch.int32, device=device),
        req_pool_indices=slots,
        seq_lens=seq_lens,
        bs=2,
        sliding_window_size=WINDOW,
    )
    for b in range(2):
        positions = torch.arange(int(starts[b]), int(seq_lens[b]), device=device)
        assert positions.numel() == min(WINDOW, int(seq_lens[b]))
        read = kv[indices[indptr[b] : indptr[b + 1]]]
        expected = _value(slots[b].expand(positions.numel()), positions)
        assert torch.equal(read, expected)


def test_gqa_ring_pool_page_size_is_not_widened_by_target_dcp():
    """A private GQA ring must use the page width charged by its target."""
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

    configurator = object.__new__(KVCacheConfigurator)
    configurator.is_draft_worker = True
    configurator.use_mla_backend = False
    configurator.model_config = SimpleNamespace(hf_config=GQA_DRAFT)
    for ring_enabled, expected_scale in ((False, 8), (True, 1)):
        with (
            get_context().override_server_args(
                tp_size=8,
                dcp_size=8,
                page_size=16,
                speculative_dflash_draft_kv_ring=ring_enabled,
            ),
            get_parallel().override(attn_dcp_size=8),
        ):
            assert configurator.loc_space_scale == expected_scale
            assert configurator.pool_page_size == 16 * expected_scale


def _reason(**overrides):
    kwargs = dict(
        draft_backend="triton", disaggregation_mode="null", draft_window_size=None
    )
    kwargs.update(overrides)
    return rocm_dspark_gqa_ring_unsupported_reason(lambda: GQA_DRAFT, **kwargs)


def test_triton_gqa_ring_is_allowed():
    assert _reason() is None


def test_ring_is_rejected_where_it_has_no_reader_or_writer():
    """Only ROCm's Triton draft backend reads the ring, PD does not send it, and
    unified memory puts the draft KV in the target's pool."""
    for overrides in (
        dict(draft_backend="aiter"),
        dict(draft_backend="flashinfer"),
        dict(disaggregation_mode="prefill"),
        dict(enable_unified_memory=True),
        dict(draft_window_size=2048),
    ):
        assert _reason(**overrides) is not None, overrides


def test_ring_needs_static_ragged_verify():
    """Compact and cap-accept verify inject draft KV by pool ids, not ring rows."""
    for mode in ("compact", "cap-accept"):
        with envs.SGLANG_RAGGED_VERIFY_MODE.override(mode):
            assert "SGLANG_RAGGED_VERIFY_MODE" in _reason()


def _resolve(*, dspark, dflash=False):
    args = ServerArgs(
        model_path="dummy",
        speculative_algorithm="DSPARK",
        speculative_draft_model_path="draft",
        speculative_num_draft_tokens=8,
        speculative_draft_attention_backend="triton",
        dcp_size=8,
    )
    with (
        envs.SGLANG_ENABLE_DSPARK_DRAFT_KV_RING.override(dspark),
        envs.SGLANG_ENABLE_DFLASH_DRAFT_KV_RING.override(dflash),
    ):
        resolve_rocm_dspark_draft_kv_ring(args, lambda cfg: GQA_DRAFT)
    return resolution_result(args, "speculative_dflash_draft_kv_ring")


def test_dspark_switch_turns_the_ring_on():
    assert _resolve(dspark=True)


def test_dflash_switch_alone_leaves_the_dspark_ring_off():
    """The DFLASH ring switch must not turn on the DSpark ring."""
    assert not _resolve(dspark=False, dflash=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
