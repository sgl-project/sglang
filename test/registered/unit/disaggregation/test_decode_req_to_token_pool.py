import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.disaggregation.decode import (
    DecodeReqToTokenPool,
    HybridMambaDecodeReqToTokenPool,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.utils import max_verify_batch_size
from sglang.srt.layers.dp_attention import get_eager_max_batch_size
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpPLELayer,
    _commit_ple_batch,
    _prepare_ple_batch,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _init_decode_pool(pool):
    DecodeReqToTokenPool.__init__(
        pool,
        size=1,
        row_width=4,
        device="cpu",
        enable_memory_saver=False,
        pre_alloc_size=1,
    )
    return pool


def test_decode_pool_supports_noop_aux_cache_contract():
    pool = _init_decode_pool(DecodeReqToTokenPool.__new__(DecodeReqToTokenPool))
    req_to_token = pool.req_to_token.clone()

    pool.alloc_aux_to_lengths(
        req_pool_indices_cpu=torch.tensor([1]),
        target_seq_lens_cpu=torch.tensor([3]),
    )
    pool.reset_aux_cache_allocator()

    assert torch.equal(pool.req_to_token, req_to_token)


@pytest.mark.parametrize(
    "pool_cls, pool_kwargs",
    [(ReqToTokenPool, {}), (DecodeReqToTokenPool, {"pre_alloc_size": 0})],
    ids=["ReqToTokenPool", "DecodeReqToTokenPool"],
)
def test_request_pool_clear_keeps_row_generations_monotonic(pool_cls, pool_kwargs):
    """Readers that stash req_generation[row] detect row reuse by inequality with
    the live value. clear() (flush_cache) must not reset the counter: the row's
    next request would otherwise match the generation its previous request had."""
    pool = pool_cls(
        size=2,
        row_width=4,
        device="cpu",
        enable_memory_saver=False,
        **pool_kwargs,
    )

    def start_request() -> int:
        (row,) = pool.alloc([SimpleNamespace(kv=ReqKvInfo())])
        return row

    row = start_request()
    stored_generation = pool.req_generation[row].item()

    pool.clear()

    assert start_request() == row
    assert pool.req_generation[row].item() != stored_generation


def _decode_mamba_spec_state_size(*, size, pre_alloc_size, verify_rows, replayssm_spec):
    captured = {}

    def capture(self, *, mamba_spec_state_size, **kwargs):
        captured["rows"] = mamba_spec_state_size

    with patch.object(HybridMambaDecodeReqToTokenPool, "_init_mamba_pool", capture):
        HybridMambaDecodeReqToTokenPool(
            size=size,
            row_width=4,
            device="cpu",
            enable_memory_saver=False,
            cache_params=None,
            mamba_layer_ids=[0],
            speculative_num_draft_tokens=6,
            enable_mamba_extra_buffer=False,
            pre_alloc_size=pre_alloc_size,
            enable_overlap_schedule=True,
            mamba_size=size + pre_alloc_size,
            enable_linear_replayssm_spec=replayssm_spec,
            max_verify_batch_size=verify_rows,
        )
    return captured["rows"]


@pytest.mark.parametrize(
    "size, pre_alloc_size, verify_rows, replayssm_spec, pp_spec_stable_rows, rows",
    [
        (8, 16, 8, False, False, 8),
        (6, 12, 8, False, False, 8),
        (8, 16, 8, True, False, 8 + 16),
        (8, 16, 8, False, True, 8 + 16),
        (6, 0, 8, True, False, 8),
    ],
    ids=[
        "batch_rows",
        "padded_batch_rows",
        "replayssm_spec_ring",
        "pp_spec_stable_rows",
        "replayssm_spec_padded_batch_rows",
    ],
)
def test_decode_mamba_spec_scratch_rows(
    size, pre_alloc_size, verify_rows, replayssm_spec, pp_spec_stable_rows, rows
):
    """The spec scratch covers the padded verify batch, not in-transfer slots,
    unless verify indexes it by req_pool_idx (ReplaySSM-spec, PP-spec rows)."""
    with envs.SGLANG_ENABLE_PP_SPEC.override(pp_spec_stable_rows):
        spec_rows = _decode_mamba_spec_state_size(
            size=size,
            pre_alloc_size=pre_alloc_size,
            verify_rows=verify_rows,
            replayssm_spec=replayssm_spec,
        )
    assert spec_rows == rows


_PLE_CHANNELS, _PLE_STATE_LEN = 3, 4


def _ple_decode_pool(*, size, pre_alloc_size, verify_rows, draft_tokens):
    cache_params = Mamba2CacheParams(
        shape=Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=8,
            n_groups=1,
            num_heads=2,
            head_dim=4,
            state_size=4,
            conv_kernel=4,
        ),
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
        layers=[0],
    )
    return HybridMambaDecodeReqToTokenPool(
        size=size,
        row_width=16,
        device="cpu",
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=[0],
        speculative_num_draft_tokens=draft_tokens,
        speculative_eagle_topk=1,
        enable_mamba_extra_buffer=False,
        pre_alloc_size=pre_alloc_size,
        enable_overlap_schedule=True,
        short_conv_layer_ids=[0],
        short_conv_state_shape=(_PLE_CHANNELS, _PLE_STATE_LEN),
        ngram_context_len=2,
        ngram_eos_token_id=0,
        max_verify_batch_size=verify_rows,
    )


@pytest.mark.parametrize(
    "pre_alloc_size", [4, 0], ids=["default_extra_slots", "no_extra_slots"]
)
def test_decode_ple_verify_writes_fit_padded_batch(pre_alloc_size):
    """PLE verify writes index the scratch by padded batch position;
    the decode pool must hold those rows with or without in-transfer slots."""
    size, draft_tokens = 2, 4
    with (
        patch(
            "sglang.srt.layers.dp_attention.get_parallel",
            return_value=SimpleNamespace(attn_dp_enabled=True, attn_tp_size=4),
        ),
        patch("sglang.srt.layers.cp.padding.get_cp_padding_align_size", return_value=1),
        patch(
            "sglang.srt.layers.dp_attention.get_cuda_graph_batch_size_alignment",
            return_value=4,
        ),
    ):
        eager_rows = get_eager_max_batch_size(size)
        verify_rows = max_verify_batch_size(size)
    assert eager_rows == verify_rows == 4

    with envs.SGLANG_ENABLE_PP_SPEC.override(False):
        pool = _ple_decode_pool(
            size=size,
            pre_alloc_size=pre_alloc_size,
            verify_rows=verify_rows,
            draft_tokens=draft_tokens,
        )
        router = SimpleNamespace(
            read=lambda layer_id, part: SimpleNamespace(
                view=pool.short_conv_pool, view_layer_id=layer_id
            ),
            wait_for_part=lambda part: None,
            part_view=lambda part: pool.ngram_pool,
        )
        backend = SimpleNamespace(
            req_to_token_pool=pool,
            token_to_kv_pool=SimpleNamespace(layer_router=router),
            # Padded rows resolve to the dummy state slot.
            forward_state_slots=lambda rows, _: SimpleNamespace(
                working=torch.zeros(rows, dtype=torch.int32)
            ),
        )
        tokens = verify_rows * draft_tokens
        fb = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            _original_forward_mode=None,
            tbo_parent_token_range=None,
            spec_algorithm=None,
            spec_info=SimpleNamespace(topk=1, draft_token_num=draft_tokens),
            global_num_token_non_padded_cpu=None,
            extend_seq_lens=None,
            extend_seq_lens_cpu=None,
            req_pool_indices=torch.zeros(verify_rows, dtype=torch.int64),
            out_cache_loc=torch.zeros(tokens, dtype=torch.int64),
        )
        ple = SimpleNamespace(
            layer_id=0,
            conv_channels=_PLE_CHANNELS,
            short_conv_dilation=1,
            short_conv_state_len=_PLE_STATE_LEN,
            conv1d=SimpleNamespace(
                weight=torch.ones(_PLE_CHANNELS, 1, _PLE_STATE_LEN + 1)
            ),
        )
        with forward_context(ForwardContext(attn_backend=backend)):
            batch = _prepare_ple_batch(
                torch.zeros(tokens, dtype=torch.int64),
                fb,
                ngram_size=3,
                ngram_eos_token_id=0,
            )
            Qwen4ExpPLELayer._short_conv(
                ple, torch.zeros(tokens, _PLE_CHANNELS), fb, batch
            )
            _commit_ple_batch(batch, fb)

    assert batch.lengths.shape[0] == verify_rows


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
