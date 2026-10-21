"""Fixed DSPARK geometry and coexistence with the compact candidate scorer."""

from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.attention.dsv4.v41_indexer import scoring
from sglang.srt.layers.attention.dsv4.v41_indexer.types import DecodeInputs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "enabled,target,mode,group,rows,compact,cp,expected",
    [
        (True, True, ForwardMode.TARGET_VERIFY, 2, 4, False, 1, 2),
        (True, True, ForwardMode.TARGET_VERIFY, 4, 8, False, 1, 4),
        (True, True, ForwardMode.TARGET_VERIFY, 6, 12, False, 1, 6),
        (False, True, ForwardMode.TARGET_VERIFY, 6, 12, False, 1, 1),
        (True, False, ForwardMode.TARGET_VERIFY, 6, 12, False, 1, 1),
        (True, True, ForwardMode.DECODE, 6, 2, False, 1, 1),
        (True, True, ForwardMode.TARGET_VERIFY, 6, 11, False, 1, 1),
        (True, True, ForwardMode.TARGET_VERIFY, 6, 12, True, 1, 1),
        (True, True, ForwardMode.TARGET_VERIFY, 6, 12, False, 2, 1),
        (True, True, ForwardMode.TARGET_VERIFY, 8, 16, False, 1, 1),
        (True, True, ForwardMode.TARGET_VERIFY, None, 2, False, 1, 1),
    ],
)
def test_only_fixed_request_major_target_rows_are_grouped(
    enabled, target, mode, group, rows, compact, cp, expected
):
    from sglang.srt.layers.attention import deepseek_v4_backend as backend_module

    metadata = object()
    backend = SimpleNamespace(
        is_dspark_target=target,
        speculative_num_draft_tokens=group,
        forward_metadata=SimpleNamespace(
            c1_indexer_metadata=metadata,
            core_metadata=SimpleNamespace(
                sparse_page_indices=lambda ratio: torch.empty(
                    rows, 8, dtype=torch.int32
                )
            ),
        ),
    )
    layer = SimpleNamespace(compress_ratio=1, indexer=None, freqs_cis=None, layer_id=20)
    batch = SimpleNamespace(forward_mode=mode, batch_size=2)
    x = torch.empty(rows, 8)
    req = torch.empty(rows, dtype=torch.int64)
    with (
        envs.SGLANG_OPT_USE_DSV41_SM90_GROUPED_INDEXER.override(enabled),
        patch.object(
            backend_module,
            "read_ragged_verify_mode",
            return_value=RaggedVerifyMode.COMPACT
            if compact
            else RaggedVerifyMode.STATIC,
        ),
        patch.object(
            backend_module,
            "get_parallel",
            return_value=SimpleNamespace(attn_cp_size=cp),
        ),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("host read")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host read")),
    ):
        inputs = DeepseekV4AttnBackend._make_low_ratio_decode_indexer_inputs(
            backend, layer, x, x, req, req, batch
        )
    assert inputs.verify_group_size == expected
    assert inputs.paged_metadata is metadata
    assert inputs.req_rows is req
    assert inputs.is_verify == mode.is_target_verify()


@pytest.mark.parametrize(
    "enabled,verify,group,heads,dim,dtype,candidates,expected",
    [
        (True, True, 6, 32, 128, torch.bfloat16, None, "grouped"),
        (True, True, 6, 64, 128, torch.bfloat16, None, "grouped"),
        (False, True, 6, 32, 128, torch.bfloat16, None, "paged"),
        (True, False, 6, 32, 128, torch.bfloat16, None, "paged"),
        (True, True, 1, 32, 128, torch.bfloat16, None, "paged"),
        (True, True, 5, 32, 128, torch.bfloat16, None, "paged"),
        (True, True, 6, 16, 128, torch.bfloat16, None, "paged"),
        (True, True, 6, 32, 64, torch.bfloat16, None, "paged"),
        (True, True, 6, 32, 128, torch.float16, None, "paged"),
        (True, True, 6, 32, 128, torch.bfloat16, "mask", "paged"),
        (True, True, 6, 32, 128, torch.bfloat16, "compact", "compact"),
    ],
)
def test_dispatch_preserves_candidates_and_fallbacks(
    enabled, verify, group, heads, dim, dtype, candidates, expected
):
    from sglang.kernels.ops.attention.dsv4 import sm90_fp4_grouped_indexer as grouped

    rows, width = 12, 131
    q = torch.empty(rows, heads, dim, dtype=dtype)
    weights = torch.empty(rows, heads, dtype=dtype)
    req = torch.arange(2).repeat_interleave(6)
    lens = torch.arange(rows) + 11
    cache = torch.empty(4, 64 * 68, dtype=torch.uint8)
    req_to_token = torch.empty(2, width * 2, dtype=torch.int32)
    indexer = SimpleNamespace(queries=lambda *_: q, head_weights=lambda _: weights)
    inputs = DecodeInputs(
        indexer=indexer,
        layer_id=2,
        compress_ratio=2,
        freqs_cis=torch.empty(2 * width, 64),
        x=torch.empty(rows, 8),
        q_lora=q,
        positions=lens * 2 - 1,
        req_rows=req,
        paged_metadata=SimpleNamespace(max_compressed_seq_len=width),
        is_verify=verify,
        verify_group_size=group,
        out_page_indices=torch.empty(rows, 8, dtype=torch.int32),
    )
    pool = SimpleNamespace(get_index_k_with_scale_buffer=lambda _: cache)
    blocks = (
        torch.zeros(rows, 2, dtype=torch.int32) if candidates == "compact" else None
    )
    mask = torch.zeros(rows, width, dtype=torch.bool) if candidates == "mask" else None
    with (
        envs.SGLANG_OPT_USE_DSV41_SM90_GROUPED_INDEXER.override(enabled),
        patch.object(
            scoring, "get_platform", return_value=SimpleNamespace(is_sm90=True)
        ),
        patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("host read")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host read")),
        patch.object(scoring, "plan_topk_v2", return_value=torch.empty(1)),
        patch.object(grouped, "fp4_index_logits_grouped_sm90") as fast,
        patch.object(scoring, "fp4_index_logits_paged") as paged,
        patch.object(scoring, "fp4_index_logits_candidates") as compact,
    ):
        result = scoring.decode_scores(
            inputs=inputs,
            token_to_kv_pool=pool,
            req_to_token=req_to_token,
            candidate_mask=mask,
            candidate_blocks=blocks,
        )
    calls = {"grouped": fast, "paged": paged, "compact": compact}
    for name, mock in calls.items():
        assert mock.call_count == int(name == expected)
    assert result.scores is calls[expected].return_value
    assert result.candidate_blocks is blocks
    if expected == "grouped":
        assert fast.call_args.args[2] is req_to_token
        assert fast.call_args.args[-4:] == (64, 2, width, group)
        torch.testing.assert_close(fast.call_args.args[4], lens.to(torch.int32))
