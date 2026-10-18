"""MegaMoE capacity selection must keep one cross-rank wire protocol.

Exercises the real host dispatch and instance cache in ``mega_moe_flydsl`` with
CPU tensors, and the startup capacity check in ``mega_moe_hook``. Only the
AITER constructor/kernel and the per-forward DP context are replaced; GPU
collectives and graph replay are out of scope.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import sglang.srt.arg_groups.mega_moe_hook as hook
import sglang.srt.layers.moe.mega_moe_flydsl as mega
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

LARGE = 16384
DECODE = 512
EXPERTS = 384
TOPK = 6
HIDDEN = 8


def _layer(value=0):
    return SimpleNamespace(
        num_experts=EXPERTS,
        should_fuse_routed_scaling_factor_in_topk=True,
        **{
            name: torch.full((48, 4), value, dtype=torch.uint8)
            for name in ("_mega_w1", "_mega_w1_scale", "_mega_w2", "_mega_w2_scale")
        },
    )


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(
        rank=0,
        capture=False,
        extend=False,
        global_tokens=None,
        events=[],
        instances={},
    )
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR", str(LARGE))
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR", str(DECODE))
    monkeypatch.setenv("SGLANG_AITER_MEGA_RANK_SYNC", "1")
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_QUANT", "a8w4")
    monkeypatch.setattr(mega, "_MEGA_MOE_INSTANCE", {})
    mega._warn_decode_mtpr_merged.cache_clear()
    # CPU-only PyTorch raises from the real query.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(mega, "get_is_capture_mode", lambda: state.capture)
    monkeypatch.setattr(mega, "get_is_extend_in_batch", lambda: state.extend)
    monkeypatch.setattr(mega, "get_dp_global_num_tokens", lambda: state.global_tokens)
    monkeypatch.setattr(mega, "_ensure_mori_shmem", lambda: None)
    monkeypatch.setattr(mega, "_ep_rank_world", lambda: (state.rank, 8))
    monkeypatch.setattr(
        mega,
        "ExpertLocationDispatchInfo",
        SimpleNamespace(init_new=lambda layer_id: None),
    )
    utils = ModuleType("sglang.srt.models.deepseek_common.utils")
    utils._use_aiter = True
    monkeypatch.setitem(sys.modules, utils.__name__, utils)

    class FakeMegaMoEV2:
        def __init__(self, **kwargs):
            assert not torch.cuda.is_current_stream_capturing()
            self.rank = kwargs["rank"]
            self.capacity = kwargs["max_tok_per_rank"]
            state.events.append(("build", self.rank, self.capacity))
            state.instances[self.rank, self.capacity] = self

        def forward(self, x, weights, ids, slice_output=False):
            state.events.append(("forward", self.rank, self.capacity))
            return torch.full_like(x, self.capacity)

    monkeypatch.setattr(
        mega, "_import_aiter_megamoe", lambda: SimpleNamespace(op=FakeMegaMoEV2)
    )

    def run(
        local_rows,
        global_tokens=None,
        *,
        rank=0,
        mode=ForwardMode.DECODE,
        extend=False,
        capture=False,
        layer=None,
    ):
        state.rank = rank
        state.extend = extend
        state.capture = capture
        state.global_tokens = global_tokens
        moe = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=HIDDEN,
                num_experts_per_tok=TOPK,
                moe_intermediate_size=HIDDEN,
                swiglu_limit=0.0,
            ),
            num_fused_shared_experts=0,
            experts=state.layer if layer is None else layer,
            gate=lambda h, forward_batch=None: torch.zeros(h.shape[0], EXPERTS),
            topk=lambda h, logits, **kwargs: SimpleNamespace(
                topk_ids=torch.zeros(h.shape[0], TOPK, dtype=torch.int64),
                topk_weights=torch.ones(h.shape[0], TOPK),
            ),
            is_hash=False,
            layer_id=0,
            routed_scaling_factor=1.0,
        )
        forward_batch = (
            None
            if mode is None
            else SimpleNamespace(forward_mode=mode, num_token_non_padded=None)
        )
        hidden = torch.zeros(local_rows, HIDDEN, dtype=torch.float32)
        return mega._run_mega_routed(moe, hidden, forward_batch, None)

    state.layer = _layer()
    state.run = run
    return state


def _built(runtime, rank=0):
    return [c for event, pe, c in runtime.events if event == "build" and pe == rank]


def _forwarded(runtime):
    return [c for event, _, c in runtime.events if event == "forward"]


def _run_step(runtime, ranks):
    for rank, kwargs in enumerate(ranks):
        runtime.run(rank=rank, **kwargs)
    return _forwarded(runtime)


@pytest.mark.parametrize("capture", [False, True])
def test_rank_local_rows_across_the_decode_capacity_fail_instead_of_splitting(
    runtime, capture
):
    # EP8 / DP8 with dense TP1 and the DP LM head gathers neither DP token
    # counts nor a DP-max graph bucket, so one rank can decode 128 rows while
    # another decodes 1024. Startup rejects such configs; if one slips through,
    # the overflowing rank must fail rather than run the large capacity alone.
    mode = ForwardMode.TARGET_VERIFY
    runtime.run(128, [128], rank=0, mode=mode, capture=capture)
    with pytest.raises(AssertionError, match=f"exceed MTPR {DECODE}"):
        runtime.run(1024, [1024], rank=1, mode=mode, capture=capture)
    assert _forwarded(runtime) == [DECODE]


def test_decode_step_ignores_rank_local_counts_and_graph_buckets(runtime):
    # Within the decode capacity: each rank sees only its own count, replays
    # the bucket of its own batch, or runs eager beside the graph replays.
    step = [
        dict(local_rows=7, global_tokens=[7]),
        dict(local_rows=480, global_tokens=[480]),
        dict(local_rows=56, mode=ForwardMode.TARGET_VERIFY, capture=True),
        dict(local_rows=448, mode=ForwardMode.TARGET_VERIFY, capture=True),
        dict(local_rows=448, global_tokens=[448], mode=ForwardMode.TARGET_VERIFY),
        dict(local_rows=0, global_tokens=[0], mode=ForwardMode.IDLE),
        dict(local_rows=0, mode=ForwardMode.IDLE, capture=True),
        dict(local_rows=7, global_tokens=[7, 480, 56, 448, 448, 0, 0, 7]),
    ]
    assert _run_step(runtime, step) == [DECODE] * len(step)


def test_draft_extend_and_idle_ranks_share_the_decode_capacity(runtime):
    # The draft extend after a verify runs DRAFT_EXTEND_V2 on ranks with
    # requests, eager or from its graph, and IDLE on the rest.
    step = [
        dict(local_rows=56, mode=ForwardMode.DRAFT_EXTEND_V2, capture=True),
        dict(local_rows=448, global_tokens=[448], mode=ForwardMode.DRAFT_EXTEND_V2),
        dict(local_rows=0, global_tokens=[0], mode=ForwardMode.IDLE),
    ]
    assert _run_step(runtime, step) == [DECODE] * len(step)


def test_an_extend_on_any_rank_moves_the_step_to_the_large_capacity(runtime):
    # is_extend_in_batch is max-reduced, and decode graphs do not replay in
    # this step; a prefill graph does, captured with the flag cleared.
    prefill_graph = dict(mode=ForwardMode.EXTEND, capture=True)
    step = [
        dict(local_rows=128, global_tokens=[128] * 8, **prefill_graph),
        dict(local_rows=4096, global_tokens=[4096], mode=ForwardMode.EXTEND),
        dict(local_rows=7, global_tokens=[7]),
        dict(local_rows=448, global_tokens=[448], mode=ForwardMode.TARGET_VERIFY),
        dict(local_rows=448, global_tokens=[448], mode=ForwardMode.DRAFT_EXTEND_V2),
        dict(local_rows=0, global_tokens=[0], mode=ForwardMode.IDLE),
    ]
    for kwargs in step[1:]:
        kwargs["extend"] = True
    assert _run_step(runtime, step) == [LARGE] * len(step)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize(
    "mode",
    [
        ForwardMode.EXTEND,
        ForwardMode.MIXED,
        ForwardMode.SPLIT_PREFILL,
        ForwardMode.DLLM_EXTEND,
        None,
    ],
)
def test_prefill_forwards_keep_the_large_capacity(runtime, mode, capture):
    runtime.run(128, [128] * 8, mode=mode, capture=capture)
    assert _forwarded(runtime) == [LARGE]


def test_idle_rank_enters_with_the_decode_capacity(runtime):
    output = runtime.run(0, [0], rank=7, mode=ForwardMode.IDLE)

    assert _forwarded(runtime) == [DECODE]
    assert output.shape == (0, HIDDEN)


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.EXTEND])
def test_first_call_builds_every_capacity_in_one_order(runtime, mode):
    for rank in range(2):
        runtime.run(1, [1], rank=rank, mode=mode, extend=mode.is_extend())
    assert _built(runtime, 0) == _built(runtime, 1) == [DECODE, LARGE]

    runtime.run(1, capture=True)
    runtime.run(1, [1], mode=ForwardMode.EXTEND, extend=True)
    assert _built(runtime, 0) == [DECODE, LARGE]


def test_each_use_rebinds_the_layer_weights_without_copying(runtime):
    runtime.run(1, [1], mode=ForwardMode.EXTEND, extend=True)
    next_layer = _layer(value=7)
    runtime.run(1, [1], layer=next_layer)

    decode = runtime.instances[0, DECODE]
    assert decode._s1_w1.data_ptr() == next_layer._mega_w1.data_ptr()
    assert decode._s1_w1_scale.data_ptr() == next_layer._mega_w1_scale.data_ptr()
    assert decode.w2 is next_layer._mega_w2
    assert decode.w2_scale is next_layer._mega_w2_scale

    # EPLB migrates expert rows in place; the bound views must observe it.
    next_layer._mega_w1[0].fill_(9)
    assert torch.all(decode._s1_w1[0] == 9)


@pytest.mark.parametrize("decode_mtpr", ["0", str(LARGE)])
def test_disabled_decode_capacity_keeps_one_instance(runtime, monkeypatch, decode_mtpr):
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR", decode_mtpr)
    runtime.run(1, [1] * 8)

    assert _built(runtime) == [LARGE]
    assert _forwarded(runtime) == [LARGE]


def test_decode_capacity_equal_to_the_large_one_warns_once(
    runtime, monkeypatch, caplog
):
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR", str(LARGE))
    with caplog.at_level("WARNING", logger=mega.logger.name):
        runtime.run(1, [1] * 8)
        runtime.run(1, [1] * 8)
    assert [r.levelname for r in caplog.records].count("WARNING") == 1


@pytest.mark.parametrize("decode_mtpr", ["-1", "384", str(2 * LARGE)])
def test_invalid_decode_capacity_is_rejected(runtime, monkeypatch, decode_mtpr):
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR", decode_mtpr)
    with pytest.raises(ValueError, match="power of two no larger than"):
        runtime.run(1, [1] * 8)


@pytest.mark.xfail(
    strict=True,
    reason="no guard yet: a capacity first built inside graph capture is not rejected",
)
def test_capture_cache_miss_is_rejected_before_symmetric_allocation(
    runtime, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before graph capture"):
        runtime.run(1, None, capture=True)
    assert runtime.events == []


def _cfg(**overrides):
    # Recipe c128 EP8 arm: DP8, DSpark with 7 draft tokens.
    cfg = dict(
        disaggregation_mode="null",
        attn_cp_size=1,
        attn_dp_size=8,
        ep_join_mode=None,
        max_running_requests=256,
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=7,
        speculative_eagle_topk=1,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


@pytest.fixture
def startup(monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_USE_FLYDSL_MEGA_MOE", "1")
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR", str(LARGE))
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR", str(DECODE))
    monkeypatch.setattr(hook, "get_platform", lambda: SimpleNamespace(is_hip=True))
    monkeypatch.setattr(hook, "resolving_view", lambda server_args: server_args)
    return hook.validate_flydsl_decode_capacity


@pytest.mark.parametrize(
    ("overrides", "rows"),
    [
        ({}, 32 * 7),
        (dict(attn_dp_size=1), 256 * 7),
        (dict(speculative_algorithm=None), 32),
        (
            dict(
                speculative_algorithm="EAGLE",
                speculative_num_draft_tokens=4,
                speculative_eagle_topk=8,
            ),
            32 * 8,
        ),
        (dict(max_running_requests=None), None),
    ],
)
def test_decode_rows_per_rank(overrides, rows):
    assert hook.flydsl_decode_tokens_per_rank(_cfg(**overrides)) == rows


@pytest.mark.parametrize(
    ("max_running_requests", "decode_mtpr"),
    # Recipe c128 / c256 / c384 arms: 224 / 448 / 672 rows per rank.
    [(256, 512), (512, 512), (768, 1024)],
)
def test_startup_accepts_a_decode_capacity_that_holds_every_rank(
    startup, monkeypatch, max_running_requests, decode_mtpr
):
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR", str(decode_mtpr))
    startup(_cfg(max_running_requests=max_running_requests))


def test_startup_rejects_a_decode_capacity_one_rank_can_overflow(startup):
    # One rank may then decode 1024 rows beside another's 128.
    with pytest.raises(ValueError, match="at least 1024"):
        startup(_cfg(max_running_requests=8 * 1024, speculative_algorithm=None))


def test_startup_needs_a_request_cap(startup):
    with pytest.raises(ValueError, match="--max-running-requests"):
        startup(_cfg(max_running_requests=None))


def test_startup_rejects_attention_context_parallelism(startup):
    with pytest.raises(ValueError, match="context parallelism"):
        startup(_cfg(attn_cp_size=2))


@pytest.mark.parametrize(
    ("env", "overrides"),
    [
        ({}, dict(disaggregation_mode="prefill")),
        ({"SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR": "0"}, {}),
        ({"SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR": str(LARGE)}, {}),
        ({"SGLANG_AMD_USE_FLYDSL_MEGA_MOE": "0"}, {}),
    ],
)
def test_startup_skips_runs_without_a_decode_capacity(
    startup, monkeypatch, env, overrides
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    startup(_cfg(max_running_requests=None, **overrides))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
