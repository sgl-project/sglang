"""Replay packing contracts; real-model and RequestWindow checks are separate."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.encoder_swa_replay import (
    _validate_batch_runtime,
    run_encoder_swa_replay,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_case(prefixes, resets, *, device="cpu", ngram_size=4):
    slots = [5, 1, 4, 2, 3][: len(prefixes)]
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        encoder_swa_reset=resets,
        reqs=[
            SimpleNamespace(
                full_untruncated_fill_ids=[i * 10000 + p for p in range(8500)]
            )
            for i in range(len(prefixes))
        ],
        req_pool_indices=torch.tensor(slots, device=device),
        req_pool_indices_cpu=torch.tensor(slots),
        prefix_lens=prefixes,
        extend_lens=[4096] * len(prefixes),
        input_ids=torch.tensor([99], device=device),
        seq_lens=torch.tensor([99] * len(prefixes), device=device),
        return_logprob=True,
        has_grammar=True,
        sampling_info=object(),
        spec_info=object(),
        engram_history=object(),
    )
    window = SimpleNamespace(reset=Mock())
    table = torch.arange(6 * 8500, device=device, dtype=torch.int32).reshape(6, 8500)
    runner = SimpleNamespace(
        device=device,
        token_to_kv_pool=SimpleNamespace(request_window=window),
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        model=SimpleNamespace(
            model=SimpleNamespace(
                engram_hasher=(
                    SimpleNamespace(max_ngram_size=ngram_size) if ngram_size else None
                )
            )
        ),
        forward=Mock(),
    )
    return SimpleNamespace(model_runner=runner), batch


def run_case(worker, batch, *, budget=8192, max_batch_size=64):
    replays = []

    def construct(replay, runner, **kwargs):
        assert runner is worker.model_runner
        assert kwargs == {
            "capture_hidden_mode": CaptureHiddenMode.NULL,
            "return_hidden_states_before_norm": False,
        }
        replays.append(replay)
        return SimpleNamespace()

    with (
        patch("sglang.srt.model_executor.encoder_swa_replay._validate_batch_runtime"),
        patch(
            "sglang.srt.runtime_context.get_schedule",
            return_value=SimpleNamespace(chunked_prefill_size=budget),
        ),
        patch.object(
            envs.SGLANG_ENCODER_SWA_REPLAY_MAX_BATCH_SIZE,
            "get",
            return_value=max_batch_size,
        ),
        patch.object(ForwardBatch, "init_new", side_effect=construct),
    ):
        run_encoder_swa_replay(worker, batch)
    return replays


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_noncontiguous_requests_keep_their_tokens_slots_and_ngram_history(device):
    worker, batch = make_case(
        [4096, 0, 64, 8192, 4096], [True, True, True, True, False], device=device
    )
    before = vars(batch).copy()
    (replay,) = run_case(worker, batch)
    assert replay.reqs == [batch.reqs[i] for i in (0, 2, 3)]
    assert replay.prefix_lens == [3968, 0, 8064]
    assert replay.extend_lens == [128, 64, 128]
    assert replay.extend_num_tokens == 320
    assert replay.seq_lens.tolist() == [4096, 64, 8192]
    assert replay.seq_lens_cpu.tolist() == [4096, 64, 8192]
    assert replay.seq_lens_sum == 12352
    assert replay.req_pool_indices.tolist() == [5, 4, 2]
    assert replay.req_pool_indices_cpu.tolist() == [5, 4, 2]
    expected_ids = (
        list(range(3968, 4096)) + list(range(20000, 20064)) + list(range(38064, 38192))
    )
    assert replay.input_ids.tolist() == expected_ids
    table = worker.model_runner.req_to_token_pool.req_to_token
    expected_loc = torch.cat(
        (table[5, 3968:4096], table[4, :64], table[2, 8064:8192])
    ).long()
    assert torch.equal(replay.out_cache_loc, expected_loc)
    assert replay.engram_history.tolist() == [
        [3965, 3966, 3967],
        [0, 0, 0],
        [38061, 38062, 38063],
    ]
    assert replay.multimodal_inputs == [None, None, None]
    assert replay.extend_logprob_start_lens == replay.extend_lens
    assert (
        not replay.return_logprob and replay.is_prefill_only and not replay.has_grammar
    )
    assert replay.sampling_info is None and replay.spec_info is None
    assert worker.model_runner.forward.call_count == 1
    assert worker.model_runner.forward.call_args.args[0].encoder_swa_replay
    assert worker.model_runner.token_to_kv_pool.request_window.reset.call_args.args[
        0
    ].tolist() == [5, 1, 4, 2]
    assert vars(batch).keys() == before.keys()
    for key, value in before.items():
        assert getattr(batch, key) is value, key


@pytest.mark.parametrize(
    "budget,expected",
    [(128, [[128], [64], [128]]), (256, [[128, 64], [128]]), (320, [[128, 64, 128]])],
)
def test_replay_tokens_are_bounded_independently_of_suffix_lengths(budget, expected):
    worker, batch = make_case([4096, 64, 8192], [True] * 3)
    batch.extend_lens = [2] * 3
    replays = run_case(worker, batch, budget=budget)
    assert [r.extend_lens for r in replays] == expected
    assert all(r.extend_num_tokens <= budget for r in replays)
    assert worker.model_runner.forward.call_count == len(expected)


@pytest.mark.parametrize("budget", [None, -1, 0, 64])
def test_disabled_or_small_chunks_keep_a_complete_replay_tail(budget):
    worker, batch = make_case([4096, 4096], [True, True])
    replays = run_case(worker, batch, budget=budget)
    assert [r.extend_lens for r in replays] == [[128], [128]]


def test_unsupported_grouped_runtime_fails_before_window_reset():
    worker, batch = make_case([4096, 4096], [True, True])
    with (
        envs.SGLANG_ENCODER_SWA_REPLAY_MAX_BATCH_SIZE.override(64),
        patch(
            "sglang.srt.model_executor.encoder_swa_replay._validate_batch_runtime",
            side_effect=NotImplementedError("unsupported runtime"),
        ),
        pytest.raises(NotImplementedError, match="unsupported runtime"),
    ):
        run_encoder_swa_replay(worker, batch)
    worker.model_runner.token_to_kv_pool.request_window.reset.assert_not_called()
    worker.model_runner.forward.assert_not_called()


def test_default_batch_size_preserves_singleton_forwards():
    assert envs.SGLANG_ENCODER_SWA_REPLAY_MAX_BATCH_SIZE.default == 1
    worker, batch = make_case([4096, 4096], [True, True])
    replays = run_case(worker, batch, max_batch_size=1)
    assert [r.extend_lens for r in replays] == [[128], [128]]


def test_batch_size_limit_is_independent_of_token_budget():
    worker, batch = make_case([64, 64, 64], [True] * 3)
    replays = run_case(worker, batch, max_batch_size=2)
    assert [r.extend_lens for r in replays] == [[64, 64], [64]]


def test_invalid_batch_size_fails_before_reset():
    worker, batch = make_case([4096], [True])
    with pytest.raises(ValueError, match="max batch size must be positive"):
        run_case(worker, batch, max_batch_size=0)
    worker.model_runner.token_to_kv_pool.request_window.reset.assert_not_called()


@pytest.mark.parametrize(
    "prefix,expected", [(64, [0, 0, 0]), (128, [0, 0, 0]), (130, [0, 0, 1])]
)
def test_short_prefix_ngram_padding(prefix, expected):
    worker, batch = make_case([prefix], [True])
    (replay,) = run_case(worker, batch)
    assert replay.engram_history.tolist() == [expected]


def test_no_hasher_needs_no_ngram_history():
    worker, batch = make_case([4096, 4096], [True, True], ngram_size=None)
    (replay,) = run_case(worker, batch)
    assert replay.engram_history is None


@pytest.mark.parametrize(
    "prefixes,resets", [([0, 0], [True, True]), ([4096, 4096], [False, False])]
)
def test_zero_prefix_resets_without_replay_and_live_windows_are_preserved(
    prefixes, resets
):
    worker, batch = make_case(prefixes, resets)
    assert run_case(worker, batch) == []
    assert worker.model_runner.forward.call_count == 0
    assert worker.model_runner.token_to_kv_pool.request_window.reset.call_count == int(
        any(resets)
    )


def test_odd_boundary_fails_before_resetting_any_window():
    worker, batch = make_case([4096, 4095], [True, True])
    with pytest.raises(ValueError, match="even cached-prefix boundary"):
        run_case(worker, batch)
    worker.model_runner.forward.assert_not_called()
    worker.model_runner.token_to_kv_pool.request_window.reset.assert_not_called()


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.TARGET_VERIFY])
def test_non_extend_modes_do_not_reset(mode):
    worker, batch = make_case([4096], [True])
    batch.forward_mode = mode
    assert run_case(worker, batch) == []
    worker.model_runner.token_to_kv_pool.request_window.reset.assert_not_called()


def test_disabled_request_window_is_a_noop():
    worker, batch = make_case([4096], [True])
    worker.model_runner.token_to_kv_pool.request_window = None
    assert run_case(worker, batch) == []
    worker.model_runner.forward.assert_not_called()


@pytest.mark.parametrize(
    "unsupported",
    [
        None,
        "eager",
        "gpu",
        "dp",
        "cp",
        "dcp",
        "pp",
        "sp",
        "sp_config",
        "input_sharding",
        "compression_ratio",
        "prefill_graph",
        "a2a",
        "attention",
    ],
)
def test_runtime_validation_rejects_unvalidated_execution_modes(unsupported):
    from sglang.srt.model_executor.runner.eager_runner import EagerRunner

    parallel = SimpleNamespace(
        attn_dp_size=1,
        attn_cp_size=1,
        attn_dcp_size=1,
        pp_size=1,
        enable_layernorm_sp=unsupported == "sp_config",
        enable_attn_tp_input_scattered=unsupported == "input_sharding",
    )
    if unsupported in ("dp", "cp", "dcp", "pp"):
        setattr(
            parallel,
            {
                "dp": "attn_dp_size",
                "cp": "attn_cp_size",
                "dcp": "attn_dcp_size",
                "pp": "pp_size",
            }[unsupported],
            2,
        )
    runner = SimpleNamespace(
        prefill_cuda_graph_runner=(
            object.__new__(EagerRunner)
            if unsupported == "eager"
            else object()
            if unsupported == "prefill_graph"
            else None
        ),
        attn_backend=SimpleNamespace(trtllm_attn=unsupported == "attention"),
        model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(
                            compress_ratio=128
                            if unsupported == "compression_ratio"
                            else 2,
                        ),
                    )
                ]
            )
        ),
    )
    with (
        patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
        patch(
            "sglang.srt.runtime_context.get_platform",
            return_value=SimpleNamespace(is_sm90=unsupported != "gpu"),
        ),
        patch(
            "sglang.srt.runtime_context.get_forward",
            return_value=SimpleNamespace(sp_active=unsupported == "sp"),
        ),
        patch(
            "sglang.srt.layers.moe.get_moe_a2a_backend",
            return_value=SimpleNamespace(is_none=lambda: unsupported != "a2a"),
        ),
    ):
        if unsupported not in (None, "eager"):
            with pytest.raises(NotImplementedError, match="requires SM90 FlashMLA"):
                _validate_batch_runtime(runner)
        else:
            _validate_batch_runtime(runner)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
