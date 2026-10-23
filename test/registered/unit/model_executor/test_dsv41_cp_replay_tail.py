"""CPU checks of production CP tail metadata and DSpark/logits handoff."""

from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from test_dsv41_cp_replay_layout import (
    definitions,
    folded_fixture,
)

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize("cp", [2, 8])
@pytest.mark.parametrize("indices,lens", [([5], [1]), ([5, 6, 12, 20, 31], [3, 2])])
def test_tail_builder_preserves_full_batch_owners(cp, indices, lens):
    full_positions = torch.arange(40)
    token_indices = torch.tensor(indices)
    all_gather_slots = None
    for rank in range(cp):
        functions = definitions(
            "srt/layers/attention/deepseek_v4_backend.py",
            ["_late_layer_tail_cp_layout"],
            {
                "get_parallel": lambda: NS(attn_cp_rank=rank, attn_cp_size=cp),
                "InterleaveContextParallelMetadata": NS,
            },
            klass="DeepseekV4AttnBackend",
        )
        result = functions["_late_layer_tail_cp_layout"](
            None,
            NS(positions=full_positions, batch_size=len(lens)),
            token_indices,
            torch.tensor(lens),
        )
        expected = [i for i in indices if i % cp == rank]
        assert result["local_token_indices"].tolist() == [
            (i - rank) // cp for i in expected
        ]
        assert result["local_positions"][: len(expected)].tolist() == expected
        assert sum(result["local_lens_cpu"]) == len(expected)
        meta = result["cp_metadata"]
        width = meta.per_rank_actual_token[rank]
        if all_gather_slots is None:
            all_gather_slots = torch.full((cp * width,), -1)
        all_gather_slots[rank * width : rank * width + len(expected)] = torch.tensor(
            expected, dtype=torch.int64
        )
        assert result["pad_rows"] + len(expected) == width
    assert all_gather_slots[meta.gather_index].tolist() == indices


@pytest.mark.parametrize("cp,encoder", [(1, False), (1, True), (8, False), (8, True)])
def test_tail_metadata_limits_write_mask_and_output_ownership_to_cp_encoder(
    cp, encoder
):
    layout = definitions(
        "kernels/ops/attention/dsv4/attn_metadata_kernels.py",
        ["late_layer_tail_lens", "late_layer_tail_layout"],
    )
    helpers = definitions(
        "srt/layers/attention/deepseek_v4_backend.py", ["LateLayerTail", "_tail_rows"]
    )
    functions = definitions(
        "srt/layers/attention/deepseek_v4_backend.py",
        [
            "_build_late_layer_tail_metadata",
            "_late_layer_tail_cp_layout",
            "_compressor_out_loc",
        ],
        {
            **layout,
            **helpers,
            "SWA_WINDOW": 128,
            "is_cp_active": lambda fb: cp > 1,
            "get_exec": lambda: NS(
                features=NS(enable_encoder_swa_bounded_replay=encoder)
            ),
            "get_parallel": lambda: NS(attn_cp_rank=0, attn_cp_size=cp),
            "late_layer_whole_extend": lambda *args, **kwargs: None,
            "InterleaveContextParallelMetadata": NS,
            "async_h2d": lambda data, dtype, device: torch.tensor(
                data, dtype=dtype, device=device
            ),
        },
        klass="DeepseekV4AttnBackend",
    )
    fb = NS(
        extend_seq_lens_cpu=[129],
        seq_lens_cpu=torch.tensor([257]),
        seq_lens=torch.tensor([257]),
        out_cache_loc=torch.arange(128, 257),
        encoder_swa_compress_rows=torch.tensor([128]) if encoder else None,
        positions=torch.arange(128, 257),
        req_pool_indices=torch.tensor([3]),
        batch_size=1,
    )
    pool = NS(
        request_window=object() if encoder else None,
        translate_loc_from_full_to_swa=lambda rows: rows,
    )
    captured = {}

    def initialize(**kwargs):
        captured.update(kwargs)
        return NS(
            core_attn_metadata=NS(request_window_layout=NS(write_loc=torch.arange(128)))
        )

    backend = NS(
        encoder_row_floor=None,
        model_runner=NS(spec_algorithm=NS(is_dspark=lambda: True)),
        init_forward_metadata_prefill=initialize,
        token_to_kv_pool=pool,
        low_ratios=(),
    )
    backend._compressor_out_loc = Mock(wraps=functions["_compressor_out_loc"])
    backend._late_layer_tail_cp_layout = lambda *args: functions[
        "_late_layer_tail_cp_layout"
    ](backend, *args)
    result = functions["_build_late_layer_tail_metadata"](backend, fb)
    if cp > 1 and encoder:
        assert captured["out_cache_loc"][:-1].eq(0).all()
        assert captured["out_cache_loc"][-1] == fb.out_cache_loc[-1]
        assert result.late_layer_tail.global_token_indices.tolist() == list(
            range(1, 129)
        )
        backend._compressor_out_loc.assert_called_once_with(fb)
    else:
        assert torch.equal(captured["out_cache_loc"], fb.out_cache_loc[1:])
        assert result.late_layer_tail.global_token_indices is None
        backend._compressor_out_loc.assert_not_called()


def test_encoder_off_runner_retains_full_gather_and_logits_metadata():
    full = object()
    batch = NS(
        input_ids=torch.arange(9), positions=torch.arange(9), attn_cp_metadata=full
    )
    hidden = torch.ones((9, 2))
    model = NS(
        get_input_embeddings=lambda: lambda ids: ids,
        model=lambda *args, **kwargs: (hidden, hidden),
        get_cp_output_tail=lambda fb: None,
        pp_group=NS(is_last_rank=True),
        lm_head=object(),
        logits_processor=Mock(return_value=NS()),
    )

    @contextmanager
    def shard(emb, pos, fb, ids):
        yield emb, pos, ids

    def gather(value, fb, stream):
        assert fb.attn_cp_metadata is full
        return value

    execute = definitions(
        "srt/model_executor/runner/eager_runner.py",
        ["_execute_extend_cp"],
        {
            "torch": NS(Tensor=torch.Tensor, cuda=NS(current_stream=lambda: None)),
            "cp_shard_model_inputs": shard,
            "cp_gather_after_forward": gather,
        },
        klass="EagerRunner",
        strip_imports=True,
    )["_execute_extend_cp"]
    result = execute(NS(model_runner=NS(model=model)), batch, {})
    args = model.logits_processor.call_args.args
    assert args[0] is batch.input_ids
    assert args[3] is batch
    assert batch.attn_cp_metadata is full
    assert not hasattr(result, "hidden_states_token_indices")


@pytest.mark.parametrize(
    "capture,aux,logprob,fail_gather",
    [
        ("full", False, False, False),
        ("full", True, True, False),
        ("last", True, False, False),
        ("null", False, False, False),
        ("full", False, False, True),
    ],
)
def test_dspark_capture_and_fold_keep_compact_rows(capture, aux, logprob, fail_gather):
    original, folded, replay = folded_fixture(prefix=256, extend=1)
    tails = [min(128, n) for n in folded.batch.extend_lens]
    indices, offset = [], 0
    for n, tail_len in zip(folded.batch.extend_lens, tails):
        indices.extend(range(offset + n - tail_len, offset + n))
        offset += n
    indices = torch.tensor(indices)
    tail = NS(
        cp_metadata=object(),
        global_token_indices=indices,
        output_token_indices=indices,
        extend_seq_lens=torch.tensor(tails),
        extend_seq_lens_cpu=tails,
    )
    mode = NS(
        need_capture=lambda: capture != "null",
        is_full=lambda: capture == "full",
        is_last=lambda: capture == "last",
    )
    hidden = folded.batch.input_ids[indices, None]
    full_metadata = object()
    batch = NS(
        input_ids=folded.batch.input_ids,
        positions=torch.arange(folded.num_rows),
        attn_cp_metadata=full_metadata,
        return_logprob=logprob,
        extend_logprob_start_lens_cpu=[0, 99, 99],
        forward_mode=NS(is_extend_without_speculative=lambda: True),
        capture_hidden_mode=mode,
        mm_inputs=None,
    )
    capture_fn = definitions(
        "srt/layers/logits_processor.py",
        ["_get_hidden_states_to_store"],
        definitions("srt/layers/aux_hidden_states.py", ["pack_aux_hidden_states"]),
        klass="LogitsProcessor",
    )["_get_hidden_states_to_store"]
    last_rows = torch.cumsum(torch.tensor(tails), dim=0) - 1

    def logits(ids, states, head, metadata, aux_states, **kwargs):
        assert torch.equal(ids, batch.input_ids[indices])
        assert metadata.extend_seq_lens_cpu == tails
        expected_starts = (
            [min(a, b) for a, b in zip([0, 99, 99], tails)] if logprob else tails
        )
        assert metadata.extend_logprob_start_lens_cpu == expected_starts
        stored = capture_fn(
            None,
            states,
            kwargs.get("hidden_states_before_norm"),
            aux_states,
            states[last_rows],
            hidden[last_rows],
            [value[last_rows] for value in aux_states]
            if aux_states is not None
            else None,
            None,
            metadata,
        )
        return NS(hidden_states=stored, hidden_states_token_indices=None)

    class Body:
        late_layer_start = 2
        hc_pre_from_prev_sublayer = True

        def __call__(self, *args, **kwargs):
            return ((hidden, hidden), [hidden]) if aux else (hidden, hidden)

    model = NS(
        model=Body(),
        capture_aux_hidden_states=aux,
        pp_group=NS(is_last_rank=True),
        lm_head=object(),
        logits_processor=logits,
        get_input_embeddings=lambda: lambda ids: ids,
    )
    hook = definitions(
        "srt/models/deepseek_v4.py",
        ["get_cp_output_tail"],
        {
            "get_attn_backend": lambda: NS(
                tail_forward_metadata=NS(late_layer_tail=tail)
            )
        },
        klass="DeepseekV4ForCausalLM",
    )["get_cp_output_tail"]
    model.get_cp_output_tail = lambda fb: hook(model, fb)
    gathered = []

    def gather(value, fb, stream):
        assert fb.attn_cp_metadata is tail.cp_metadata
        gathered.append(value)
        if fail_gather:
            raise RuntimeError("gather failed")
        return value

    @contextmanager
    def shard(emb, pos, fb, ids):
        yield emb, pos, ids

    execute = definitions(
        "srt/model_executor/runner/eager_runner.py",
        ["_execute_extend_cp"],
        {
            "torch": NS(Tensor=torch.Tensor, cuda=NS(current_stream=lambda: None)),
            "LogitsMetadata": NS(
                from_forward_batch=lambda fb: NS(capture_hidden_mode=mode)
            ),
            "cp_gather_after_forward": gather,
            "cp_shard_model_inputs": shard,
        },
        klass="EagerRunner",
        strip_imports=True,
    )["_execute_extend_cp"]
    runner = NS(model_runner=NS(model=model))
    if fail_gather:
        with pytest.raises(RuntimeError, match="gather failed"):
            execute(runner, batch, {})
    else:
        output = execute(runner, batch, {})
        assert len(gathered) == (2 if aux else 1)
        replay["drop_folded_rows"](logits_output=output, folded=folded, tail_lens=tails)
        if capture == "full":
            assert torch.equal(output.hidden_states[:, 0], original.input_ids)
            assert output.hidden_states_token_indices.tolist() == list(
                range(original.extend_num_tokens)
            )
        elif capture == "last":
            rows = torch.cumsum(torch.tensor(original.extend_lens), dim=0) - 1
            assert torch.equal(output.hidden_states[:, 0], original.input_ids[rows])
            assert output.hidden_states_token_indices is None
        else:
            assert output.hidden_states is None
            assert output.hidden_states_token_indices is None
    assert batch.attn_cp_metadata is full_metadata


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
