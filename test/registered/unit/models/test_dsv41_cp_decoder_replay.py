"""CP replay contracts: original rank ownership, canonical IDs and DSpark rows."""

from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    LateLayerTail,
)
from sglang.srt.layers.cp.utils import cp_relayout_tail_input_ids
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4Model


@pytest.mark.parametrize("lengths", [[8192], [131, 7, 259], [1, 2, 3, 4], [129, 130]])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_tail_routing_matches_original_rank_ownership(lengths, cp_size):
    n = sum(lengths)
    selected = []
    positions = []
    offset = 0
    for req, length in enumerate(lengths):
        selected.extend(range(offset + max(0, length - 128), offset + length))
        positions.extend(range(500 * req, 500 * req + length))
        offset += length
    indices = torch.tensor(selected)
    lens = torch.tensor([min(128, x) for x in lengths], dtype=torch.int32)
    canonical_ids = torch.arange(n) + 1000
    canonical_ids[indices[len(indices) // 2]] = 99  # canonical image token
    full_len = (n + cp_size - 1) // cp_size
    packed = torch.zeros(full_len * cp_size, dtype=torch.int64)
    for rank in range(cp_size):
        ids = canonical_ids[rank::cp_size]
        packed[rank * full_len : rank * full_len + len(ids)] = ids
    full_meta = NS(per_rank_actual_token=[full_len] * cp_size)
    batch = NS(batch_size=len(lengths), positions=torch.tensor(positions))
    backend = object.__new__(DeepseekV4AttnBackend)
    for rank in range(cp_size):
        with patch(
            "sglang.srt.layers.attention.deepseek_v4_backend.get_parallel",
            return_value=NS(attn_cp_rank=rank, attn_cp_size=cp_size),
        ):
            layout = backend._late_layer_tail_cp_layout(batch, indices, lens)
        meta = layout["cp_metadata"]
        got = cp_relayout_tail_input_ids(packed, full_meta, meta, indices)
        width = max(meta.per_rank_actual_token)
        expected_indices = [i for i in selected if i % cp_size == rank]
        expected_ids = canonical_ids[expected_indices]
        local = got[rank * width : (rank + 1) * width]
        torch.testing.assert_close(local[: len(expected_ids)], expected_ids)
        assert not local[len(expected_ids) :].any()
        torch.testing.assert_close(got[meta.gather_index], canonical_ids[indices])
        original_local_ids = canonical_ids[rank::cp_size]
        torch.testing.assert_close(
            original_local_ids[layout["local_token_indices"]], expected_ids
        )
        torch.testing.assert_close(
            layout["local_positions"][: len(expected_ids)],
            batch.positions[expected_indices],
        )


@pytest.mark.parametrize("fail_gather", [False, True])
@pytest.mark.parametrize("use_tail", [False, True])
def test_runner_preserves_image_ids_and_global_injection_indices(fail_gather, use_tail):
    # The scheduler's hash placeholder must never reach routing or logits IDs.
    original = torch.tensor([11, 12, 1_000_000_001, 14, 15])
    canonical = torch.tensor([11, 12, 99, 14, 15])
    indices = torch.tensor([2, 3, 4])
    tail_meta = object()
    full_meta = NS(total_seq_lens=5)
    tail = LateLayerTail(
        token_indices=torch.tensor([1]),
        positions=torch.tensor([3]),
        extend_seq_lens=torch.tensor([3]),
        extend_seq_lens_cpu=[3],
        swa_out_cache_loc=indices,
        cp_metadata=tail_meta,
        global_token_indices=indices,
    )
    output = NS(hidden_states_token_indices=None)
    model = object.__new__(DeepseekV4ForCausalLM)
    torch.nn.Module.__init__(model)
    model.capture_aux_hidden_states = True
    model.pp_group = NS(is_last_rank=True)
    model.prepare_model_inputs = Mock(return_value=(canonical, torch.ones(5, 4)))
    model.model = Mock()
    model.model.late_layer_start = 21 if use_tail else None
    model.model.return_value = (
        (torch.ones(1, 4), torch.ones(1, 4)),
        [torch.ones(1, 4)],
    )
    model.lm_head = None
    model.logits_processor = Mock(return_value=output)
    runner = object.__new__(EagerRunner)
    runner.model_runner = NS(
        model=model, attn_backend=NS(tail_forward_metadata=NS(late_layer_tail=tail))
    )
    batch = NS(
        input_ids=original,
        positions=torch.arange(5),
        attn_cp_metadata=full_meta,
        forward_mode=NS(is_extend_without_speculative=lambda: True),
    )
    seen = []

    def gather(value, fb, *_):
        seen.append(fb.attn_cp_metadata)
        if fail_gather:
            raise RuntimeError("test gather failure")
        return value

    with (
        patch(
            "sglang.srt.model_executor.runner.eager_runner.cp_shard_model_inputs",
            return_value=nullcontext(
                (torch.ones(1, 4), torch.tensor([3]), torch.tensor([14]))
            ),
        ),
        patch(
            "sglang.srt.model_executor.runner.eager_runner.cp_gather_after_forward",
            side_effect=gather,
        ),
        patch(
            "sglang.srt.layers.logits_processor.LogitsMetadata.from_forward_batch",
            return_value=NS(),
        ),
        patch("torch.cuda.current_stream", return_value=Mock()),
    ):
        if fail_gather:
            with pytest.raises(RuntimeError, match="test gather failure"):
                runner._execute_extend_cp(batch, {})
        else:
            result = runner._execute_extend_cp(batch, {})
            expected_ids = canonical[indices] if use_tail else canonical
            torch.testing.assert_close(
                model.logits_processor.call_args.args[0], expected_ids
            )
            if use_tail:
                assert result.hidden_states_token_indices is indices
                meta = model.logits_processor.call_args.args[3]
                assert meta.extend_seq_lens_cpu == [3]
            else:
                assert result.hidden_states_token_indices is None
    assert all(x is (tail_meta if use_tail else full_meta) for x in seen)
    assert batch.attn_cp_metadata is full_meta
    torch.testing.assert_close(batch.input_ids, original)


def test_model_tail_keeps_local_image_mask_and_global_routing():
    full_meta = NS(per_rank_actual_token=[3, 3], total_seq_lens=6)
    # Original global tails [3, 4, 5], rank 0 owns only global row 4.
    tail_meta = NS(per_rank_actual_token=[2, 2], gather_index=torch.tensor([2, 0, 3]))
    tail = LateLayerTail(
        token_indices=torch.tensor([2]),
        positions=torch.tensor([4, 0]),
        extend_seq_lens=torch.tensor([3]),
        extend_seq_lens_cpu=[3],
        swa_out_cache_loc=torch.tensor([3, 4, 5]),
        cp_metadata=tail_meta,
        global_token_indices=torch.tensor([3, 4, 5]),
        pad_rows=1,
    )
    model = object.__new__(DeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.pp_group = NS(world_size=1)
    model.config = NS(model_type="deepseek_v41", vision_n_layers=1, image_token_id=99)
    model.start_layer = 0
    model.end_layer = 2
    model.late_layer_start = 1
    model.engram_hasher = Mock(return_value=torch.zeros(6, 1, dtype=torch.int64))
    model.dspark_layers_to_capture = [0, 1]
    engram = Mock(side_effect=lambda hidden, *a, **kw: hidden + 7)
    engram.layer_hash_index = 0
    seen = []

    def layer(**kwargs):
        seen.append(kwargs)
        state = kwargs["state"]
        return state

    model.layers = [
        NS(engram=None, hc_cfg=None, forward_hc_pre_from_prev=layer),
        NS(engram=engram, hc_cfg=None, forward_hc_pre_from_prev=layer),
    ]
    batch = NS(
        input_ids=torch.tensor([10, 11, 12, 13, 1_000_000_001, 15]),
        attn_cp_metadata=full_meta,
        capture_hidden_mode=object(),
        return_logprob=False,
        contains_mm_inputs=lambda: True,
        forward_mode=NS(
            is_extend=lambda: True, is_extend_without_speculative=lambda: True
        ),
    )
    backend = NS(tail_forward_metadata=NS(late_layer_tail=tail))

    def enter(fb):
        fb.attn_cp_metadata = tail_meta
        return full_meta

    def leave(saved, fb):
        fb.attn_cp_metadata = saved

    backend.enter_late_layer_tail = enter
    backend.exit_late_layer_tail = leave
    aux = []
    with (
        patch("sglang.srt.models.deepseek_v4.is_cp_active", return_value=True),
        patch(
            "sglang.srt.models.deepseek_v4.get_forward",
            return_value=NS(sp_active=False),
        ),
        patch(
            "sglang.srt.models.deepseek_v4.get_parallel",
            return_value=NS(attn_cp_rank=0, attn_cp_size=2),
        ),
        patch("sglang.srt.models.deepseek_v4.get_attn_backend", return_value=backend),
        patch(
            "sglang.srt.models.deepseek_v4.check_cuda_graph_backend", return_value=True
        ),
    ):
        model._forward_layers_hc_pre_from_prev(
            positions=torch.tensor([0, 2, 4]),
            hidden_states=torch.ones(3, 1, 4),
            forward_batch=batch,
            input_ids=torch.tensor([10, 12, 99]),
            input_ids_global=torch.tensor([10, 12, 99, 11, 13, 15]),
            capture_dspark=True,
            dspark_aux_hidden_states=aux,
        )
    torch.testing.assert_close(seen[1]["input_ids"], torch.tensor([99, 0]))
    torch.testing.assert_close(
        seen[1]["input_ids_global"], torch.tensor([99, 0, 13, 15])
    )
    torch.testing.assert_close(seen[1]["state"].residual[0], torch.ones(1, 4))
    assert [a.shape[0] for a in aux] == [2, 2]
    assert batch.attn_cp_metadata is full_meta


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
