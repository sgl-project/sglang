"""CPU checks of production CP tail metadata and DSpark/logits handoff."""

from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from test_dsv41_cp_replay_layout import definitions, folded_fixture


@pytest.mark.parametrize("cp", [2, 4, 8])
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


@pytest.mark.parametrize("capture_aux", [False, True])
@pytest.mark.parametrize("return_logprob", [False, True])
@pytest.mark.parametrize("fail_gather", [False, True])
def test_eager_dspark_uses_compact_metadata_and_restores_full_batch(
    capture_aux, return_logprob, fail_gather
):
    full_meta, compact_meta = object(), object()
    indices = torch.tensor([1, 4, 9])
    hidden, before, aux = torch.ones((3, 2)), torch.ones((3, 4)), torch.ones((3, 8))
    tail = NS(
        cp_metadata=compact_meta,
        output_token_indices=indices,
        extend_seq_lens=torch.tensor([1, 2]),
        extend_seq_lens_cpu=[1, 2],
    )
    batch = NS(
        input_ids=torch.arange(10),
        positions=torch.arange(10),
        attn_cp_metadata=full_meta,
        return_logprob=return_logprob,
        forward_mode=NS(is_extend_without_speculative=lambda: True),
        extend_logprob_start_lens_cpu=[0, 99],
    )

    class Body:
        late_layer_start = 2

        def __call__(self, *args, **kwargs):
            return ((hidden, before), [aux]) if capture_aux else (hidden, before)

    class Model:
        def get_input_embeddings(self):
            return lambda ids: ids

    model = Model()
    model.model = Body()
    model.pp_group = NS(is_last_rank=True)
    model.capture_aux_hidden_states = capture_aux
    model.lm_head = object()
    model.logits_processor = Mock(return_value=NS())
    runner = NS(
        model_runner=NS(
            model=model, attn_backend=NS(tail_forward_metadata=NS(late_layer_tail=tail))
        )
    )
    gathered = []

    def gather(value, fb, stream):
        assert fb.attn_cp_metadata is compact_meta
        gathered.append(value)
        if fail_gather:
            raise RuntimeError("gather failed")
        return value

    @contextmanager
    def shard(emb, pos, fb, ids):
        yield emb, pos, ids

    functions = definitions(
        "srt/model_executor/runner/eager_runner.py",
        ["_execute_extend_cp"],
        {
            "torch": NS(Tensor=torch.Tensor, cuda=NS(current_stream=lambda: None)),
            "DeepseekV4ForCausalLM": Model,
            "LogitsMetadata": NS(from_forward_batch=lambda fb: NS()),
            "cp_gather_after_forward": gather,
            "cp_shard_model_inputs": shard,
        },
        klass="EagerRunner",
        strip_imports=True,
    )
    if fail_gather:
        with pytest.raises(RuntimeError, match="gather failed"):
            functions["_execute_extend_cp"](runner, batch, {})
    else:
        output = functions["_execute_extend_cp"](runner, batch, {})
        args = model.logits_processor.call_args.args
        assert torch.equal(args[0], batch.input_ids[indices])
        assert args[3].extend_seq_lens_cpu == [1, 2]
        assert args[3].extend_logprob_start_lens_cpu == (
            [0, 2] if return_logprob else [1, 2]
        )
        assert output.hidden_states_token_indices is indices
        assert len(gathered) == (2 if capture_aux else 1)
    assert batch.attn_cp_metadata is full_meta


def test_tail_hidden_rows_drop_folded_replay_before_draft_injection():
    original, folded, replay = folded_fixture(prefix=1024, extend=257)
    lens = folded.batch.extend_lens
    tails = [min(128, n) for n in lens]
    indices, offset = [], 0
    for n, tail in zip(lens, tails):
        indices.extend(range(offset + n - tail, offset + n))
        offset += n
    indices = torch.tensor(indices)
    out = NS(
        hidden_states=folded.batch.input_ids[indices, None],
        hidden_states_token_indices=indices,
    )
    replay["drop_folded_rows"](logits_output=out, folded=folded, tail_lens=tails)
    assert (out.hidden_states_token_indices >= 0).all()
    assert torch.equal(
        out.hidden_states[:, 0], original.input_ids[out.hidden_states_token_indices]
    )


def test_compact_tail_uses_existing_persistent_write_mask():
    functions = definitions(
        "srt/layers/attention/deepseek_v4_backend.py",
        ["_compressor_out_loc"],
        klass="DeepseekV4AttnBackend",
    )
    original, folded, _ = folded_fixture()
    fb = NS(
        out_cache_loc=folded.batch.out_cache_loc,
        encoder_swa_compress_rows=folded.keep_rows,
    )
    masked = functions["_compressor_out_loc"](fb)
    assert torch.equal(masked[folded.keep_rows], original.out_cache_loc)
    assert (masked[folded.scheduled_row < 0] == 0).all()
