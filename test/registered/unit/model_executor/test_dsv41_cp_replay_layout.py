"""CPU tensor/control-flow checks; these do not execute CUDA or model math.

Load production definitions without importing CUDA-bound backend modules. Only
their external runtime/kernel dependencies are supplied by the fixtures.
"""

import ast
from copy import copy
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Optional

import msgspec
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def definitions(path, names, env=None, *, klass=None):
    tree = ast.parse((ROOT / "python/sglang" / path).read_text())
    scope = tree.body
    if klass is not None:
        scope = next(
            n for n in scope if isinstance(n, ast.ClassDef) and n.name == klass
        ).body
    selected = [
        n
        for n in scope
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert {n.name for n in selected} == set(names)
    namespace = dict(
        torch=torch, msgspec=msgspec, Any=Any, Optional=Optional, copy=copy
    )
    namespace.update(env or {})
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def folded_fixture(prefix=256, extend=17):
    prefixes, lengths, slots = [prefix, 0, 64], [extend, 3, 1], [3, 7, 5]
    all_ids = [list(range(10000 * (i + 1), 10000 * (i + 1) + 2048)) for i in range(3)]
    table = torch.arange(8 * 2048).reshape(8, 2048)
    batch = NS(
        reqs=[NS(full_untruncated_fill_ids=ids) for ids in all_ids],
        req_pool_indices=torch.tensor(slots),
        req_pool_indices_cpu=torch.tensor(slots),
        input_ids=torch.tensor(
            [t for ids, p, n in zip(all_ids, prefixes, lengths) for t in ids[p : p + n]]
        ),
        out_cache_loc=torch.cat(
            [table[s, p : p + n] for s, p, n in zip(slots, prefixes, lengths)]
        ),
        prefix_lens=prefixes,
        extend_lens=lengths,
        extend_num_tokens=sum(lengths),
        prefill_input_ids_cpu=None,
        extend_logprob_start_lens=lengths,
        engram_history=None,
        global_num_tokens=[sum(lengths) + min(prefix, 128) + 64],
        draft_global_num_tokens=[sum(lengths) + min(prefix, 128) + 64],
    )
    runner = NS(device="cpu", req_to_token_pool=NS(req_to_token=table))
    replay = definitions(
        "srt/model_executor/encoder_swa_replay.py",
        [
            "FoldedExtend",
            "_fold_batch",
            "_to_device",
            "drop_folded_rows",
            "_tail_rows_on_original",
        ],
        {
            "async_h2d": lambda x, dtype, device: torch.tensor(
                x, dtype=dtype, device=device
            ),
            "is_pin_memory_available": lambda device: False,
        },
    )
    folded = replay["_fold_batch"](
        batch=batch, runner=runner, rows=[(0, max(0, prefix - 128), prefix), (2, 0, 64)]
    )
    return batch, folded, replay


def interleave_functions(cp):
    parallel = NS(attn_cp_size=cp)
    functions = definitions(
        "srt/layers/cp/interleave.py",
        ["cp_interleave_to_sequence_order", "cp_sequence_to_interleave_order"],
        {"get_parallel": lambda: parallel, "is_interleave_extend": lambda fb: True},
    )
    return definitions(
        "srt/layers/cp/utils.py",
        ["cp_interleave_input_ids"],
        {
            **functions,
            "get_parallel": lambda: parallel,
            "is_cp_active": lambda fb: True,
            "get_moe_a2a_backend": lambda: NS(is_none=lambda: True),
        },
    )


@pytest.mark.parametrize("cp", [2, 4, 8])
@pytest.mark.parametrize(
    "prefix,extend", [(2, 1), (126, 1), (128, 1), (130, 3), (256, 17), (1024, 129)]
)
def test_folded_rows_round_trip_without_changing_upstream_replay(cp, prefix, extend):
    original, folded, replay = folded_fixture(prefix, extend)
    functions = interleave_functions(cp)
    physical = (folded.num_rows + cp - 1) // cp * cp
    fb = NS(
        attn_cp_metadata=NS(
            total_seq_lens=folded.num_rows,
            per_rank_actual_token=[physical // cp] * cp,
            gather_index=None,
        )
    )
    packed = functions["cp_interleave_input_ids"](folded.batch.input_ids, fb)
    recovered = functions["cp_interleave_to_sequence_order"](packed, fb)
    assert torch.equal(recovered, folded.batch.input_ids)
    output = NS(hidden_states=recovered[:, None], hidden_states_token_indices=None)
    replay["drop_folded_rows"](logits_output=output, folded=folded)
    assert torch.equal(output.hidden_states[:, 0], original.input_ids)
    assert folded.batch.global_num_tokens == original.global_num_tokens
    assert folded.batch.draft_global_num_tokens == original.draft_global_num_tokens
    assert torch.equal(
        folded.scheduled_row[folded.keep_rows], torch.arange(original.extend_num_tokens)
    )


def test_compact_input_ids_keep_original_owners():
    functions = interleave_functions(4)
    # Tail rows are not a new round-robin sequence. The explicit gather map
    # retains their owners from the full folded batch, with padding slots.
    gather = torch.tensor([8, 0, 4, 9, 1])
    fb = NS(attn_cp_metadata=NS(per_rank_actual_token=[3] * 4, gather_index=gather))
    ids = torch.tensor([105, 108, 109, 113, 120])
    packed = functions["cp_interleave_input_ids"](ids, fb)
    expected = torch.tensor([108, 120, 0, 0, 109, 0, 0, 0, 105, 113, 0, 0])
    assert torch.equal(packed, expected)
    assert torch.equal(functions["cp_interleave_to_sequence_order"](packed, fb), ids)
