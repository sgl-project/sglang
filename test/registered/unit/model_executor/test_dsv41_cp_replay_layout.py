"""CPU tensor/control-flow checks; these do not execute CUDA or model math.

Load production definitions without importing CUDA-bound backend modules. Only
their external runtime/kernel dependencies are supplied by the fixtures.
"""

import ast
import importlib.util
import itertools
from copy import copy
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Optional
from unittest.mock import Mock

import msgspec
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]


def definitions(path, names, env=None, *, klass=None, strip_imports=False):
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
    if strip_imports:

        class RuntimeImports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return (
                    None if node.module and node.module.startswith("sglang.") else node
                )

        selected = [RuntimeImports().visit(n) for n in selected]
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
            "InterleaveContextParallelMetadata": NS,
            "get_parallel": lambda: parallel,
            "is_cp_active": lambda fb: True,
            "get_moe_a2a_backend": lambda: NS(is_none=lambda: True),
        },
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


def test_window_writes_are_global_and_padding_never_commits():
    original, folded, _ = folded_fixture()
    window = definitions(
        "srt/mem_cache/dsv41_request_window.py",
        ["WindowLayout", "_first_row_offsets", "window_layout"],
    )
    fb = folded.batch
    req = torch.repeat_interleave(fb.req_pool_indices, torch.tensor(fb.extend_lens))
    pos = torch.cat(
        [torch.arange(p, p + n) for p, n in zip(fb.prefix_lens, fb.extend_lens)]
    )
    real = folded.num_rows
    physical = (real + 63) // 64 * 64
    raw_req = torch.nn.functional.pad(req, (0, physical - real))
    raw_pos = torch.nn.functional.pad(pos, (0, physical - real))
    prep = NS(
        seq_lens_casual=raw_pos.to(torch.int32) + 1,
        positions_casual=raw_pos,
        page_table=torch.empty((physical, 0), dtype=torch.int32),
    )
    helpers = definitions(
        "srt/layers/attention/deepseek_v4_backend.py",
        ["_pad_rows", "_pad_last_dim"],
        {
            "PAGE_INDEX_ALIGNED_SIZE": 64,
            "ceil_align": lambda n, a: (n + a - 1) // a * a,
            "F": torch.nn.functional,
        },
    )
    env = definitions(
        "srt/layers/attention/deepseek_v4_backend.py",
        ["make_core_attn_metadata"],
        {
            **window,
            **helpers,
            "SWA_WINDOW": 128,
            "BuildPageTablePositions": NS(execute=lambda **kwargs: prep),
            "DSV4AttnMetadata": NS,
            "_create_flashmla_metadata": lambda: None,
        },
        klass="DeepseekV4AttnBackend",
        strip_imports=True,
    )
    backend = NS(
        encoder_replay=False,
        encoder_row_floor=folded.row_floor,
        page_size=256,
        index_topk=512,
        present_ratios=(0,),
        low_ratios=(),
        cuda_int32_kwargs={},
        trtllm_attn=False,
        token_to_kv_pool=NS(request_window=NS(capacity=256, zero_row=2560)),
    )
    core = env["make_core_attn_metadata"](
        backend,
        req_to_token=None,
        req_pool_indices_repeated=raw_req,
        seq_lens_casual=prep.seq_lens_casual,
        max_seq_len=512,
        out_loc=fb.out_cache_loc,
        is_prefill=True,
        need_compress=False,
        num_tokens=real,
        num_groups=3,
    )
    layout = core.request_window_layout
    assert torch.equal(layout.req, req)
    assert torch.equal(layout.pos, pos)
    assert layout.write_loc.numel() == layout.commit_mask.numel() == real
    assert (core.swa_page_indices[real:] == -1).all()
    assert (core.swa_topk_lengths[real:] == 0).all()
    assert core.swa_out_cache_loc.numel() == real
    for row in range(real):
        expected = min(128, int(pos[row] - folded.row_floor[row]) + 1)
        assert int(core.swa_topk_lengths[row]) == expected


@pytest.mark.parametrize("rank", range(8))
def test_torch_indexer_uses_rank_local_request_counts(rank):
    lengths, starts, slots = [129, 3, 1], [3968, 0, 64], [2, 7, 9]
    owners, positions, counts = [], [], []
    offset = 0
    for length, start, slot in zip(lengths, starts, slots):
        local = 0
        for i in range(length):
            if (offset + i) % 8 == rank:
                owners.append(slot)
                positions.append(start + i)
                local += 1
        counts.append(local)
        offset += length
    pos = torch.tensor(positions, dtype=torch.int64)
    functions = definitions(
        "srt/layers/attention/dsv4/v41_indexer/scoring.py",
        ["RequestScores", "prefill_requests", "prefill_k_slots"],
        {
            "_score_chunks": lambda **kwargs: iter(()),
            "itertools": itertools,
            "async_h2d": lambda values, dtype, device: torch.tensor(
                values, dtype=dtype, device=device
            ),
        },
    )
    indexer = NS(queries=lambda q, freqs: q, head_weights=lambda x: x, index_topk=4)
    table = torch.arange(8192).repeat(10, 1) + torch.arange(10)[:, None] * 20000
    seq_lens = [start + length for start, length in zip(starts, lengths)]
    inputs = NS(
        compress_ratio=2,
        indexer=indexer,
        req_rows=None,
        seq_lens_cpu=seq_lens,
        k_slots=functions["prefill_k_slots"](
            req_to_token=table,
            req_pool_indices=torch.tensor(slots),
            seq_lens_cpu=seq_lens,
            ratio=2,
        ),
        positions=pos,
        rows_per_request_device=torch.tensor(counts),
        req_pool_indices=torch.tensor(slots),
        reset_outputs=lambda: None,
        q_lora=torch.zeros((len(pos), 1)),
        freqs_cis=torch.zeros((8192, 1)),
        x=torch.ones((len(pos), 1)),
        layer_id=0,
    )
    pool = NS(
        request_window=object(), get_low_ratio_index_k_dequant=lambda layer, loc: loc
    )
    requests = list(functions["prefill_requests"](inputs=inputs, token_to_kv_pool=pool))
    for request, _ in requests:
        request_owners = {owners[i] for i in request.tok.tolist()}
        assert len(request_owners) == 1
        owner = request_owners.pop()
        assert int(request.slots[0]) == owner * 10000
        assert torch.equal(request.lens, (pos[request.tok] + 1) // 2)
    pool.request_window = None
    with pytest.raises(AssertionError, match="torch indexer needs the batch's rows"):
        next(functions["prefill_requests"](inputs=inputs, token_to_kv_pool=pool))


def contract():
    spec = importlib.util.spec_from_file_location(
        "admission_contract",
        ROOT / "python/sglang/srt/mem_cache/dsv41_replay_contract.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("sm90", [False, True])
def test_implicit_cp_is_validated_after_declarations(sm90):
    cfg = NS(
        enable_prefill_cp=True,
        cp_strategy="interleave",
        tp_size=8,
        attn_cp_size=1,
        attn_dp_size=1,
        ep_size=8,
        pp_size=1,
        dp_size=1,
        dcp_size=1,
        moe_a2a_backend="none",
        enable_encoder_swa_bounded_replay=True,
        cuda_graph_config=NS(prefill=NS(backend="disabled")),
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="mooncake",
        speculative_algorithm="DSPARK",
        enable_kv_cache_sharding=False,
        enable_dsa_cache_layer_split=False,
        enable_cp_tp_group_sharing=False,
        enable_unified_cache_external_linker=False,
        enable_hierarchical_cache=True,
        hicache_host_memory_mode="cache",
        hicache_storage_backend="mooncake",
    )
    observed = []

    def validate(resolved, **hardware):
        observed.append(resolved.attn_cp_size)
        contract().validate_cp_encoder_replay(resolved, **hardware)

    def declare(args, origin, **changes):
        for key, value in changes.items():
            setattr(args, key, value)

    functions = definitions(
        "srt/arg_groups/deepseek_v4_hook.py",
        ["validate_deepseek_v4_cp"],
        {
            "resolving_view": lambda args: args,
            "context_parallel_attn_dp_size": lambda cfg, model: 1,
            "declare_resolution": declare,
            "is_npu": lambda: False,
            "get_platform": lambda: NS(is_hip=False, is_cuda=True, is_sm90=sm90),
            "model_config_of": lambda args: NS(hf_config=NS(model_type="deepseek_v41")),
            "validate_cp_encoder_replay": validate,
            "logger": Mock(),
        },
        strip_imports=True,
    )
    if sm90:
        functions["validate_deepseek_v4_cp"](cfg)
    else:
        with pytest.raises(ValueError, match="SM90"):
            functions["validate_deepseek_v4_cp"](cfg)
    assert observed == [8]


def test_plain_cp1_keeps_original_cp_validator_early_return():
    validate = Mock(side_effect=AssertionError("CP1 must not enter the CP extension"))
    functions = definitions(
        "srt/arg_groups/deepseek_v4_hook.py",
        ["validate_deepseek_v4_cp"],
        {"resolving_view": lambda args: args, "validate_cp_encoder_replay": validate},
        strip_imports=True,
    )
    functions["validate_deepseek_v4_cp"](NS(enable_prefill_cp=False))
    validate.assert_not_called()


def test_cp_only_contract_does_not_inspect_legacy_options():
    validate = contract().validate_cp_encoder_replay
    validate(
        NS(enable_encoder_swa_bounded_replay=True, attn_cp_size=1),
        is_cuda=False,
        is_sm90=False,
    )
    validate(NS(enable_encoder_swa_bounded_replay=False), is_cuda=False, is_sm90=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
