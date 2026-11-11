"""CPU contract tests for DSV4.1 A3's block-noncausal draft attention.

The public device operators are mocked. In particular these tests establish
the logical/physical index contract, not device numerical or graph correctness.
"""

import ast
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[5]
BACKEND = (
    ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py"
)
METADATA = ROOT / "python/sglang/kernels/ops/speculative/dspark/dspark_attn_metadata.py"

register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def extract(path, names, namespace, *, cls=None):
    body = ast.parse(path.read_text()).body
    if cls is not None:
        body = next(
            n for n in body if isinstance(n, ast.ClassDef) and n.name == cls
        ).body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def make_backend(*, a5=False):
    reference = extract(
        METADATA,
        {
            "build_block_seq_lens_causal",
            "compute_dspark_window_gather",
            "build_dspark_swa_page_indices",
            "_compact_dspark_window_then_block",
        },
        {
            "torch": torch,
            "DsparkWindowGather": SimpleNamespace,
            "ceil_align": lambda x, n: (x + n - 1) // n * n,
        },
    )
    public = SimpleNamespace(
        attention=Mock(side_effect=lambda q, **kw: (q + 1, None)),
        metadata=Mock(return_value=torch.ones(2048, dtype=torch.int32)),
    )
    legacy = Mock(side_effect=lambda q, **kw: (q + 2, None))
    methods = {
        "_is_dspark_draft_block",
        "_init_dspark_sparse_metadata",
        "_build_dspark_sparse_indices",
        "_dsv41_a3_draft_attention_kwargs",
        "_dsv41_a3_draft_kernel_metadata",
        "_forward_swa",
        "_refresh_graph_dspark_sparse_metadata",
    }
    namespace = extract(
        BACKEND,
        methods,
        {
            "torch": torch,
            "F": F,
            "is_npu_arch35": lambda: a5,
            "_dsv41_a3_ops": lambda: public,
            "_sparse_attn_ops": lambda *a, **kw: (None, legacy),
            "_sparse_attn_kv_quant_kwargs": lambda: {"kv_quant_mode": 1},
            "BuildBlockSeqLensCausal": SimpleNamespace(
                execute=reference["build_block_seq_lens_causal"]
            ),
            "ComputeDsparkWindowGather": SimpleNamespace(
                execute=reference["compute_dspark_window_gather"]
            ),
            "BuildDsparkSwaPageIndices": SimpleNamespace(
                execute=reference["build_dspark_swa_page_indices"]
            ),
        },
        cls="DeepseekV4AscendAttnBackend",
    )
    backend_type = type("DraftBackend", (), {n: namespace[n] for n in methods})
    backend = backend_type()
    backend._is_dspark_draft_worker = True
    backend._dsv4_sliding_window_size = 128
    backend._dsv4_q_head_num = 32
    backend._dsv4_head_dim = 512
    backend.public = public
    backend.legacy = legacy
    return backend


def initialize_metadata(backend, prefixes, *, gamma=5, padding=0):
    prefixes = torch.tensor(prefixes, dtype=torch.int32)
    bs = len(prefixes)
    mode = SimpleNamespace(is_target_verify=lambda: True)
    fb = SimpleNamespace(
        seq_lens=prefixes,
        req_pool_indices=torch.arange(bs),
        out_cache_loc=torch.arange(bs * gamma + padding),
        spec_info=SimpleNamespace(draft_token_num=gamma),
        spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
        forward_mode=mode,
    )
    backend.forward_metadata = SimpleNamespace(
        actual_seq_lengths_q_pa=torch.arange(0, bs * gamma + 1, gamma).int(),
        actual_seq_lengths_kv=prefixes + gamma,
        swa_page_table=torch.tensor([[7, 3, 11]] * bs),
        kernel_metadata={"c1a_metadata": torch.ones(2048, dtype=torch.int32)},
    )
    backend._init_dspark_sparse_metadata(fb)
    return fb, backend.forward_metadata


@pytest.mark.parametrize("prefix", [0, 1, 127, 128, 129, 256, 257])
def test_logical_draft_indices_match_old_physical_gather_with_nonsequential_pages(
    prefix,
):
    a3 = make_backend()
    a5 = make_backend(a5=True)
    gamma, page_size = 5, 128
    page_table = torch.tensor([[7, 3, 11], [2, 9, 5]], dtype=torch.int64)
    positions = torch.arange(page_table.shape[1] * page_size)
    physical = page_table[:, positions // page_size] * page_size + positions % page_size
    full = torch.arange(physical.numel()).view_as(physical)
    mapping = torch.empty(full.numel(), dtype=torch.int64)
    mapping[full.flatten()] = physical.flatten()
    a5.req_to_token = full
    a5.token_to_kv_pool = SimpleNamespace(full_to_swa_index_mapping=mapping)
    prefixes = torch.tensor([prefix, max(0, prefix - 1)], dtype=torch.int32)
    out_loc = torch.cat([full[i, p : p + gamma] for i, p in enumerate(prefixes)])
    kwargs = dict(
        seq_lens=prefixes,
        req_pool_indices=torch.arange(2),
        out_cache_loc=out_loc,
        block_size=gamma,
    )
    logical = a3._build_dspark_sparse_indices(**kwargs)
    old_physical = a5._build_dspark_sparse_indices(**kwargs)
    assert logical.shape == (2 * gamma, 256)
    assert logical.dtype == torch.int32
    for req in range(2):
        p = int(prefixes[req])
        # DSpark keeps W context tokens PLUS the complete noncausal block.
        expected = torch.arange(max(0, p - 128), p + gamma)
        for query in range(gamma):
            row = req * gamma + query
            torch.testing.assert_close(logical[row, : len(expected)].long(), expected)
            assert (logical[row, len(expected) :] == -1).all()
            mapped = torch.full_like(logical[row], -1)
            keep = logical[row] >= 0
            idx = logical[row, keep].long()
            mapped[keep] = physical[req, idx].int()
            torch.testing.assert_close(mapped, old_physical[row])


def test_a3_callspec_uses_noncausal_sparse_indices_and_public_metadata_lengths():
    backend = make_backend()
    _, fm = initialize_metadata(backend, [1, 129])
    kwargs = backend._dsv41_a3_draft_attention_kwargs()
    assert kwargs["ori_mask_mode"] == 0
    assert kwargs["cmp_mask_mode"] == 0
    assert kwargs["cmp_ratio"] == 1
    assert kwargs["ori_win_left"] == 132
    assert kwargs["ori_win_right"] == 0
    assert kwargs["layout_q"] == "TND"
    assert kwargs["layout_kv"] == "PA_BBND"
    assert kwargs["cu_seqlens_q"].tolist() == [0, 5, 10]
    assert kwargs["seqused_ori_kv"].tolist() == [6, 134]
    assert kwargs["ori_topk_length"].shape == (10, 1)
    assert kwargs["ori_topk_length"].dtype == torch.int32
    assert kwargs["ori_topk_length"].flatten().tolist() == [6] * 5 + [133] * 5
    metadata = backend._dsv41_a3_draft_kernel_metadata(2)
    args, meta_kwargs = backend.public.metadata.call_args
    assert args == (32, 1, 512)
    assert meta_kwargs["batch_size"] == 2
    assert meta_kwargs["has_ori_kv"] is True
    assert meta_kwargs["has_cmp_kv"] is False
    assert meta_kwargs["ori_topk"] == 256
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(meta_kwargs[key], value)
        else:
            assert meta_kwargs[key] == value
    assert metadata.shape == (2048,)
    # This path has no dependency on the old host-metadata CPU mirrors.
    assert not hasattr(fm, "seq_lens_cpu_int")


def test_a3_draft_kv_lengths_include_block_without_cpu_mirror():
    backend = make_backend()
    fb, fm = initialize_metadata(backend, [127, 129])
    # A mirrorless base backend sees committed prefix lengths, not the draft
    # tokens already stored by this block. Derive the complete KV length here.
    fm.actual_seq_lengths_kv = fb.seq_lens.clone()
    backend._init_dspark_sparse_metadata(fb)
    assert fm.actual_seq_lengths_kv.tolist() == [132, 134]
    assert fm.actual_seq_lengths_kv.dtype == torch.int32
    assert not hasattr(fm, "seq_lens_cpu_int")
    assert backend._dsv41_a3_draft_attention_kwargs()["seqused_ori_kv"].tolist() == [
        132,
        134,
    ]


def test_a3_trims_attention_tp_padding_and_zero_pads_output_without_changing_indptr():
    backend = make_backend()
    fb, fm = initialize_metadata(backend, [129], padding=1)
    assert fm.ori_sparse_indices.shape == (5, 1, 256)
    q = torch.randn(6, 32, 512, dtype=torch.bfloat16)
    sink = torch.zeros(32)
    cache = torch.zeros(12, 128, 1, 512, dtype=torch.bfloat16)
    backend.token_to_kv_pool = SimpleNamespace(get_swa_buffer=lambda layer: cache)
    out = backend._forward_swa(q, SimpleNamespace(layer_id=0, scaling=0.125), fb, sink)
    args, kwargs = backend.public.attention.call_args
    assert args[0].shape == (5, 32, 512)
    assert kwargs["cu_seqlens_q"][-1].item() == args[0].shape[0]
    assert kwargs["ori_sparse_indices"].shape[0] == args[0].shape[0]
    assert kwargs["ori_kv"] is cache
    assert kwargs["ori_block_table"].dtype == torch.int32
    assert kwargs["sinks"] is sink
    assert kwargs["metadata"] is fm.kernel_metadata["c1a_metadata"]
    assert "kv_quant_mode" not in kwargs
    torch.testing.assert_close(out[:5], q[:5] + 1)
    assert out.shape == q.shape and out.dtype == q.dtype
    assert torch.count_nonzero(out[5]) == 0
    backend.legacy.assert_not_called()


def test_a3_nonblock_draft_path_keeps_causal_swa_callspec():
    backend = make_backend()
    _, fm = initialize_metadata(backend, [129])
    fm.ori_sparse_indices = None
    fm.ori_win_left = 127
    kwargs = backend._dsv41_a3_draft_attention_kwargs()
    assert kwargs["ori_mask_mode"] == 4
    assert kwargs["ori_win_left"] == 127
    assert "ori_topk_length" not in kwargs
    backend._dsv41_a3_draft_kernel_metadata(1)
    assert backend.public.metadata.call_args.kwargs["ori_topk"] == 0


def test_a5_swa_keeps_legacy_operator_and_padded_queries():
    backend = make_backend(a5=True)
    backend.forward_metadata = SimpleNamespace(
        actual_seq_lengths_q_pa=torch.tensor([0, 5]),
        actual_seq_lengths_kv=torch.tensor([134]),
        ori_sparse_indices=torch.zeros(6, 1, 256, dtype=torch.int32),
        swa_page_table=torch.tensor([[7, 3]]),
        kernel_metadata={"c1a_metadata": torch.zeros(2048, dtype=torch.int32)},
    )
    backend.token_to_kv_pool = SimpleNamespace(get_swa_buffer=lambda layer: Mock())
    q = torch.ones(6, 64, 512, dtype=torch.bfloat16)
    out = backend._forward_swa(
        q, SimpleNamespace(layer_id=0, scaling=0.125), None, None
    )
    assert backend.legacy.call_args.args[0] is q
    assert backend.legacy.call_args.kwargs["layout_kv"] == "PA_ND"
    assert backend.legacy.call_args.kwargs["ori_mask_mode"] == 4
    assert backend.legacy.call_args.kwargs["kv_quant_mode"] == 1
    torch.testing.assert_close(out, q + 2)
    backend.public.attention.assert_not_called()


def test_graph_refresh_updates_logical_indices_in_place_and_keeps_padding_safe():
    backend = make_backend()
    fb, fm = initialize_metadata(backend, [129, 257])
    stable = fm.ori_sparse_indices
    pointer = stable.data_ptr()
    ctx = SimpleNamespace(
        fm=fm,
        graph_mode=SimpleNamespace(is_target_verify=lambda: True),
        is_idle_replay=False,
        live_seq_lens=torch.tensor([130], dtype=torch.int32),
        raw_bs=1,
        forward_batch=fb,
        tokens_per_bs=5,
    )
    backend._refresh_graph_dspark_sparse_metadata(ctx)
    assert fm.ori_sparse_indices is stable and stable.data_ptr() == pointer
    assert stable[0, 0, :133].tolist() == list(range(2, 135))
    assert (stable[:5, :, 133:] == -1).all()
    assert (stable[5:, :, 0] == 0).all()
    assert (stable[5:, :, 1:] == -1).all()
    ctx.is_idle_replay = True
    backend._refresh_graph_dspark_sparse_metadata(ctx)
    assert (stable[:, :, 0] == 0).all()
    assert (stable[:, :, 1:] == -1).all()
