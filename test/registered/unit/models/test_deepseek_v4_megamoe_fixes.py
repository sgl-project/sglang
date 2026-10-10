"""CPU tests for the DeepSeek-V4 MegaMoE fixes at attention TP > 1: the capture-safe all-gather of
uneven chunks after an a2a MoE, the MLP-sync padding rows in the bounded-replay tail and in the
request indices, and the per-rank MegaMoE token gate.

Each test loads the function under test from its source file (ast) and runs it with small stubs,
so it needs no GPU and no process group.

    python -m pytest test/registered/unit/models/test_deepseek_v4_megamoe_fixes.py -v
"""

import __future__

import ast
import pathlib
import textwrap
import types

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PY = pathlib.Path(__file__).resolve().parents[4] / "python"
V4 = "sglang/srt/models/deepseek_v4.py"
HIP = "sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py"
SPARSE = "sglang/srt/layers/attention/dsv4/dsv41_sparse.py"
MEGA = "sglang/srt/layers/moe/mega_moe_flydsl.py"
KERN = "sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py"
CUDA_BACKEND = "sglang/srt/layers/attention/deepseek_v4_backend.py"
FLAGS = __future__.annotations.compiler_flag


def _tree(rel):
    src = (PY / rel).read_text()
    return src, ast.parse(src)


def _find(tree, name, cls=None):
    nodes = tree.body
    if cls:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls
        ).body
    return next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)


def load(rel, names, glb, cls=None):
    """Exec the named top-level functions (or methods of cls) of a source file into glb."""
    src, tree = _tree(rel)
    for name in names:
        code = textwrap.dedent(ast.get_source_segment(src, _find(tree, name, cls)))
        exec(
            compile(code, f"{rel}:{name}", "exec", flags=FLAGS, dont_inherit=True), glb
        )
    return glb


# ---------- capture-safe all-gather of uneven chunks (deepseek_v4.py) ----------


def load_gather():
    """The a2a-scatter gather block at the end of _run_moe_ffn_dp_sync, as a function."""
    src, tree = _tree(V4)
    meth = _find(tree, "_run_moe_ffn_dp_sync", "DeepseekV4DecoderLayer")
    block = next(
        n
        for n in meth.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "_use_tp_attn_a2a_scatter"
        and "gather" in ast.get_source_segment(src, n)
    )
    body = textwrap.indent(textwrap.dedent(ast.get_source_segment(src, block)), "    ")
    return (
        "def gather(hidden_states, _a2a_scatter_chunks, _use_tp_attn_a2a_scatter=True):\n"
        + body
        + "\n    return hidden_states\n"
    )


class FakeGroup:
    """One rank's view of an attention TP group whose ranks hold `parts` (unequal row counts)."""

    def __init__(self, parts, rank):
        self.parts, self.rank = parts, rank

    def all_gather_into_tensor(self, out, local):
        m = local.shape[0]
        assert out.shape[0] == m * len(self.parts)
        assert all(p.shape[0] <= m for p in self.parts)
        padded = [
            torch.nn.functional.pad(p, (0, 0, 0, m - p.shape[0])) for p in self.parts
        ]
        assert torch.equal(local, padded[self.rank])
        out.copy_(torch.cat(padded))


def run_gather(sizes, rank):
    parts = [torch.randn(n, 8) for n in sizes]
    group = FakeGroup(parts, rank)
    glb = {
        "torch": torch,
        "get_parallel": lambda: types.SimpleNamespace(attn_tp_group=group),
    }
    exec(load_gather(), glb)
    chunks = [torch.empty(n, 8) for n in sizes]
    return glb["gather"](parts[rank].clone(), chunks), torch.cat(parts)


@pytest.mark.parametrize(
    "sizes", [[3, 2, 3, 1], [2, 2, 2, 2], [2, 0, 1, 0], [1, 1, 1, 0]]
)
@pytest.mark.parametrize("rank", [0, 1, 3])
def test_gather_uneven_sizes_matches_concat(sizes, rank):
    out, ref = run_gather(sizes, rank)
    assert torch.equal(out, ref)


def test_gather_uses_tensor_collective_only():
    code = load_gather()
    assert "all_gather_into_tensor" in code and "attn_tp_all_gather" not in code


# ---------- tail rows with MLP-sync padding (deepseek_v4_backend_hip_radix.py) ----------


class _Stop(Exception):
    pass


def tail_out_cache_loc(extend_lens, padded_rows):
    glb = {"torch": torch, "SWA_WINDOW": 128}
    load(KERN, ["late_layer_tail_layout"], glb)
    load(CUDA_BACKEND, ["_tail_rows"], glb)
    load(HIP, ["_build_late_layer_tail_metadata"], glb, cls="DeepseekV4HipRadixBackend")
    seen = {}

    def init_prefill(**kw):
        seen.update(kw)
        raise _Stop

    self = types.SimpleNamespace(init_forward_metadata_prefill=init_prefill)
    real = sum(extend_lens)
    fb = types.SimpleNamespace(
        extend_seq_lens_cpu=extend_lens,
        seq_lens_cpu=torch.tensor(extend_lens),
        out_cache_loc=torch.arange(100, 100 + real + padded_rows),
        positions=torch.arange(real + padded_rows),
        req_pool_indices=torch.arange(len(extend_lens)),
        seq_lens=torch.tensor(extend_lens),
    )
    with pytest.raises(_Stop):
        glb["_build_late_layer_tail_metadata"](self, fb)
    return seen["out_cache_loc"], seen["num_tokens"]


def test_tail_rows_padded_one_request_drops_padding():
    loc, n = tail_out_cache_loc([6], padded_rows=2)
    assert n == 6 and torch.equal(loc, torch.arange(100, 106))


@pytest.mark.parametrize(
    "lens,rows", [([6], (100, 106)), ([200], (172, 300)), ([3, 4], (100, 107))]
)
def test_tail_rows_unpadded(lens, rows):
    loc, _ = tail_out_cache_loc(lens, padded_rows=0)
    assert torch.equal(loc, torch.arange(*rows))


# ---------- request index of padding rows (dsv41_sparse.py) ----------


def req_indices(req, lens, num_tokens):
    glb = load(SPARSE, ["token_req_indices"], {"torch": torch})
    mode = types.SimpleNamespace(
        is_decode=lambda: False, is_target_verify=lambda: False, is_extend=lambda: True
    )
    fb = types.SimpleNamespace(
        req_pool_indices=torch.tensor(req),
        forward_mode=mode,
        extend_seq_lens_cpu=lens,
        extend_seq_lens=torch.tensor(lens),
    )
    return glb["token_req_indices"](fb, num_tokens=num_tokens)


def test_req_indices_padded_rows_use_request_zero():
    out = req_indices([3, 5], [2, 3], num_tokens=8)
    assert out.tolist() == [3, 3, 5, 5, 5, 0, 0, 0]


@pytest.mark.parametrize("num_tokens", [None, 5])
def test_req_indices_unpadded(num_tokens):
    assert req_indices([3, 5], [2, 3], num_tokens).tolist() == [3, 3, 5, 5, 5]


# ---------- per-rank MegaMoE gate (mega_moe_flydsl.py) ----------


def gate(global_tokens, attn_tp, mtpr=8192, rows=1, capture=False):
    envs = types.SimpleNamespace(
        SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR=types.SimpleNamespace(get=lambda: mtpr)
    )
    glb = {
        "torch": torch,
        "envs": envs,
        "get_moe_a2a_backend": lambda: types.SimpleNamespace(is_megamoe=lambda: True),
        "get_is_capture_mode": lambda: capture,
        "get_dp_global_num_tokens": lambda: global_tokens,
        "is_dsa_enable_prefill_cp": lambda: False,
        "get_parallel": lambda: types.SimpleNamespace(attn_tp_size=attn_tp),
    }
    load(MEGA, ["_mtpr", "should_use_mega_moe"], glb)
    moe = types.SimpleNamespace(
        experts=types.SimpleNamespace(_mega_moe_weights_built=True)
    )
    return glb["should_use_mega_moe"](moe, torch.empty(rows, 8))


@pytest.mark.parametrize(
    "tokens,tp,expect",
    [
        ([32768], 4, True),  # 8192 per rank = MTPR
        ([32769], 4, False),  # ceil -> 8193 per rank
        ([16384], 4, True),  # a 16k chunk: the global count alone would exceed MTPR
        ([8193], 4, True),  # ceil -> 2049
        ([8192], 1, True),
        ([8193], 1, False),
        ([100, 32768], 4, True),
        ([32772, 3], 4, False),
    ],
)
def test_gate_per_rank_ceil_div_around_mtpr(tokens, tp, expect):
    assert gate(tokens, tp) is expect


def test_gate_without_global_counts_uses_local_rows():
    assert gate(None, 4, rows=8192) is True
    assert gate(None, 4, rows=8193) is False


def test_gate_capture_mode_always_true():
    assert gate([10**6], 4, capture=True) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
