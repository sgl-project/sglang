"""CPU tests for the DeepSeek-V4 MegaMoE fixes at attention TP > 1: the MLP-sync padding rows in the
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
SPARSE = "sglang/srt/layers/attention/dsv4/dsv41_sparse.py"
MEGA = "sglang/srt/layers/moe/mega_moe_flydsl.py"
FLAGS = __future__.annotations.compiler_flag


def _tree(rel):
    src = (PY / rel).read_text()
    return src, ast.parse(src)


def _find(tree, name):
    return next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def load(rel, names, glb):
    """Exec the named top-level functions of a source file into glb."""
    src, tree = _tree(rel)
    for name in names:
        code = textwrap.dedent(ast.get_source_segment(src, _find(tree, name)))
        exec(
            compile(code, f"{rel}:{name}", "exec", flags=FLAGS, dont_inherit=True), glb
        )
    return glb


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
