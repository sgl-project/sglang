"""Exercise the metadata-only gate without initializing a distributed server."""

import __future__

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load_gate(*, hip=True, gfx950=True):
    source = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/srt/layers/attention/linear/kda_backend.py"
    )
    tree = ast.parse(source.read_text())
    backend = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "KDAAttnBackend"
    )
    gate = next(
        n
        for n in backend.body
        if isinstance(n, ast.FunctionDef) and n.name == "_can_run_fused_chain_verify"
    )
    namespace = {
        "torch": torch,
        "is_hip": lambda: hip,
        "is_gfx95_supported": lambda: gfx950,
    }
    exec(
        compile(
            ast.Module(body=[gate], type_ignores=[]),
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[gate.name]


class _CudaMetadata:
    is_cuda = True

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)


def _inputs(batch, *, tokens=6, heads=16, head_dim=128, weight_dtype=torch.float32):
    def tensor(*shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")

    dim = 3 * heads * head_dim
    layer = SimpleNamespace(
        num_q_heads=heads,
        num_k_heads=heads,
        num_v_heads=heads,
        head_q_dim=head_dim,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        conv_weights=tensor(dim, 4, dtype=weight_dtype),
        bias=None,
        A_log=tensor(1, 1, heads, 1, dtype=torch.float32),
        dt_bias=tensor(heads * head_dim, dtype=torch.float32),
        lower_bound=-5.0,
    )
    return dict(
        layer=layer,
        mixed_qkv=_CudaMetadata(tensor(batch * tokens, dim)),
        a=tensor(batch * tokens, heads * head_dim),
        b=tensor(batch * tokens, heads),
        draft_token_num=tokens,
        conv_states=tensor(3, 3, dim),
        ssm_states=tensor(3, heads, head_dim, head_dim, dtype=torch.float32),
        intermediate_state_cache=tensor(
            batch, tokens, heads, head_dim, head_dim, dtype=torch.float32
        ),
        intermediate_conv_window_cache=tensor(batch, tokens, 3, dim),
        cache_indices=tensor(batch, dtype=torch.int32),
        intermediate_state_indices=tensor(batch, dtype=torch.int32),
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
        replayssm_rawv=None,
    )


@pytest.mark.parametrize(
    "batch,tokens",
    [
        (1, 6),  # supported min batch, T=6
        (16, 6),  # supported max batch, T=6
        (17, 6),  # unsupported batch boundary, T=6
        (16, 8),  # supported max batch, T=8
    ],
)
def test_hip_only_routes_supported_batch_range(batch, tokens):
    gate = _load_gate()
    backend = SimpleNamespace(_fused_chain_verify_fn=object())
    assert gate(backend, **_inputs(batch, tokens=tokens)) is (batch <= 16)


@pytest.mark.parametrize(
    "changes",
    [
        dict(tokens=4),
        dict(heads=8),
    ],
)
def test_hip_unmeasured_shapes_fall_back(changes):
    gate = _load_gate()
    backend = SimpleNamespace(_fused_chain_verify_fn=object())
    assert not gate(backend, **_inputs(4, **changes))


def test_hip_unmeasured_architecture_falls_back():
    gate = _load_gate(gfx950=False)
    backend = SimpleNamespace(_fused_chain_verify_fn=object())
    assert not gate(backend, **_inputs(4))


def test_cuda_keeps_existing_shape_coverage():
    gate = _load_gate(hip=False)
    backend = SimpleNamespace(_fused_chain_verify_fn=object())
    assert gate(backend, **_inputs(32, tokens=4, weight_dtype=torch.bfloat16))


def test_tree_and_disabled_paths_stay_unfused():
    gate = _load_gate()
    backend = SimpleNamespace(_fused_chain_verify_fn=None)
    kwargs = _inputs(4)
    assert not gate(backend, **kwargs)
    backend._fused_chain_verify_fn = object()
    kwargs["retrieve_parent_token"] = torch.empty(
        4, 6, dtype=torch.int32, device="meta"
    )
    assert not gate(backend, **kwargs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
