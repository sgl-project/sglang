import sys
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.attention.linear.kda_backend as kda_backend
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load_gate(monkeypatch, *, hip=True, gfx950=True):
    monkeypatch.setattr(kda_backend, "is_hip", lambda: hip)
    monkeypatch.setattr(kda_backend, "is_gfx95_supported", lambda: gfx950)
    return KDAAttnBackend._can_run_fused_chain_verify


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
        replayssm_rawk=None,
        replayssm_g=None,
        replayssm_beta=None,
    )


@pytest.mark.parametrize(
    "gfx950,batch,changes,expected",
    [
        (True, 1, {}, True),
        (True, 16, dict(tokens=8), True),
        (True, 17, {}, False),
        (True, 4, dict(tokens=4), False),
        (True, 4, dict(weight_dtype=torch.bfloat16), False),
        (False, 4, {}, False),
    ],
)
def test_hip_routes_only_measured_shapes(monkeypatch, gfx950, batch, changes, expected):
    gate = _load_gate(monkeypatch, gfx950=gfx950)
    backend = SimpleNamespace(_fused_chain_verify_fn=object())
    assert gate(backend, **_inputs(batch, **changes)) is expected


def test_cuda_keeps_existing_shape_coverage(monkeypatch):
    gate = _load_gate(monkeypatch, hip=False)
    backend = SimpleNamespace(_fused_chain_verify_fn=object())
    assert gate(backend, **_inputs(32, tokens=4, weight_dtype=torch.bfloat16))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
