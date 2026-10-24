"""QwenSparseAttnBackend wiring of AITER's fused FlyDSL QSA sparse decode (gfx950).

Kernel correctness is covered in AITER
(``op_tests/test_flydsl_qsa_sparse_decode.py``); this checks that the
backend hands the kernel the same operands as the Triton fallback.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl")
qsa_flydsl = pytest.importorskip("aiter.ops.flydsl.qsa_sparse_decode")

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _gfx950_available() -> bool:
    if not torch.cuda.is_available() or not is_gfx95_supported():
        return False
    probe = torch.empty(1, 1, 256, dtype=torch.bfloat16, device="cuda")
    return qsa_flydsl.can_use_flydsl_qsa_sparse_decode(probe, probe, probe)


pytestmark = pytest.mark.skipif(
    not _gfx950_available(), reason="gfx950 FlyDSL required"
)

_DIM = 256
# Qwen3.8-Flash-Next: 512 compressed blocks x ratio 4 + 3.
_TOPK = 2051
_MAX_CTX = 8192


def _make_inputs(rows, num_kv_heads, group, num_slots=40000, seed=0):
    """Random decode batch: sorted valid prefixes of varying length, then ``-1``
    padding and one masked entry at ``seq_len``; row 0 selects nothing."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(rows, num_kv_heads * group, _DIM, generator=gen, device="cuda")
    k = torch.randn(num_slots, num_kv_heads, _DIM, generator=gen, device="cuda")
    v = torch.randn(num_slots, num_kv_heads, _DIM, generator=gen, device="cuda")
    req_to_token = torch.randint(
        1,
        num_slots,
        (rows + 4, _MAX_CTX),
        generator=gen,
        device="cuda",
        dtype=torch.int32,
    )
    row_req = torch.randperm(rows + 4, generator=gen, device="cuda")[:rows]
    seq_lens = torch.randint(
        _TOPK // 2, _MAX_CTX, (rows,), generator=gen, device="cuda", dtype=torch.int32
    )
    indices = torch.full((rows, _TOPK), -1, dtype=torch.int32, device="cuda")
    for r in range(rows):
        seq_len = int(seq_lens[r])
        count = 0 if (r == 0 and rows > 1) else (r * 977) % _TOPK + 1
        count = min(count, seq_len)
        picked = torch.randperm(seq_len, generator=gen, device="cuda")[:count]
        indices[r, :count] = picked.sort().values.int()
        if count < _TOPK:
            indices[r, count] = seq_len
    return (
        q.bfloat16(),
        k.bfloat16(),
        v.bfloat16(),
        req_to_token,
        row_req,
        indices,
        seq_lens,
    )


def test_backend_decode_matches_triton_fallback(monkeypatch):
    """The backend must hand the kernel the same rows, lengths and request map
    the Triton fallback uses, including requests resolved per row."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_module

    rows, num_kv_heads, group = 6, 2, 12
    q, k, v, req_to_token, row_req, indices, seq_lens = _make_inputs(
        rows, num_kv_heads, group
    )
    backend = qsa_module.QwenSparseAttnBackend.__new__(qsa_module.QwenSparseAttnBackend)
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda layer_id: k, get_value_buffer=lambda layer_id: v
    )
    backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    backend.qsa_profile = None
    backend._cuda_graph_max_tokens = 0
    backend._fa2_scratch = {}
    backend._flydsl_decode_workspace = None
    metadata = SimpleNamespace(
        is_cuda_graph=False, sequence_lengths=seq_lens, row_req_pool_indices=row_req
    )
    monkeypatch.setattr(backend, "_resolve_metadata", lambda forward_batch: metadata)
    monkeypatch.setattr(qsa_module, "_resolve_trtllm_sparse_decode", lambda: None)
    layer = SimpleNamespace(layer_id=0, scaling=_DIM**-0.5)
    # Per-row requests win over the batch's; poison the batch's to prove it.
    forward_batch = SimpleNamespace(req_pool_indices=torch.zeros_like(row_req))

    outputs = {}
    for enabled in (False, True):
        qsa_module._resolve_flydsl_sparse_decode.cache_clear()
        with envs.SGLANG_USE_AITER.override(enabled):
            outputs[enabled] = backend._forward_paged_attention(
                q, layer, forward_batch, indices
            )
    qsa_module._resolve_flydsl_sparse_decode.cache_clear()
    assert backend._flydsl_decode_workspace is not None, "fused path did not run"
    # Row 0 selects nothing; only the fused kernel defines that output (zeros).
    torch.testing.assert_close(
        outputs[True][1:].float(), outputs[False][1:].float(), atol=1e-2, rtol=1e-2
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
