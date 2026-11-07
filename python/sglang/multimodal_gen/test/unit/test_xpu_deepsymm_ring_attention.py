# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import torch.multiprocessing as mp

HEAD_SIZE = 128
PRECISION_ENV = "SGLANG_DIFFUSION_XPU_DEEPSYMM_RING_PRECISION"
_MAX_REL_L2 = {"bf16": 0.02, "fp8": 0.1, "fp8_mixed": 0.3, "mxfp4": 0.4}


def _xpu_count():
    return torch.xpu.device_count() if torch.xpu.is_available() else 0


def _impl(*, num_kv_heads=40, head_size=HEAD_SIZE, causal=False):
    pytest.importorskip("sgl_kernel.flash_attn", reason="needs sgl-kernel-xpu")
    from sglang.multimodal_gen.runtime.layers.attention.backends.xpu_backend import (
        XPUAttentionImpl,
    )

    return XPUAttentionImpl(
        num_heads=40,
        head_size=head_size,
        causal=causal,
        softmax_scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
    )


def _precision(query, impl, *, causal=False, is_cross_attention=False):
    from sglang.multimodal_gen.runtime.layers.attention.backends.xpu_deepsymm_ring import (
        deepsymm_ring_precision,
    )

    return deepsymm_ring_precision(
        query, impl, causal=causal, is_cross_attention=is_cross_attention
    )


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(PRECISION_ENV, raising=False)
    q = torch.empty(1, 8, 40, HEAD_SIZE, dtype=torch.bfloat16)
    assert _precision(q, object()) is None


def test_unknown_precision_raises(monkeypatch):
    monkeypatch.setenv(PRECISION_ENV, "int8")
    q = torch.empty(1, 8, 40, HEAD_SIZE, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=PRECISION_ENV):
        _precision(q, object())


@pytest.mark.parametrize(
    "impl_kwargs, call_kwargs, device, dtype, expected",
    [
        ({}, {}, "xpu", torch.bfloat16, "mxfp4"),
        ({"num_kv_heads": 8}, {}, "xpu", torch.bfloat16, "bf16"),
        ({"head_size": 64}, {}, "xpu", torch.bfloat16, "bf16"),
        ({"head_size": 96}, {}, "xpu", torch.bfloat16, None),
        ({}, {"causal": True}, "xpu", torch.bfloat16, None),
        ({}, {"is_cross_attention": True}, "xpu", torch.bfloat16, None),
        ({}, {}, "xpu", torch.float16, None),
        ({}, {}, "cpu", torch.bfloat16, None),
    ],
)
def test_precision_rules(
    monkeypatch, impl_kwargs, call_kwargs, device, dtype, expected
):
    if _xpu_count() < 1:
        pytest.skip("needs an XPU")
    monkeypatch.setenv(PRECISION_ENV, "mxfp4")
    impl = _impl(**impl_kwargs)
    q = torch.empty(1, 8, 40, impl.head_size, dtype=dtype, device=device)
    assert _precision(q, impl, **call_kwargs) == expected


def _reference(q, k, v):
    q, k, v = (t.float().transpose(1, 2) for t in (q, k, v))
    return torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2)


def _ring_worker(rank, world_size, port, seq_per_rank, num_heads):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.xpu.set_device(rank)
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        destroy_model_parallel,
        get_sp_group,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.xpu_deepsymm_ring import (
        deepsymm_ring_attn,
    )

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=world_size, ring_degree=world_size
    )
    generator = torch.Generator().manual_seed(1234)
    shape = (1, seq_per_rank * world_size, num_heads, HEAD_SIZE)
    q, k, v = (torch.randn(shape, generator=generator).bfloat16() for _ in range(3))
    local = slice(rank * seq_per_rank, (rank + 1) * seq_per_rank)
    expected = _reference(q[:, local], k, v)
    device = torch.device("xpu", rank)
    for precision, max_rel_l2 in _MAX_REL_L2.items():
        for _ in range(2):
            out = deepsymm_ring_attn(
                q[:, local].to(device),
                k[:, local].to(device),
                v[:, local].to(device),
                ring_group=get_sp_group().ring_group,
                precision=precision,
            )
        assert out.shape == expected.shape and out.dtype == torch.bfloat16
        rel_l2 = ((out.cpu().float() - expected).norm() / expected.norm()).item()
        print(f"rank {rank} {precision}: rel_l2 {rel_l2:.4f}", flush=True)
        assert rel_l2 < max_rel_l2, (precision, rel_l2)
    destroy_model_parallel()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("seq_per_rank, num_heads", [(2048, 8), (4725, 40)])
def test_two_rank_ring_matches_full_attention(seq_per_rank, num_heads):
    if _xpu_count() < 2:
        pytest.skip("needs two XPUs")
    pytest.importorskip("deep_symm.ring_attn", reason="needs deep_symm")
    from sglang.srt.utils.network import get_free_port_below_ephemeral

    mp.spawn(
        _ring_worker,
        args=(2, get_free_port_below_ephemeral(), seq_per_rank, num_heads),
        nprocs=2,
    )
