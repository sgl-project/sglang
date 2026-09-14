"""Small-head attention against dequantized FP32, with dynamic paged inputs."""

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.small_paged_attention import (
    small_paged_attention,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="nightly", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="The production fast paths target SM10x",
)


def cache(page, blocks=9):
    x = torch.randn(blocks, page, 512, device="cuda", dtype=torch.bfloat16)
    size = ((page * 584 + 575) // 576) * 576
    raw = torch.zeros(blocks, size, device="cuda", dtype=torch.uint8)
    data = raw[:, : page * 576].view(blocks, page, 576)
    scales = raw[:, page * 576 : page * 584].view(blocks, page, 8)
    deq = x.clone()
    for i in range(7):
        a = x[:, :, i * 64 : (i + 1) * 64].float()
        sf = torch.exp2((a.abs().amax(-1) / 448).clamp_min(1e-4).log2().ceil())
        quant = (a / sf[:, :, None]).to(torch.float8_e4m3fn)
        data[:, :, i * 64 : (i + 1) * 64] = quant.view(torch.uint8)
        scales[:, :, i] = sf.to(torch.float8_e8m0fnu).view(torch.uint8)
        deq[:, :, i * 64 : (i + 1) * 64] = (quant.float() * sf[:, :, None]).to(
            torch.bfloat16
        )
    data[:, :, 448:] = x[:, :, 448:].contiguous().view(torch.uint8)
    return raw[:, : page * 584].view(blocks, page, 1, 584), deq.reshape(-1, 512).float()


def reference(q, kb, ids, lens, sink, eb=None, ei=None, el=None):
    def gather(x, i, l):
        i = i[:, 0]
        valid = (
            (i >= 0)
            & (i < x.shape[0])
            & (torch.arange(i.shape[1], device=i.device)[None, :] < l[:, None])
        )
        return x[i.clamp(0, x.shape[0] - 1).long()], valid

    kv, valid = gather(kb, ids, lens)
    if eb is not None:
        ek, ev = gather(eb, ei, el)
        kv = torch.cat([kv, ek], 1)
        valid = torch.cat([valid, ev], 1)
    scores = (q[:, 0, :16].float() @ kv.transpose(1, 2)) * 512**-0.5
    scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
    with_sink = torch.cat([scores, sink[None, :16, None].expand(q.shape[0], -1, -1)], 2)
    prob = torch.softmax(with_sink, dim=-1)[:, :, :-1]
    # +inf sink and the all-masked/-inf-sink case define zero output.
    prob = torch.nan_to_num(prob)
    return prob @ kv


@pytest.mark.parametrize(
    "b,k,ek", [(1, 128, 0), (5, 192, 0), (6, 128, 512), (6, 128, 1024), (8, 128, 512)]
)
@pytest.mark.parametrize("page", [128, 256])
@pytest.mark.parametrize("cache_dtype", [torch.uint8, torch.float8_e4m3fn])
def test_dynamic_graph(b, k, ek, page, cache_dtype):
    torch.manual_seed(943)
    kv, kb = cache(page)
    ev, eb = cache(128)
    kv, ev = kv.view(cache_dtype), ev.view(cache_dtype)
    q = torch.randn(b, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, kb.shape[0], (b, 1, k), device="cuda", dtype=torch.int32)
    ei = (
        torch.randint(0, eb.shape[0], (b, 1, ek), device="cuda", dtype=torch.int32)
        if ek
        else None
    )
    lens = torch.full((b,), k, device="cuda", dtype=torch.int32)
    el = torch.full((b,), ek, device="cuda", dtype=torch.int32) if ek else None
    sink = torch.randn(64, device="cuda")

    def run():
        return small_paged_attention(q, kv, ids, lens, sink, ev if ek else None, ei, el)

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for case in range(4):
        q.normal_()
        ids[:, :, ::7] = -1
        ids[:, :, ::11] = kb.shape[0] + 31
        if ek:
            ei[:, :, ::5] = -1
            ei[:, :, ::13] = eb.shape[0] + 77
        lens.copy_(torch.arange(b, device="cuda") * 37 % (k + 1))
        if ek:
            el.copy_(torch.arange(b, device="cuda") * 91 % (ek + 1))
        if case == 0:
            lens.fill_(k)
            if ek:
                el.fill_(ek)
        if case == 1:
            lens.fill_(1)
            if ek:
                el.zero_()
        if case == 2:
            lens.zero_()
            if ek:
                el.zero_()
        sink[0] = torch.inf
        sink[1] = -torch.inf
        sink[2] = 1000
        graph.replay()
        gold = reference(q, kb, ids, lens, sink, eb if ek else None, ei, el)
        assert torch.isfinite(out).all()
        rms = (
            (
                (out.float() - gold).square().mean()
                / gold.square().mean().clamp_min(1e-30)
            )
            .sqrt()
            .item()
        )
        assert rms < 0.005, rms


@pytest.mark.parametrize(
    "b,k,ek", [(1, 128, 512), (5, 192, 0), (6, 128, 512), (8, 128, 1024)]
)
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
def test_inverse_rope_dynamic(b, k, ek, pos_dtype):
    from sglang.kernels.ops.attention.dsv4.elementwise import fused_rope_inplace

    torch.manual_seed(19214)
    kv, kb = cache(256)
    ev, eb = cache(128)
    q = torch.randn(b, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, kb.shape[0], (b, 1, k), device="cuda", dtype=torch.int32)
    ei = (
        torch.randint(0, eb.shape[0], (b, 1, ek), device="cuda", dtype=torch.int32)
        if ek
        else None
    )
    lens = torch.full((b,), k, device="cuda", dtype=torch.int32)
    el = torch.full((b,), ek, device="cuda", dtype=torch.int32) if ek else None
    sink = torch.randn(64, device="cuda")
    angles = torch.randn(65536, 32, device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)
    pos = torch.randint(0, 65536, (b,), device="cuda", dtype=pos_dtype)
    args = (q, kv, ids, lens, sink, ev if ek else None, ei, el)
    for _ in range(3):
        small_paged_attention(*args, inverse_rope=(freqs, pos))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = small_paged_attention(*args, inverse_rope=(freqs, pos))
    for case in range(4):
        q.normal_()
        pos.random_(0, 65536)
        if case:
            ids[:, :, ::7] = -1
            lens.copy_(torch.arange(b, device="cuda") * 31 % (k + 1))
            if ek:
                el.copy_(torch.arange(b, device="cuda") * 93 % (ek + 1))
        if case == 3:
            lens.zero_()
            if ek:
                el.zero_()
        graph.replay()
        expected = small_paged_attention(*args)
        fused_rope_inplace(
            expected[..., -64:], None, freqs, positions=pos, inverse=True
        )
        # Zero signs can differ after the rotation; all numerical values must match.
        assert torch.equal(out, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Backend fallback uses the SM10x paged FlashMLA implementation",
)
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("breakable", [False, True])
def test_backend_inverse_rope_contract(enabled, pos_dtype, breakable):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from sgl_kernel.flash_mla import FlashMLASchedMeta

    from sglang.kernels.ops.attention.dsv4.elementwise import fused_rope_inplace
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        DeepseekV4AttnBackend,
        DSV4AttnMetadata,
    )
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        enable_breakable_cuda_graph,
    )
    from sglang.srt.runtime_context import restore_context, snapshot_context
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    state = snapshot_context()
    try:
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=256)
        )
        b = 6
        kv, kb = cache(128)
        raw = kv.as_strided((kv.shape[0], kv.stride(0)), (kv.stride(0), 1))
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.swa_page_size = 128
        pool.get_swa_key_buffer_radix = lambda layer_id: raw.view(torch.float8_e4m3fn)
        pool.get_swa_key_bytes_per_token = lambda: 584
        core = DSV4AttnMetadata.__new__(DSV4AttnMetadata)
        core.swa_page_indices = torch.randint(
            0, kb.shape[0], (b, 128), device="cuda", dtype=torch.int32
        )
        core.swa_topk_lengths = torch.full((b,), 128, device="cuda", dtype=torch.int32)
        core.c0_flashmla_metadata = FlashMLASchedMeta()
        backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
        backend.token_to_kv_pool = pool
        backend.forward_metadata = SimpleNamespace(
            core_attn_metadata=core, late_layer_tail=None
        )
        backend.mtp_enabled = False
        backend.trtllm_attn = False
        backend.small_paged_attention_enabled = enabled
        backend.head_dim_v = 512
        backend.softmax_scale = 512**-0.5
        q = torch.randn(b, 64, 512, device="cuda", dtype=torch.bfloat16)
        sink = torch.randn(64, device="cuda")
        pos = torch.randint(0, 1024, (b,), device="cuda", dtype=pos_dtype)
        angles = torch.randn(1024, 32, device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        args = dict(
            q=q,
            k=q,
            v=q,
            layer=SimpleNamespace(layer_id=0, tp_q_head_num=16, v_head_dim=512),
            forward_batch=SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY),
            compress_ratio=0,
            save_kv_cache=False,
            attn_sink=sink,
        )
        with enable_breakable_cuda_graph() if breakable else nullcontext():
            expected = backend._forward_attention(**args)
            fused_rope_inplace(
                expected[:, :16, -64:], None, freqs, positions=pos, inverse=True
            )
            out = backend._forward_attention(**args, inverse_rope=(freqs, pos))
            assert out.shape == (b, 16 if enabled and not breakable else 64, 512)
            assert torch.equal(out, expected)
    finally:
        restore_context(state)
