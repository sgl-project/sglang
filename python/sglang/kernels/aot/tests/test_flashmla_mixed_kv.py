import pytest
import torch

from sgl_kernel.flash_mla import (
    flash_mla_with_mixed_kvcache,
    get_mla_capabilities,
    get_mla_metadata,
)
from sglang.srt.layers.attention.dsv4.torch_quant import (
    cast_scale_inv_to_ue8m0,
    dequantize_dsv41_packed_main_kv,
    quantize_dsv41_packed_main_kv,
)


MAIN_LAYOUT = "DSV41_MAIN_KV_E2M1_BLOCK16_ROPE_BF16_V1"


def _quantize_v4(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_pages, page_slots, _, _ = k.shape
    nope = k.squeeze(2)[..., :448].float().view(num_pages, page_slots, 7, 64)
    scales = cast_scale_inv_to_ue8m0(nope.abs().amax(dim=-1) / 448.0)
    quantized = (nope / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)

    pages = torch.zeros(
        (num_pages, page_slots * 584), dtype=torch.uint8, device=k.device
    )
    data = pages[:, : page_slots * 576].view(num_pages, page_slots, 576)
    data[..., :448] = quantized.view(torch.uint8).reshape(
        num_pages, page_slots, 448
    )
    data[..., 448:] = (
        k.squeeze(2)[..., 448:].contiguous().view(torch.uint8)
    )
    scale_rows = pages[:, page_slots * 576 :].view(num_pages, page_slots, 8)
    scale_rows[..., :7] = scales.to(torch.float8_e8m0fnu).view(torch.uint8)

    decoded = torch.cat(
        (
            (
                quantized.to(torch.bfloat16)
                * scales.to(torch.bfloat16).unsqueeze(-1)
            ).reshape(num_pages, page_slots, 448),
            k.squeeze(2)[..., 448:],
        ),
        dim=-1,
    )
    return pages.view(num_pages, page_slots, 1, 584), decoded


def _gather(
    cache: torch.Tensor, indices: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    b, s_q, topk = indices.shape
    cache = cache.reshape(-1, 512)
    invalid = (indices < 0) | (indices >= cache.shape[0])
    invalid |= (
        torch.arange(topk, device=indices.device).view(1, 1, -1)
        >= lengths.view(b, 1, 1)
    )
    safe = indices.masked_fill(invalid, 0)
    return (
        cache.index_select(0, safe.reshape(-1)).view(b, s_q, topk, 512),
        invalid,
    )


def _reference(
    q,
    swa,
    swa_indices,
    swa_lengths,
    main,
    main_indices,
    main_lengths,
    sink,
    scale,
):
    swa, swa_invalid = _gather(swa, swa_indices, swa_lengths)
    main, main_invalid = _gather(main, main_indices, main_lengths)
    kv = torch.cat((swa, main), dim=2).float()
    invalid = torch.cat((swa_invalid, main_invalid), dim=2)
    logits = torch.matmul(q.float(), kv.transpose(-1, -2)) * scale
    logits.masked_fill_(invalid.unsqueeze(2), float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    out = torch.matmul(torch.exp(logits - lse.unsqueeze(-1)), kv)
    out *= (1 / (1 + torch.exp(sink.view(1, 1, -1) - lse))).unsqueeze(-1)
    empty = torch.isneginf(lse)
    out.masked_fill_(empty.unsqueeze(-1), 0)
    lse.masked_fill_(empty, float("inf"))
    return out.to(torch.bfloat16), lse.transpose(1, 2)


def _inputs(num_heads: int):
    torch.manual_seed(20260916 + num_heads)
    b, s_q, topk, swa_page_slots, main_page_slots = 2, 2, 64, 128, 128
    q = (torch.randn(b, s_q, num_heads, 512, device="cuda") / 10).to(
        torch.bfloat16
    )
    swa = (torch.randn(2, swa_page_slots, 1, 512, device="cuda") / 10).to(
        torch.bfloat16
    )
    main = (torch.randn(2, main_page_slots, 512, device="cuda") / 10).to(
        torch.bfloat16
    )
    swa_cache, swa_dequant = _quantize_v4(swa)
    main_cache = quantize_dsv41_packed_main_kv(main)
    main_dequant = dequantize_dsv41_packed_main_kv(main_cache, main_page_slots)
    swa_indices = torch.randint(
        0, 2 * swa_page_slots, (b, s_q, topk), dtype=torch.int32, device="cuda"
    )
    main_indices = torch.randint(
        0, 2 * main_page_slots, (b, s_q, topk), dtype=torch.int32, device="cuda"
    )
    swa_indices[..., -1] = -1
    main_indices[..., -2:] = -1
    swa_lengths = torch.tensor([57, 64], dtype=torch.int32, device="cuda")
    main_lengths = torch.tensor([61, 43], dtype=torch.int32, device="cuda")
    sink = torch.randn(num_heads, dtype=torch.float32, device="cuda")
    return (
        q,
        swa_cache,
        swa_dequant,
        swa_indices,
        swa_lengths,
        main_cache,
        main_dequant,
        main_indices,
        main_lengths,
        sink,
    )


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 required",
)


@pytest.mark.parametrize("num_heads", [64, 128])
def test_mixed_kv_binding_matches_reference(num_heads: int):
    (
        q,
        swa_cache,
        swa,
        swa_indices,
        swa_lengths,
        main_cache,
        main,
        main_indices,
        main_lengths,
        sink,
    ) = _inputs(num_heads)
    sched, _ = get_mla_metadata()
    scale = 512**-0.5
    out, lse = flash_mla_with_mixed_kvcache(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        main_cache_bytes=main_cache,
        main_indices=main_indices,
        swa_layout="V4",
        main_layout=MAIN_LAYOUT,
        main_page_slots=128,
        main_page_bytes=128 * 384,
        head_dim_v=512,
        tile_scheduler_metadata=sched,
        softmax_scale=scale,
        attn_sink=sink,
        swa_topk_length=swa_lengths,
        main_topk_length=main_lengths,
    )
    out_ref, lse_ref = _reference(
        q,
        swa,
        swa_indices,
        swa_lengths,
        main,
        main_indices,
        main_lengths,
        sink,
        scale,
    )
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=2.01 / 128)
    torch.testing.assert_close(lse, lse_ref, atol=1e-6, rtol=8.01 / 65536)


def test_mixed_kv_capability_is_versioned():
    capabilities = get_mla_capabilities()
    assert capabilities["mixed_kvcache_api_version"] == 1
    assert capabilities["mixed_kvcache_supported"]
    assert ["V4", MAIN_LAYOUT] in capabilities["supported_mixed_layout_pairs"]


def test_mixed_kv_cuda_graph_replay():
    (
        q,
        swa_cache,
        swa,
        swa_indices,
        swa_lengths,
        main_cache,
        main,
        main_indices,
        main_lengths,
        sink,
    ) = _inputs(64)
    q_static = q.clone()
    sched, _ = get_mla_metadata()
    kwargs = dict(
        q=q_static,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        main_cache_bytes=main_cache,
        main_indices=main_indices,
        swa_layout="V4",
        main_layout=MAIN_LAYOUT,
        main_page_slots=128,
        main_page_bytes=128 * 384,
        head_dim_v=512,
        tile_scheduler_metadata=sched,
        softmax_scale=512**-0.5,
        attn_sink=sink,
        swa_topk_length=swa_lengths,
        main_topk_length=main_lengths,
    )
    flash_mla_with_mixed_kvcache(**kwargs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = flash_mla_with_mixed_kvcache(**kwargs)

    q_static.copy_(q * 0.75)
    graph.replay()
    torch.cuda.synchronize()
    out_ref, lse_ref = _reference(
        q_static,
        swa,
        swa_indices,
        swa_lengths,
        main,
        main_indices,
        main_lengths,
        sink,
        512**-0.5,
    )
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=2.01 / 128)
    torch.testing.assert_close(lse, lse_ref, atol=1e-6, rtol=8.01 / 65536)
