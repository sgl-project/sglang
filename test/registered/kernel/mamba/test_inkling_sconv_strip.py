import pytest
import torch

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    fused_conv_strip_commit,
    fused_conv_window_scatter_with_mask,
    scatter_mamba_states_after_mtp_verify,
)
from sglang.srt.configs.inkling import (
    InklingConvCacheParams,
    InklingConvStateShape,
    InklingStateDType,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.models.inkling_common.kernels.sconv import (
    PAD_SLOT_ID,
    save_intermediate_conv_windows,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
DTYPE = torch.bfloat16

K = 4
KM1 = K - 1

CONV_DIMS = (128, 128, 64, 64, 1024, 1024)


class Problem:
    def __init__(
        self,
        *,
        bs: int,
        ndt: int,
        dim: int,
        seed: int,
        layers: int = 2,
        slots: int = 32,
        padded_bs: int | None = None,
        extra_pad_rows: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        self.layers, self.slots, self.bs, self.ndt, self.dim = (
            layers,
            slots,
            bs,
            ndt,
            dim,
        )
        self.padded_bs = bs if padded_bs is None else padded_bs
        self.conv_init = torch.randn(
            (layers, slots, KM1, dim), device=DEVICE, dtype=DTYPE
        )
        self.cache_indices = torch.full(
            (self.padded_bs,), PAD_SLOT_ID, device=DEVICE, dtype=torch.int32
        )

        self.cache_indices[:bs] = (torch.randperm(slots - 1)[:bs] + 1).to(
            DEVICE, torch.int32
        )
        self.hidden = torch.randn(
            (self.padded_bs * ndt + extra_pad_rows, dim), device=DEVICE, dtype=DTYPE
        )
        self.hidden[bs * ndt :] = float("nan")

        nan = float("nan")
        self.dense = torch.full(
            (layers, self.padded_bs + 1, ndt, KM1, dim), nan, device=DEVICE, dtype=DTYPE
        )
        self.strip = torch.full(
            (layers, self.padded_bs + 1, ndt, dim), nan, device=DEVICE, dtype=DTYPE
        )
        for layer in range(layers):
            for out in (self.dense[layer], self.strip[layer]):
                save_intermediate_conv_windows(
                    sconv_cache=self.conv_init[layer],
                    hidden_states=self.hidden_at(layer),
                    cache_indices=self.cache_indices,
                    intermediate_out=out,
                    batch_size=self.padded_bs,
                    draft_token_num=ndt,
                )

    def hidden_at(self, layer: int) -> torch.Tensor:
        return self.hidden if layer == 0 else self.hidden + layer

    def steps(self, accepted_lens: list[int]) -> torch.Tensor:
        return torch.tensor(accepted_lens, device=DEVICE, dtype=torch.int32) - 1

    def commit_dense(self, steps: torch.Tensor) -> torch.Tensor:
        conv = self.conv_init.clone()
        fused_conv_window_scatter_with_mask(conv, self.dense, self.cache_indices, steps)
        return conv

    def commit_strip(self, steps: torch.Tensor) -> torch.Tensor:
        conv = self.conv_init.clone()
        fused_conv_strip_commit(
            conv, self.strip, self.cache_indices, self.cache_indices, steps
        )
        return conv


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "bs,padded_bs,extra_pad,ndt,dim",
    [(63, 64, 0, 16, 1024), (63, 63, 1, 8, 96)],
)
def test_strip_commit_matches_dense(
    bs: int, padded_bs: int, extra_pad: int, ndt: int, dim: int
) -> None:
    p = Problem(
        bs=bs,
        padded_bs=padded_bs,
        ndt=ndt,
        dim=dim,
        slots=192,
        extra_pad_rows=extra_pad,
        seed=bs * 131 + ndt + dim,
    )
    patterns = [
        [1] * bs,
        [ndt] * bs,
        [0] * bs,
        [i % (ndt + 1) for i in range(bs)],
    ]
    for accepted in patterns:
        steps = p.steps(accepted + [1] * (padded_bs - bs))
        strip = p.commit_strip(steps)
        torch.testing.assert_close(
            strip, p.commit_dense(steps), rtol=0, atol=0, msg=f"{accepted=}"
        )
        assert not strip.isnan().any(), "pad rows leaked into a real slot"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dispatcher_commits_track_slot_before_working_slot() -> None:
    p = Problem(bs=4, ndt=8, dim=64, seed=43, slots=64)
    steps = p.steps([1, 2, 5, 8])

    track_steps = p.steps([1, 1, 2, 0])
    used = set(p.cache_indices.tolist())
    track_slots = torch.tensor(
        [s for s in range(p.slots) if s not in used][: p.bs],
        device=DEVICE,
        dtype=torch.int32,
    )

    ref = p.conv_init.clone()
    fused_conv_window_scatter_with_mask(ref, p.dense, p.cache_indices, steps)
    fused_conv_window_scatter_with_mask(ref, p.dense, track_slots, track_steps)

    conv = p.conv_init.clone()
    caches = MambaPool.SpeculativeState(
        conv=[conv],
        temporal=torch.empty(0, device=DEVICE),
        intermediate_ssm=None,
        intermediate_conv_window=[p.strip],
    )
    scatter_mamba_states_after_mtp_verify(
        caches, p.cache_indices, steps, track_slots, track_steps
    )
    torch.testing.assert_close(conv, ref, rtol=0, atol=0, msg="dispatcher ordering")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_replay_reads_fresh_inputs() -> None:
    p = Problem(bs=4, ndt=8, dim=64, seed=47)
    conv = p.conv_init.clone()
    steps = p.steps([1] * p.bs)
    idx = p.cache_indices

    def body() -> None:
        for layer in range(p.layers):
            save_intermediate_conv_windows(
                sconv_cache=conv[layer],
                hidden_states=p.hidden,
                cache_indices=idx,
                intermediate_out=p.strip[layer],
                batch_size=p.bs,
                draft_token_num=p.ndt,
            )
        fused_conv_strip_commit(conv, p.strip, idx, idx, steps)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            body()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    for seed, accepted in ((101, [3, 8, 1, 5]), (202, [8, 1, 4, 0])):
        torch.manual_seed(seed)
        p.hidden.copy_(torch.randn_like(p.hidden))
        steps.copy_(p.steps(accepted))
        conv.copy_(p.conv_init)
        graph.replay()
        got = conv.clone()

        ref = p.conv_init.clone()
        for layer in range(p.layers):
            save_intermediate_conv_windows(
                sconv_cache=ref[layer],
                hidden_states=p.hidden,
                cache_indices=idx,
                intermediate_out=p.dense[layer],
                batch_size=p.bs,
                draft_token_num=p.ndt,
            )
        fused_conv_window_scatter_with_mask(ref, p.dense, idx, steps)
        torch.testing.assert_close(got, ref, rtol=0, atol=0, msg=f"replay {seed=}")


@pytest.mark.parametrize("strip", [False, True])
def test_pool_intermediate_allocation_matches_runner_reserve(strip: bool) -> None:
    ndt, slots, layers = 8, 16, [0, 1, 2]
    params = InklingConvCacheParams(
        shape=InklingConvStateShape(
            conv=[(KM1, d) for d in CONV_DIMS],
            temporal=(0, 0, 0),
            conv_intermediate_strip=strip,
        ),
        layers=layers,
        dtype=InklingStateDType(conv=DTYPE, temporal=DTYPE),
    )
    pool = MambaPool(
        size=slots,
        spec_state_size=slots,
        cache_params=params,
        mamba_layer_ids=layers,
        device="cpu",
        speculative_num_draft_tokens=ndt,
    )
    inter = pool.get_speculative_mamba2_params_all_layers().intermediate_conv_window
    if strip:
        expected = [(len(layers), slots + 1, ndt, d) for d in CONV_DIMS]
    else:
        expected = [(len(layers), slots + 1, ndt, KM1, d) for d in CONV_DIMS]
    assert [tuple(t.shape) for t in inter] == expected
    assert all(t.dtype == DTYPE for t in inter)
    assert all(t.device.type == "cpu" for t in inter)

    per_req = params.spec_intermediate_bytes_per_req(ndt)
    assert sum(t.numel() * t.element_size() for t in inter) == per_req * (slots + 1)


def test_strip_layout_requires_a_linear_draft_chain() -> None:
    params = InklingConvCacheParams(
        shape=InklingConvStateShape(
            conv=[(KM1, 64)], temporal=(0, 0, 0), conv_intermediate_strip=True
        ),
        layers=[0],
        dtype=InklingStateDType(conv=DTYPE, temporal=DTYPE),
    )
    with pytest.raises(ValueError, match="linear draft chain"):
        MambaPool(
            size=4,
            spec_state_size=4,
            cache_params=params,
            mamba_layer_ids=[0],
            device="cpu",
            speculative_num_draft_tokens=8,
            speculative_eagle_topk=2,
        )


def test_strip_saves_request_rows_before_flat_padding() -> None:
    batch_size, draft_tokens, dim, extra_pad = 3, 4, 2, 3
    real_rows = batch_size * draft_tokens
    hidden = torch.arange((real_rows + extra_pad) * dim, dtype=DTYPE).view(-1, dim)
    output = torch.full((batch_size + 1, draft_tokens, dim), -1, dtype=DTYPE)
    save_intermediate_conv_windows(
        sconv_cache=torch.zeros((batch_size + 1, KM1, dim), dtype=DTYPE),
        hidden_states=hidden,
        cache_indices=torch.arange(batch_size, dtype=torch.int32),
        intermediate_out=output,
        batch_size=batch_size,
        draft_token_num=draft_tokens,
    )
    torch.testing.assert_close(
        output[:batch_size], hidden[:real_rows].view(batch_size, draft_tokens, dim)
    )
    assert torch.all(output[batch_size] == -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_strip_commit_accepts_unified_pool_strides():
    p = Problem(bs=4, ndt=6, dim=128, seed=57)
    storage = p.conv_init.transpose(0, 1).contiguous()
    conv = storage.transpose(0, 1)
    steps = p.steps([1, 2, 4, 6])
    assert not conv.is_contiguous()
    fused_conv_strip_commit(conv, p.strip, p.cache_indices, p.cache_indices, steps)
    torch.testing.assert_close(conv, p.commit_dense(steps), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_attention_prologue_writes_strips():
    draft_tokens = 6
    from sglang.kernels.ops.attention.inkling_attn_prologue import (
        inkling_attn_prologue_verify,
    )

    torch.manual_seed(61)
    batch, dq, dkv, slots = 3, 256, 128, 32
    rows = batch * draft_tokens
    qkvr = torch.randn(rows, dq + 2 * dkv, device=DEVICE, dtype=DTYPE)
    kcache = torch.randn(slots, 3, dkv, device=DEVICE, dtype=DTYPE)
    vcache = torch.randn_like(kcache)
    idx = torch.tensor([2, 5, -1], device=DEVICE, dtype=torch.int32)
    mask = torch.tensor([True, False, False], device=DEVICE)
    kw = torch.randn(dkv, 4, device=DEVICE, dtype=DTYPE)
    vw = torch.randn_like(kw)
    qg = torch.randn(128, device=DEVICE, dtype=DTYPE)
    kg = torch.randn_like(qg)
    loc = torch.arange(rows, device=DEVICE, dtype=torch.int64)
    outputs, caches, intermediates = [], [], []
    for strip in [False, True]:
        shape = (batch, draft_tokens, dkv) if strip else (batch, draft_tokens, 3, dkv)
        ki = torch.full(shape, -99, device=DEVICE, dtype=DTYPE)
        vi = torch.full_like(ki, -99)
        kb = torch.zeros(slots, 1, dkv, device=DEVICE, dtype=DTYPE)
        vb = torch.zeros_like(kb)
        outputs.append(
            inkling_attn_prologue_verify(
                qkvr,
                kcache,
                vcache,
                idx,
                mask,
                kw,
                vw,
                ki,
                vi,
                qg,
                kg,
                1e-6,
                loc,
                kb,
                vb,
                0,
                dq,
                dq + dkv,
                dq,
                dkv,
                draft_tokens,
            )
        )
        caches.append((kb, vb))
        intermediates.append((ki, vi))
    for dense, strip in zip(outputs[0][:3] + caches[0], outputs[1][:3] + caches[1]):
        torch.testing.assert_close(strip, dense, rtol=0, atol=0)
    for dense, strip in zip(intermediates[0], intermediates[1]):
        torch.testing.assert_close(strip[:2], dense[:2, :, -1], rtol=0, atol=0)
        assert torch.all(strip[2] == -99)


@pytest.mark.parametrize("strip", [False, True])
def test_inkling_unified_factory_propagates_strip_layout(strip):
    from sglang.srt.mem_cache.unified_memory_pool import init_unified_mamba_swa_pools

    params = InklingConvCacheParams(
        shape=InklingConvStateShape(
            conv=[(3, 64)],
            temporal=(0, 0, 0),
            conv_intermediate_strip=strip,
        ),
        layers=[0, 1],
    )
    bundle = init_unified_mamba_swa_pools(
        device="cpu",
        kv_cache_dtype=DTYPE,
        head_num=1,
        head_dim=64,
        v_head_dim=64,
        swa_head_num=1,
        swa_head_dim=64,
        swa_v_head_dim=64,
        page_size=1,
        start_layer=0,
        end_layer=2,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        mamba_layer_ids=[0, 1],
        mamba2_cache_params=params,
        full_max_total_num_tokens=64,
        swa_max_total_num_tokens=32,
        max_mamba_cache_size=8,
        model_context_len=16,
        extra_max_context_len=6,
        max_num_reqs=4,
        enable_memory_saver=False,
        enable_mamba_extra_buffer=True,
        disable_overlap_schedule=True,
        need_sort=False,
        speculative_num_draft_tokens=6,
    )
    pool = bundle.req_to_token_pool.mamba_pool
    cache = pool.get_speculative_mamba2_params_all_layers()
    assert not cache.conv[0].is_contiguous()
    assert cache.intermediate_conv_window[0].shape == (
        (2, 5, 6, 64) if strip else (2, 5, 6, 3, 64)
    )
