"""Decode LoRA launch reduction (SGLANG_OPT_LORA_DECODE_FUSED_LAUNCHES).

The decode split-K LoRA-A changes only the fp32 summation order of the stock
shrink. The fused shrink+expand kernels and the LoRA-A side-stream overlap of
the triton backend must reproduce the split-K chain they replace bit for bit,
eager and under a CUDA graph, including while the main stream allocates and
writes next to an overlapped LoRA-A.
"""

import pytest
import torch

from sglang.kernels.ops.gemm import lora_decode_fused, sgemm_lora_a
from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_CUDA = torch.cuda.is_available()
I32 = dict(dtype=torch.int32, device="cuda")


def _decode_batch_info(bs: int, rank: int) -> LoRABatchInfo:
    """One single-token segment per request, all on adapter 0."""
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=bs,
        num_segments=bs,
        seg_indptr=torch.arange(bs + 1, **I32),
        weight_indices=torch.zeros(bs, **I32),
        lora_ranks=torch.tensor([rank], **I32),
        scalings=torch.ones(1, device="cuda"),
        max_len=1,
        seg_lens=torch.ones(bs, **I32),
        permutation=None,
    )


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("k", [768, 2048, 7168, 33792])
@pytest.mark.parametrize("stack_num", [1, 2, 3])
@pytest.mark.parametrize("bs", [1, 8, 16])
def test_splitk_lora_a_matches_stock(monkeypatch, k: int, stack_num: int, bs: int):
    """Split-K only reorders the fp32 sum over K: within 2 bf16 ulp of the
    largest output, run-to-run identical, and unused below its minimum K."""
    rank = 16
    g = torch.Generator(device="cuda").manual_seed(k + 10 * stack_num + bs)
    x = torch.randn(bs, k, generator=g, device="cuda").bfloat16()
    w = (
        torch.randn(1, stack_num * rank, k, generator=g, device="cuda") * 0.02
    ).bfloat16()
    info = _decode_batch_info(bs, rank)
    monkeypatch.setattr(sgemm_lora_a, "_SPLITK_ENABLED", False)
    ref = sgemm_lora_a.sgemm_lora_a_fwd(x, w, info, stack_num=stack_num)
    monkeypatch.setattr(sgemm_lora_a, "_SPLITK_ENABLED", True)
    got = sgemm_lora_a.sgemm_lora_a_fwd(x, w, info, stack_num=stack_num)
    again = sgemm_lora_a.sgemm_lora_a_fwd(x, w, info, stack_num=stack_num)
    assert torch.equal(got, again)
    if not sgemm_lora_a.splitk_applies(w, info):
        assert k < 2048 and torch.equal(ref, got)
        return
    ulp = 2.0 ** (torch.log2(ref.float().abs().max()).floor().item() - 7)
    max_diff = (ref.float() - got.float()).abs().max().item()
    print(
        f"k={k} stack={stack_num} bs={bs}: bit-equal "
        f"{(ref == got).float().mean().item():.4f}, max diff {max_diff / ulp:.2f} ulp"
    )
    assert max_diff <= 2 * ulp


def _segments_info(m: int, split: int | None, seed: int) -> LoRABatchInfo:
    """Decode batch info for m tokens: one adapter (rank 16), or two segments
    (adapter 0 at rank 16, adapter 1 at rank 8) split after ``split`` tokens."""
    perm = torch.randperm(m, generator=torch.Generator().manual_seed(seed))
    if split is None:
        seg_lens, weight_indices = [m], [0]
    else:
        seg_lens, weight_indices = [split, m - split], [0, 1]
    seg_indptr = [0]
    for n in seg_lens:
        seg_indptr.append(seg_indptr[-1] + n)
    return LoRABatchInfo(
        use_cuda_graph=True,
        bs=len(seg_lens),
        num_segments=len(seg_lens),
        seg_indptr=torch.tensor(seg_indptr, **I32),
        weight_indices=torch.tensor(weight_indices, **I32),
        lora_ranks=torch.tensor([16, 8], **I32),
        scalings=torch.tensor([2.0, 0.5], device="cuda"),
        max_len=max(seg_lens),
        seg_lens=torch.tensor(seg_lens, **I32),
        permutation=perm.to(torch.int32).cuda(),
    )


def _backend(monkeypatch, info: LoRABatchInfo, fused: bool) -> TritonLoRABackend:
    monkeypatch.setattr(lora_decode_fused, "DECODE_FUSED_LAUNCHES", fused)
    monkeypatch.setattr(TritonLoRABackend, "supports_lora_a_overlap", fused)
    backend = TritonLoRABackend(max_loras_per_batch=2, device=torch.device("cuda"))
    backend.get_batch_info = lambda layout=None: info
    backend.sgemm_batch_info = info
    return backend


def _tensors(m: int, seed: int) -> dict:
    """Decode-step operands of the LoRA'd modules (two adapters, max rank 16)."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g, device="cuda") * scale).to(dtype)

    return dict(
        # row-parallel o_proj: K 768, out 7168
        x_o=rnd(m, 768),
        a_o=rnd(2, 16, 768, scale=0.05),
        b_o=rnd(2, 7168, 16, scale=0.05),
        w_o=rnd(7168, 768, scale=0.03),
        # small-K down projection: K 384
        x_d=rnd(m, 384),
        a_d=rnd(2, 16, 384, scale=0.05),
        b_d=rnd(2, 7168, 16, scale=0.05),
        base_d=rnd(m, 7168),
        # gate|up (fp32 base output) and q_a|kv_a: K 7168, two slices
        x_g=rnd(m, 7168),
        a_g=rnd(2, 32, 7168, scale=0.02),
        b_g=rnd(2, 768, 16, scale=0.05),
        base_g=rnd(m, 768, dtype=torch.float32),
        offs_g=torch.tensor([0, 384, 768], **I32),
        a_q=rnd(2, 32, 7168, scale=0.02),
        b_q=rnd(2, 2112, 16, scale=0.05),
        base_q=rnd(m, 2112),
        w_q=rnd(2112, 7168, scale=0.01),
        offs_q=torch.tensor([0, 1536, 2112], **I32),
    )


def _run_modules(backend: TritonLoRABackend, t: dict, overlap: bool) -> dict:
    """One decode step's worth of LoRA'd modules through the backend entry
    points the LoRA layers call; ``overlap`` starts each LoRA-A on the side
    stream before its base GEMM, as the layers do when the backend supports it."""
    outs = {}
    if overlap:
        backend.start_lora_a_overlap(t["x_o"], t["a_o"], num_slices=1)
    base = torch.nn.functional.linear(t["x_o"], t["w_o"])
    lora_a = backend.run_lora_a_sgemm(t["x_o"], t["a_o"])
    outs["o_proj"] = backend.run_lora_b_sgemm(lora_a, t["b_o"], base_output=base)

    def down(base):
        out = backend.run_lora_ab_fused(t["x_d"], t["a_d"], t["b_d"], base)
        if out is None:
            lora_a = backend.run_lora_a_sgemm(t["x_d"], t["a_d"])
            out = backend.run_lora_b_sgemm(lora_a, t["b_d"], base_output=base)
        return out

    outs["down"] = down(t["base_d"].clone())
    # a pending overlapped LoRA-A makes the fused small-K path defer to it
    if overlap:
        backend.start_lora_a_overlap(t["x_d"], t["a_d"], num_slices=1)
    outs["down_pending"] = down(t["base_d"].clone())
    for start in (True, False):
        if overlap and start:
            backend.start_lora_a_overlap(t["x_g"], t["a_g"], num_slices=2)
        outs[f"gate_up_{start}"] = backend.run_gate_up_lora(
            t["x_g"],
            t["a_g"],
            t["b_g"],
            base_output=t["base_g"].clone(),
            output_offset=t["offs_g"],
        )
        if overlap and start:
            backend.start_lora_a_overlap(t["x_g"], t["a_q"], num_slices=2)
        outs[f"qkv_{start}"] = backend.run_qkv_lora(
            t["x_g"],
            t["a_q"],
            t["b_q"],
            t["offs_q"],
            1536,
            base_output=t["base_q"].clone(),
            n_slices=2,
        )
    # the base GEMM and its neighbours allocate and write on the main stream
    # while the overlapped LoRA-A (split-K partials included) is in flight
    if overlap:
        backend.start_lora_a_overlap(t["x_g"], t["a_q"], num_slices=2)
    scratch = [
        torch.full((t["x_g"].shape[0] * 4096 + 256 * i,), 7.0, device="cuda")
        for i in range(16)
    ]
    base = t["base_q"] + torch.nn.functional.linear(t["x_g"], t["w_q"])
    outs["qkv_busy"] = backend.run_qkv_lora(
        t["x_g"], t["a_q"], t["b_q"], t["offs_q"], 1536, base_output=base, n_slices=2
    )
    outs["scratch"] = torch.stack([s[:256] for s in scratch])
    assert backend._pending_lora_a is None
    return outs


def _graph_outputs(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()  # warmup creates the side streams outside the capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = fn()
    graph.replay()
    torch.cuda.synchronize()
    return outs


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize(
    "m,split", [(1, None), (4, None), (8, None), (16, None), (8, 3)]
)
@pytest.mark.parametrize("use_graph", [False, True])
def test_backend_paths_bit_equal(monkeypatch, m: int, split, use_graph: bool):
    monkeypatch.setattr(sgemm_lora_a, "_SPLITK_ENABLED", True)
    info = _segments_info(m, split, seed=m)
    t = _tensors(m, seed=m)

    stock = _backend(monkeypatch, info, fused=False)
    assert not stock.supports_lora_a_overlap
    assert stock.run_lora_ab_fused(t["x_d"], t["a_d"], t["b_d"], t["base_d"]) is None
    ref = _run_modules(stock, t, overlap=False)

    fused = _backend(monkeypatch, info, fused=True)
    assert fused.supports_lora_a_overlap
    if use_graph:
        got = _graph_outputs(lambda: _run_modules(fused, t, overlap=True))
    else:
        got = _run_modules(fused, t, overlap=True)
    torch.cuda.synchronize()
    mismatched = [k for k in ref if not torch.equal(ref[k], got[k])]
    assert not mismatched, mismatched


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
