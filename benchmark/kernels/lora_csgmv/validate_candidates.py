"""Checks for the benchmark serving wrappers, including graph padding and mixed ranks."""

import itertools
import math
import random

import torch
from serving_candidates import install
from sglang.kernels.ops.gemm.chunked_sgmv_expand import chunked_sgmv_lora_expand_forward
from sglang.kernels.ops.gemm.chunked_sgmv_shrink import chunked_sgmv_lora_shrink_forward
from sglang.srt.lora.utils import LoRABatchInfo


def validate():
    namespace = dict(
        chunked_sgmv_lora_shrink_forward=chunked_sgmv_lora_shrink_forward,
        chunked_sgmv_lora_expand_forward=chunked_sgmv_lora_expand_forward,
    )
    install(namespace)
    checks = []
    torch.manual_seed(771)
    for total, mixed in [(32, False), (33, True), (128, True), (2048, False)]:
        adapters = 8
        ids = [i % adapters for i in range(total)]
        random.Random(9).shuffle(ids)
        counts = [ids.count(i) for i in range(adapters)]
        lengths = [min(16, n - j) for n in counts for j in range(0, n, 16)]
        wi = [a for a, n in enumerate(counts) for _ in range(0, n, 16)]
        segments = len(wi)

        def t(v):
            return torch.tensor(v, device="cuda", dtype=torch.int32)

        ranks = [0, 8, 16, 32, 0, 8, 16, 32] if mixed else [32] * 8
        info = LoRABatchInfo(
            use_cuda_graph=True,
            bs=total,
            num_segments=segments,
            max_len=16,
            seg_lens=None,
            permutation=t(sorted(range(total), key=ids.__getitem__)),
            weight_indices=t(wi + [0] * 4),
            seg_indptr=t([0, *itertools.accumulate(lengths)] + [total] * 4),
            lora_ranks=t(ranks),
            scalings=torch.tensor(
                [0.3, 0.7, 1, 1.5, -0.2, 0.4, 1.1, 0.9], device="cuda"
            ),
        )
        for name, k, offsets in [
            ("qkv", 4096, [0, 8192, 9216, 10240]),
            ("gate_up", 4096, [0, 12288, 24576]),
            ("down", 12288, [0, 4096]),
        ]:
            slices = len(offsets) - 1
            n = offsets[-1]
            x = torch.randn(total, k, device="cuda", dtype=torch.bfloat16)
            a = torch.randn(
                8, slices * 32, k, device="cuda", dtype=torch.bfloat16
            ) / math.sqrt(k)
            b = torch.randn(8, n, 32, device="cuda", dtype=torch.bfloat16) / math.sqrt(
                32
            )
            # Wider, strided base output is supported by the expand API.
            original_base = torch.randn(
                total, n + 17, device="cuda", dtype=torch.bfloat16
            )
            base0 = original_base.clone()
            base1 = original_base.clone()
            h0 = chunked_sgmv_lora_shrink_forward(x, a, info, slices)
            h1 = namespace["chunked_sgmv_lora_shrink_forward"](x, a, info, slices)
            for adapter, rank in enumerate(ranks):
                if rank:
                    rows = [i for i, v in enumerate(ids) if v == adapter]
                    torch.testing.assert_close(
                        h1[rows, : slices * rank],
                        h0[rows, : slices * rank],
                        atol=0.04,
                        rtol=0.025,
                    )
            offs = t(offsets)
            maxwidth = max(hi - lo for lo, hi in zip(offsets, offsets[1:]))
            chunked_sgmv_lora_expand_forward(h0, b, info, offs, maxwidth, base0[:, :n])
            namespace["chunked_sgmv_lora_expand_forward"](
                h1, b, info, offs, maxwidth, base1[:, :n]
            )
            torch.testing.assert_close(base1, base0, atol=0.08, rtol=0.03)
            assert torch.equal(base1[:, n:], original_base[:, n:])
            for adapter, rank in enumerate(ranks):
                if rank == 0:
                    rows = [i for i, v in enumerate(ids) if v == adapter]
                    assert torch.equal(base1[rows], original_base[rows])
            base1.copy_(original_base)
            live_segments = info.num_segments
            live_indptr = info.seg_indptr.clone()
            live_weights = info.weight_indices.clone()
            live_ranks = info.lora_ranks.clone()
            capture_segments = math.ceil(total / 16)
            info.num_segments = capture_segments
            info.seg_indptr.fill_(total)
            info.seg_indptr[: capture_segments + 1].copy_(
                torch.arange(capture_segments + 1, device="cuda")
                .mul(16)
                .clamp(max=total)
            )
            info.weight_indices.zero_()
            info.lora_ranks.zero_()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                hg = namespace["chunked_sgmv_lora_shrink_forward"](x, a, info, slices)
                namespace["chunked_sgmv_lora_expand_forward"](
                    hg, b, info, offs, maxwidth, base1[:, :n]
                )
            info.num_segments = live_segments
            info.seg_indptr.copy_(live_indptr)
            info.weight_indices.copy_(live_weights)
            info.lora_ranks.copy_(live_ranks)
            base1.copy_(original_base)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(base1, base0, atol=0.08, rtol=0.03)
            checks.append(
                dict(
                    tokens=total,
                    mixed_ranks=mixed,
                    layer=name,
                    max_abs=(base1.float() - base0.float()).abs().max().item(),
                    graph_replay_passed=True,
                )
            )
    return checks
