# SPDX-License-Identifier: Apache-2.0
"""Dense Sage prepared-QKV acceptance against live-only FP32 SDPA.

Run with public FlashInfer >=0.7.4 and SageAttention built for SM120:
  torchrun --standalone --nproc_per_node=4 test/manual/attention/test_h3_prepared_dense_sage_ulysses.py

Tests numerical attention error and distributed layout, not model quality.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.inference_mode()
def main():
    from sglang.multimodal_gen.runtime.distributed import (
        sage_ulysses,
        sage_ulysses_route,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.prepared_sage_attn import (
        prepared_dense_sage_attention_forward,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn import (
        SageAttentionImpl,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    assert world in (4, 8) and torch.cuda.get_device_capability() == (12, 0)
    heads, dim = world * 2, 128
    impl = SageAttentionImpl(
        num_heads=heads,
        head_size=dim,
        causal=False,
        softmax_scale=dim**-0.5,
        quantize_before_a2a=True,
        packed_trailing_padding=True,
    )
    try:
        with (
            patch.object(
                sage_ulysses_route,
                "get_sp_group",
                lambda: SimpleNamespace(ulysses_group=dist.group.WORLD),
            ),
            patch.object(
                sage_ulysses_route, "get_ulysses_parallel_world_size", lambda: world
            ),
        ):
            for used in (1, 64, 130, 4097, *range(131, 140), 130):
                total = (used + world - 1) // world * world
                local = total // world
                torch.manual_seed(2026 + rank)
                packed = torch.randn(
                    (local, heads, 3, dim), device="cuda", dtype=torch.bfloat16
                )
                q, k, v = packed.unbind(2)
                # Exercise FlashInfer's zero-amax V handling as well as tails.
                v[..., 0].zero_()
                full = torch.empty(
                    (total, heads, 3, dim), device="cuda", dtype=packed.dtype
                )
                dist.all_gather_into_tensor(full, packed)
                full_q, full_k, full_v = full.unbind(2)
                head_slice = slice(rank * 2, (rank + 1) * 2)
                reference = F.scaled_dot_product_attention(
                    full_q[:used, head_slice].unsqueeze(0).transpose(1, 2).float(),
                    full_k[:used, head_slice].unsqueeze(0).transpose(1, 2).float(),
                    full_v[:used, head_slice].unsqueeze(0).transpose(1, 2).float(),
                    scale=impl.softmax_scale,
                ).transpose(1, 2)
                reference_padded = torch.zeros((1, total, 2, dim), device="cuda")
                reference_padded[:, :used] = reference
                reference_heads = [
                    torch.empty_like(reference_padded) for _ in range(world)
                ]
                dist.all_gather(reference_heads, reference_padded)
                expected = torch.cat(reference_heads, dim=2)[
                    0, rank * local : (rank + 1) * local
                ]

                current = torch.cuda.current_stream()
                compute = torch.cuda.Stream() if used == 130 else current
                if compute != current:
                    compute.wait_stream(current)
                with torch.cuda.stream(compute):
                    actual = prepared_dense_sage_attention_forward(
                        q,
                        k,
                        v,
                        impl=impl,
                        cu_seqlens_host=(0, used, total),
                        max_seqlen=used,
                        ulysses_active=True,
                        ring_active=False,
                    )
                if compute != current:
                    current.wait_stream(compute)
                assert actual is not None and actual.shape == q.shape
                assert torch.isfinite(actual).all()
                live = max(0, min(local, used - rank * local))
                assert not actual[live:].count_nonzero().item()
                if live:
                    a, e = actual[:live].float().flatten(), expected[:live].flatten()
                    cosine = F.cosine_similarity(a, e, dim=0).item()
                    relative_rms = (a - e).norm().item() / max(e.norm().item(), 1e-12)
                    assert cosine >= 0.998 and relative_rms <= 0.06, (
                        rank,
                        used,
                        cosine,
                        relative_rms,
                    )
                assert len(sage_ulysses._TRANSPORTS) == 1
                dist.barrier()
                if rank == 0:
                    print(
                        f"PASS dense Sage world={world} used={used}: live-only SDPA tolerance and zero padding",
                        flush=True,
                    )
    finally:
        sage_ulysses.shutdown_prepared_sage_ulysses()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
