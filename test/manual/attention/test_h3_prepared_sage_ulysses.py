# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU acceptance for NCCL prepared Sage QKV and CAKE pre/post fusion.

Run with FlashInfer containing #4924, #4951 and the SM120 tail/count fixes:
  torchrun --standalone --nproc_per_node=4 test/manual/attention/test_h3_prepared_sage_ulysses.py

This compares fused and unfused glue with the same quantized-operand router;
it does not establish quality equivalence to BF16 routing or full H3 inference.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist


@torch.inference_mode()
def main():
    import os

    from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse import (
        sage_ulysses as adapter,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse import (
        sage_ulysses_route as route,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
        SubBlockSparseSchedule,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world in (4, 8) and torch.cuda.get_device_capability() == (12, 0)
    schedule = SubBlockSparseSchedule(
        sparsity=0.75,
        skip_first_steps=10,
        skip_first_layers=0,
        n_k=4,
        n_q=4,
        min_seq_len=0,
        compute_mode="sage_fp8",
        quantize_before_a2a=True,
    )
    impl = SimpleNamespace(
        schedule=schedule,
        layer_enabled=True,
        _step_enabled=lambda: True,
        softmax_scale=128**-0.5,
    )
    try:
        for used in (130, 4097):
            total = (used + world - 1) // world * world
            torch.manual_seed(1234 + rank)
            packed_qkv = torch.randn(
                total // world, world * 2, 3, 128, device="cuda", dtype=torch.bfloat16
            )
            q, k, v = packed_qkv.unbind(dim=2)
            mask = torch.arange((used + 63) // 64, device="cuda") % 3 != 0
            transport = adapter._PreparedSageTransport(dist.group.WORLD, q, used)
            try:
                with (
                    torch.inference_mode(),
                    patch.object(
                        route,
                        "get_sp_group",
                        lambda: SimpleNamespace(ulysses_group=dist.group.WORLD),
                    ),
                    patch.object(
                        route, "get_ulysses_parallel_world_size", lambda: world
                    ),
                ):
                    kwargs = dict(
                        impl=impl,
                        cu_seqlens_host=(0, used, total),
                        max_seqlen=used,
                        ulysses_active=True,
                        ring_active=False,
                        query_block_mask=mask,
                        transport=transport,
                    )
                    with patch.dict(
                        os.environ, {"SGLANG_DIFFUSION_H3_SAGE_FUSED_PREPOST": "0"}
                    ):
                        reference = adapter.prepared_sage_ulysses_forward(
                            q, k, v, **kwargs
                        ).clone()
                    with patch.dict(
                        os.environ, {"SGLANG_DIFFUSION_H3_SAGE_FUSED_PREPOST": "1"}
                    ):
                        actual = adapter.prepared_sage_ulysses_forward(
                            q, k, v, **kwargs
                        )
                    assert actual is not None and actual.shape == q.shape
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    assert torch.isfinite(actual).all()
                    if rank == world - 1:
                        assert not actual[-(total - used) :].count_nonzero().item()
                dist.barrier()
                if rank == 0:
                    print(
                        f"PASS world={world} used={used}: fused/unfused output exact",
                        flush=True,
                    )
            finally:
                transport.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
