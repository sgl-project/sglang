"""Run with python on eight GPUs; validate peer exchange and graph replay."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sglang.srt.distributed.parallel_state as ps
import test_ar_fusion as fixture
import torch
import torch.distributed as dist
from sglang.kernels.ops.attention.dcp_kernels import (
    DcpDirectAllToAll,
    dcp_lse_combine_triton,
)
from sglang.srt.layers.dcp import comm
from sglang.srt.runtime_context import get_parallel, get_platform
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import published_topology

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=90, stage="extra-b", runner_config="8-gpu-h200")


def inputs(batch, rank, dim=512):
    generator = torch.Generator(device="cuda").manual_seed(42 + rank)
    output = torch.randn(
        batch, 128, dim, dtype=torch.bfloat16, device="cuda", generator=generator
    )[:, :96]
    lse = torch.randn(128, batch, device="cuda", generator=generator).t()[:, :96]
    lse[0] = -float("inf")
    return output, lse


@pytest.mark.parametrize("dim", [128, 512])
def test_peer_exchange_bits_and_replay(dim):
    fixture._init_world()
    group = ps._WORLD
    if group.world_size != 8:
        pytest.skip("direct DCP exchange requires eight ranks")
    rank = group.rank_in_group
    exchange = DcpDirectAllToAll(group)
    # The transport also runs on H200; its production tuning gate remains B300.
    exchange.enabled = True
    small, small_lse = inputs(1, rank, dim)
    assert exchange(small, small_lse) is None
    assert dim in exchange.buffers  # eager small-batch warmup reserves storage
    output, lse = inputs(512, rank, dim)
    for _ in range(2):
        exchange(output, lse)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        partials, stats = exchange(output, lse)
        combined, _ = dcp_lse_combine_triton(partials, stats)
    for phase in range(4):
        output.copy_(inputs(512, rank + 13 * phase, dim)[0])
        lse[1, rank] = float("nan") if phase == 1 else -float("inf")
        output.view(torch.int16)[0, rank, :4] = torch.tensor(
            [0, -32768, 32640, 32705], device="cuda", dtype=torch.int16
        )
        all_o = [
            torch.empty(output.shape, device="cuda", dtype=output.dtype)
            for _ in range(8)
        ]
        all_lse = [
            torch.empty(lse.shape, device="cuda", dtype=lse.dtype) for _ in range(8)
        ]
        dist.all_gather(all_o, output.contiguous(), group=group.device_group)
        dist.all_gather(all_lse, lse.contiguous(), group=group.device_group)
        reference_o = torch.stack(all_o)[:, :, rank * 12 : (rank + 1) * 12]
        reference_lse = torch.stack(all_lse)[:, :, rank * 12 : (rank + 1) * 12]
        expected, _ = dcp_lse_combine_triton(reference_o, reference_lse)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(partials.view(torch.int16), reference_o.view(torch.int16))
        assert torch.equal(stats.view(torch.int32), reference_lse.view(torch.int32))
        torch.testing.assert_close(combined, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dim", [128, 512])
def test_flashinfer_dispatch_and_fallback(dim):
    fixture._init_world()
    group = ps._WORLD
    if group.world_size != 8:
        pytest.skip("direct DCP exchange requires eight ranks")
    if torch.cuda.get_device_name(group.device) != "NVIDIA B300 SXM6 AC":
        pytest.skip("Production dispatch is tuned for B300")
    get_parallel().override_permanently(tp_size=8, pp_size=1, nnodes=1)
    comm.init_fi_a2a_workspace(group)
    direct = comm._FI_A2A_STATE["direct"]
    dispatched = []

    def recorded(output, lse):
        result = direct(output, lse)
        dispatched.append(result is not None)
        return result

    comm._FI_A2A_STATE["direct"] = recorded
    try:
        cases = [(batch, 0, 0) for batch in (1, 511, 512, 513, 768, 1024, 1025, 512)]
        for batch, offset, padding in cases + [(512, 2, 0), (512, 0, 2)]:
            output, lse = inputs(batch, group.rank_in_group, dim)
            # Different pointer alignments must not split ranks between transports.
            offset *= group.rank_in_group % 2
            padding *= group.rank_in_group % 2
            if offset or padding:
                storage = output.new_empty(batch * 96 * (dim + padding) + offset)
                output = (
                    storage[offset:]
                    .view(batch, 96, dim + padding)[..., :dim]
                    .copy_(output)
                )

            def call():
                return comm.dcp_a2a_lse_reduce(
                    output, lse, group, comm_backend="fi_a2a"
                )

            for _ in range(2):
                call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = call()
            assert dispatched[-1] == (512 <= batch <= 1024)
            for phase in range(3):
                output.copy_(inputs(batch, group.rank_in_group + phase + 1, dim)[0])
                graph.replay()
                direct.enabled = False
                expected = call()
                direct.enabled = True
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    result, expected, rtol=0, atol=0, equal_nan=True
                )
    finally:
        direct.enabled = True
        comm._FI_A2A_STATE["direct"] = direct


def test_request_ordered_byte_gather():
    from sglang.srt.layers.attention.dsa.dsa_dcp_sharded import dcp_gather_request_rows

    fixture._init_world()
    group = ps._WORLD
    rank, world = group.rank_in_group, group.world_size
    # Empty local shares, request boundaries and both word widths above 64 MiB.
    cases = [
        ([1, 7, 9], 132),
        ([65, 63, 129], 656),
        ([65, 63, 129], 133),
        ([524288], 132),
        ([131072], 656),
        ([131073], 512),
    ]
    with (
        published_topology(tp_size=world, dcp_size=world),
        get_parallel().override(
            dcp_group=group,
            dcp_rank=rank,
            attn_dcp_rank=rank,
        ),
    ):
        for lengths, width in cases:
            gen = torch.Generator(device="cuda").manual_seed(42)
            full = torch.randint(
                0,
                256,
                (sum(lengths), width),
                dtype=torch.uint8,
                device="cuda",
                generator=gen,
            )
            local = torch.cat([part[rank::world] for part in full.split(lengths)])
            # Each rank starts at a different byte offset and has padded rows.
            storage = local.new_empty(local.shape[0] * (width + 16) + rank + 1)
            rows = storage[rank + 1 :].view(-1, width + 16)[:, :width].copy_(local)
            actual = dcp_gather_request_rows(rows, torch.tensor(lengths))
            assert actual.dtype == full.dtype and actual.is_contiguous()
            assert torch.equal(actual, full)


def test_candidate_merge_scratch_and_graph_replay():
    from functools import partial

    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
    from sglang.srt.layers.attention.dsa import dsa_dcp_sharded as dsa_dcp

    fixture._init_world()
    group = ps._WORLD
    world, rank = group.world_size, group.rank_in_group
    workspace = dsa_dcp.DcpCandidateAllGather(group, max_rows=12, topk=2048)
    graphs = []
    with (
        published_topology(tp_size=world, dcp_size=world),
        get_parallel().override(dcp_group=group, dcp_rank=rank, attn_dcp_rank=rank),
        patch.object(dsa_dcp, "_CANDIDATE_ALL_GATHER", workspace),
    ):
        for rows in (8, 4, 12, 1):
            scores = torch.rand(rows, 4096, device="cuda")
            lengths = torch.full((rows,), 8193, device="cuda", dtype=torch.int32)
            local = (lengths + world - 1 - rank) // world
            table = torch.arange(64, device="cuda", dtype=torch.int32).expand(rows, -1)

            def call(scores, local, lengths, table, **kwargs):
                return dsa_dcp.dcp_global_topk(
                    scores,
                    local,
                    lengths,
                    table,
                    2048,
                    64,
                    plan_topk_v2(lengths),
                    **kwargs,
                )

            for _ in range(2):
                call(scores, local, lengths, table)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = call(scores, local, lengths, table)
            graphs.append((graph, actual, scores, local, lengths, table))

        scratch_ptr = workspace.merge_buffer.data_ptr()
        for length in (513, 16383, 4097):
            for graph, actual, scores, local, lengths, table in reversed(graphs):
                scores.uniform_()
                lengths.fill_(length)
                local.copy_((lengths + world - 1 - rank) // world)
                expected = call(
                    scores,
                    local,
                    lengths,
                    table,
                    all_gather=partial(group.all_gather, dim=0),
                )
                graph.replay()
                torch.cuda.synchronize()
                # The top-k contract specifies the selected set, not output order.
                assert torch.equal(
                    actual.sort(dim=-1).values, expected.sort(dim=-1).values
                ), (actual.shape[0], length)
                assert workspace.merge_buffer.data_ptr() == scratch_ptr


def test_token_packing_after_dcp_gather():
    fixture._init_world()
    group = ps._WORLD
    rank, world_size = group.rank_in_group, group.world_size
    if not get_platform().is_sm100:
        pytest.skip("token packing requires SM100")
    from sglang.kernels.ops.attention.dsa.dequant_k_cache import dequantize_k_cache
    from sglang.kernels.ops.attention.dsa.flashmla_token_pack import (
        flashmla_kv_token_packed,
        flashmla_token_pack_available,
    )
    from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

    if not flashmla_token_pack_available():
        pytest.skip("installed FlashMLA predates token packing")
    torch.backends.cuda.matmul.allow_tf32 = False
    with (
        published_topology(tp_size=world_size, dcp_size=world_size),
        get_parallel().override(
            dcp_group=group,
            dcp_rank=rank,
            attn_dcp_rank=rank,
        ),
    ):
        # Unequal rank shares, ranks with no rows, request boundaries
        # inside a five-query group, non-contiguous physical pages,
        # incomplete pages, and both short and full 2048-key support.
        for case, lengths in enumerate(
            ([1], [1, 7, 9], [65, 63, 129], [513, 2051, 17])
        ):
            torch.manual_seed(100 + case)
            total = sum(lengths)
            full = (
                quantize_k_cache(
                    torch.randn(
                        (total + 63) // 64,
                        64,
                        1,
                        576,
                        device=f"cuda:{rank}",
                        dtype=torch.bfloat16,
                    ),
                    pow2_scale=True,
                )
                .view(torch.uint8)
                .view(-1, 656)
            )
            local_pool = torch.full(
                (8192, 656), 91, device=f"cuda:{rank}", dtype=torch.uint8
            )
            local_rows = []
            offset = 0
            for request, length in enumerate(lengths):
                positions = torch.arange(length, device=f"cuda:{rank}")[
                    rank::world_size
                ]
                physical = 64 + request * 2048 + positions // world_size
                local_pool[physical] = full[offset + positions]
                local_rows.append(physical)
                offset += length
            batch = SimpleNamespace(seq_lens_cpu=torch.tensor(lengths))
            metadata = SimpleNamespace(dcp_local_kv_rows=torch.cat(local_rows))
            backend = SimpleNamespace(kv_cache_dim=656, physical_page_size=64)
            gathered = (
                DeepseekSparseAttnBackend.dcp_gather_kv_cache(
                    backend,
                    local_pool.view(torch.float8_e4m3fn),
                    batch,
                    metadata,
                )
                .view(torch.uint8)
                .view(-1, 656)
            )
            assert torch.equal(gathered[:total], full[:total])
            assert torch.count_nonzero(gathered[total:]).item() == 0

            extend = [min(length, 6 + i % 3) for i, length in enumerate(lengths)]
            tokens = sum(extend)
            torch.manual_seed(200 + rank + case)
            q = torch.randn(
                tokens, 12, 576, device=f"cuda:{rank}", dtype=torch.bfloat16
            )
            rows = torch.full(
                (tokens, 2048), -1, device=f"cuda:{rank}", dtype=torch.int32
            )
            offset = row = 0
            for length, count in zip(lengths, extend):
                for query in range(count):
                    visible = length - count + query + 1
                    selected = torch.randperm(visible, device=f"cuda:{rank}")[:2048]
                    rows[row, : selected.numel()] = selected.int() + offset
                    row += 1
                offset += length
            cache = gathered.view(-1, 64, 1, 656)
            actual = flashmla_kv_token_packed(
                q[:, :, :512], q[:, :, 512:], cache, rows, 192**-0.5
            )
            reference = torch.ops.sgl_kernel.sparse_decode_fwd.default(
                q.unsqueeze(1),
                cache,
                rows.unsqueeze(1),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                512,
                192**-0.5,
                "V32",
                0,
            )[0]
            # Packing reorders KV rows, so compare both BF16 kernels
            # with a complete FP32 reference before checking their pairwise
            # error. Bound global FP32 error as well as pairwise error.
            keys = (
                dequantize_k_cache(cache.view(torch.float8_e4m3fn))
                .reshape(-1, 576)
                .float()
            )
            gold = torch.empty_like(reference, dtype=torch.float32)
            for query in range(tokens):
                selected = rows[query][rows[query] >= 0].long()
                visible_keys = keys[selected]
                scores = q[query].float() @ visible_keys.T * 192**-0.5
                gold[query, 0] = scores.softmax(-1) @ visible_keys[:, :512]
            native_rmse = (reference.float() - gold).square().mean().sqrt()
            packed_rmse = (actual.float() - gold).square().mean().sqrt()
            relative_l2 = torch.linalg.vector_norm(actual.float() - gold) / (
                torch.linalg.vector_norm(gold)
            )
            assert torch.isfinite(gold).all()
            assert relative_l2 < 2.5e-3
            assert packed_rmse <= 1.2 * native_rmse + 3e-6
            torch.testing.assert_close(actual, reference, atol=5e-3, rtol=1e-2)
            torch.cuda.synchronize()


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4, 8))
