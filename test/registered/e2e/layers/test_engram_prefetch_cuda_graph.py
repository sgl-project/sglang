"""Real TP4/DP4 Engram collectives and changing decode/VERIFY graph inputs.

No model checkpoint or server is needed. Tiny FP8 tables exercise owned-row
lookup and reduce-scatter; BF16 and CuTe-DSL MXFP8 wkv exercise both supported
projection routes. Intervening residual updates stand in for preceding layers.
"""

import unittest
from types import SimpleNamespace as NS

import torch
from torch import nn

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
from sglang.srt.layers.engram import (
    Engram,
    EngramHasher,
    EngramLayout,
    EngramPrefetcher,
    compute_hash_multipliers,
)
from sglang.srt.layers.moe.mega_moe import _wait_for_moe_prefetch
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import (
    get_forward,
    get_parallel,
    get_stream,
    reset_context,
)
from sglang.srt.utils.network import try_bind_socket
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cuda_ci(est_time=120, stage="base-c", runner_config="4-gpu-b200")


def _build_layers(quantized):
    layout = EngramLayout(
        max_ngram_size=4,
        layer_ids=(1, 3),
        num_embeddings=(211, 251),
        primes=(((67,), (71,), (73,)), ((79,), (83,), (89,))),
        n_heads=1,
        head_dim=128,
    )
    config = NS(hidden_size=128, hc_mult=2, rms_norm_eps=1e-6)
    quant = (
        Fp8Config(is_checkpoint_fp8_serialized=True, use_mxfp8=True)
        if quantized
        else None
    )
    layers = []
    for i, layer_id in enumerate(layout.layer_ids):
        torch.manual_seed(1337 + layer_id)
        with torch.device("cuda"):
            layer = Engram(config, layer_id, layout, quant_config=quant)
        full = torch.randn(layout.num_embeddings[i], 128, device="cuda") * 0.25
        lo, hi = layer.embed.row_start, layer.embed.row_start + layer.embed.rows
        layer.embed.weight.data.copy_(full[lo:hi].to(layer.embed.weight.dtype))
        layer.embed.scale.data.copy_(
            torch.ones_like(layer.embed.scale, dtype=torch.bfloat16).to(
                layer.embed.scale.dtype
            )
        )
        wkv = layer.wkv
        weight = (
            torch.randn(wkv.weight.shape, device="cuda", dtype=torch.bfloat16) * 0.03125
        )
        if quantized:
            wkv.weight.data.copy_(weight.to(wkv.weight.dtype))
            # Serialized MXFP8 scales are UE8M0 bytes, exponent 127 means 1.
            wkv.weight_scale_inv.data.fill_(127)
            wkv.quant_method.process_weights_after_loading(wkv)
        else:
            wkv.weight.data = weight
        assert layer.supports_prefetch()
        layers.append((layer_id, layer))

    # Avoid tokenizer I/O; the real GPU hasher receives an explicit token map.
    hasher = EngramHasher.__new__(EngramHasher)
    nn.Module.__init__(hasher)
    hasher.max_ngram_size, hasher.pad_id, hasher.image_token_id = 4, 0, None
    hasher.register_buffer("token_map", torch.arange(64, device="cuda"))
    hasher.register_buffer(
        "multipliers", compute_hash_multipliers(layout.layer_ids, 4, 64).cuda()
    )
    hasher.register_buffer("primes", torch.tensor(layout.primes, device="cuda"))
    hasher.register_buffer(
        "offsets", torch.tensor([[0, 67, 138], [0, 79, 162]], device="cuda")
    )
    hasher.init_history(8, "cuda")
    return hasher, layers


def _check_graph(rank, quantized, mode):
    hasher, layers = _build_layers(quantized)
    prefetch = EngramPrefetcher(get_stream("dsv41_engram_prefetch"), layers)
    block = 3 if mode.is_target_verify() else 1
    bs, world = 4, get_parallel().tp_size
    rows = bs * block
    batch = NS(
        forward_mode=mode,
        global_forward_mode=mode,
        positions=torch.arange(rows, device="cuda") + 16,
        req_pool_indices=torch.arange(bs, device="cuda", dtype=torch.int64),
        out_cache_loc=torch.ones(rows, device="cuda", dtype=torch.int64),
        dp_padding_mode=DpPaddingMode.MAX_LEN,
        spec_info=NS(draft_token_num=block),
    )
    set_dp_buffer_len(rows * world, rows, True, [rows] * world)
    ids = torch.ones(rows, device="cuda", dtype=torch.int64)
    hidden = torch.zeros(rows, 2, 128, device="cuda", dtype=torch.bfloat16)
    initial_history = torch.full_like(hasher.history, 2)
    initial_history[hasher.pad_row].zero_()

    def forward(prefetched):
        hashes = hasher(ids, batch)
        prepared = {} if prefetched else None
        by_id = dict(layers)
        with get_forward().scoped(
            moe_prefetch_fence=prefetch.moe_fence() if prefetched else None
        ):
            x = hidden
            for layer_id in range(layers[-1][0] + 1):
                layer = by_id.get(layer_id)
                if layer is not None:
                    if prefetched:
                        x = prefetch.consume(layer_id, layer, x, prepared)
                    else:
                        x = layer(x, hashes[:, layer.layer_hash_index], batch)
                if prefetched and layer_id + 1 in by_id:
                    prepared.update(
                        prefetch.start(
                            hashes,
                            batch,
                            layer_ids=(layer_id + 1,),
                            fence=get_forward().moe_prefetch_fence,
                        )
                    )
                # Emulate preceding attention before the real alternate-stream
                # fence. There are two independently armed windows per forward.
                x = x * 0.5 + 0.125
                # Exercise the production fence inside capture and replay,
                # including a MegaMoE alternate launch stream.
                with torch.cuda.stream(get_stream("engram_test_moe")):
                    _wait_for_moe_prefetch()
                torch.cuda.current_stream().wait_stream(get_stream("engram_test_moe"))
            if prefetched:
                assert not prepared
        return x

    for _ in range(3):
        hasher.history.copy_(initial_history)
        forward(True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    get_stream("engram_test_moe")  # Allocate the test stream before capture.
    capture_stream = torch.cuda.Stream()
    with get_parallel().tp_group.graph_capture(stream=capture_stream):
        with torch.cuda.graph(graph, stream=capture_stream):
            captured_output = forward(True)

    # Replay under new token IDs, residuals, padding and histories. Rank 3 has
    # no real requests on the first replay but still joins every collective.
    for step in range(4):
        real = (rank + step + 2) % (bs + 1)
        slots = torch.arange(bs, device="cuda", dtype=torch.int64)
        slots[real:] = hasher.pad_row
        batch.req_pool_indices.copy_(slots)
        loc = torch.arange(1, rows + 1, device="cuda", dtype=torch.int64)
        loc[real * block :] = 0
        batch.out_cache_loc.copy_(loc)
        batch.positions.add_(1)
        ids.copy_((torch.arange(rows, device="cuda") + 3 * rank + 7 * step) % 63 + 1)
        hidden.fill_(0.125 * (rank + step + 1))
        initial_history[:8].fill_(step + 2)
        hasher.history.copy_(initial_history)
        expected = forward(False)
        expected_history = hasher.history.clone()
        hasher.history.copy_(initial_history)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured_output, expected, rtol=0, atol=0)
        torch.testing.assert_close(hasher.history, expected_history, rtol=0, atol=0)
        if mode.is_target_verify():
            # VERIFY does not commit until acceptance is known; padding must
            # never update the spare history row.
            torch.testing.assert_close(hasher.history, initial_history, rtol=0, atol=0)
            counts = torch.zeros(bs, device="cuda", dtype=torch.int32)
            counts[:real] = 1 + step % block
            hasher.commit_after_verify(
                ids.view(bs, block), batch.req_pool_indices, counts
            )
            torch.testing.assert_close(
                hasher.history[hasher.pad_row],
                initial_history[hasher.pad_row],
                rtol=0,
                atol=0,
            )
    torch.cuda.synchronize()


def _worker(rank, port):
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=4,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
        timeout=120,
    )
    publish_build_topology(
        world_rank=rank,
        tp_size=4,
        attn_dp_size=4,
        fp8_gemm_runner_backend="flashinfer_cutedsl",
    )
    try:
        initialize_model_parallel()
        with (
            torch.no_grad(),
            envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(False),
        ):
            for quantized in (False, True):
                for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
                    _check_graph(rank, quantized, mode)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
        reset_context()


class TestEngramPrefetchCudaGraph(CustomTestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 4,
        "requires four CUDA GPUs",
    )
    def test_dp4_eager_and_graph_replay(self):
        # Restrict the rendezvous and port probe to loopback.
        with try_bind_socket(host="127.0.0.1") as sock:
            port = sock.getsockname()[1]
        torch.multiprocessing.spawn(_worker, args=(port,), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main()
