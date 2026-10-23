"""Two-rank host prefetch preserves TP, DP and CP collective participation."""

import socket
import traceback
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
from sglang.srt.layers.engram_prefetch import EngramPrefetcher
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.engram_utils import make_engram_layers
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cuda_ci(est_time=60, stage="base-b", runner_config="2-gpu-large")


def _worker(rank, port):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(False)
    publish_build_topology(world_rank=rank, tp_size=2, disable_custom_all_reduce=True)
    init_distributed_environment(
        world_size=2,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        timeout=90,
    )
    try:
        for topology in ("tp", "dp", "cp"):
            publish_build_topology(
                world_rank=rank,
                tp_size=2,
                attn_dp_size=2 if topology == "dp" else 1,
                attn_cp_size=2 if topology == "cp" else 1,
                disable_custom_all_reduce=True,
            )
            initialize_model_parallel()
            for layout in ("shared", "per_rank"):
                torch.manual_seed(81)
                layers = make_engram_layers(layout)
                prefetch = EngramPrefetcher(layers, 128, 1)
                scenarios = [(259, 259)]
                if topology == "dp":
                    scenarios += [(259, 131), (259, 0)]
                for counts in scenarios:
                    batch = None
                    if topology == "dp":
                        padding = (
                            DpPaddingMode.MAX_LEN
                            if counts[0] == counts[1]
                            else DpPaddingMode.SUM_LEN
                        )
                        batch = SimpleNamespace(
                            dp_padding_mode=padding,
                            dp_local_start_pos=None,
                            dp_local_num_tokens=None,
                            global_num_tokens_gpu=torch.tensor(counts, device="cuda"),
                            global_num_tokens_cpu=list(counts),
                        )
                        set_dp_buffer_len(
                            sum(counts),
                            counts[rank],
                            padding.is_max_len(),
                            list(counts),
                            batch.global_num_tokens_gpu,
                        )
                    tokens = counts[rank]
                    torch.manual_seed(91 + (rank if topology != "tp" else 0))
                    hashes = torch.randint(0, 1024, (tokens, 3, 4), device="cuda")
                    source = torch.randn(
                        tokens, 2, 64, dtype=torch.bfloat16, device="cuda"
                    )
                    expected = source
                    for layer in layers:
                        expected = layer(
                            expected,
                            hashes[:, layer.layer_hash_index],
                            batch,
                            cp_all_tokens=topology == "cp",
                        )
                    prefetch.begin(hashes, batch, cp_all_tokens=topology == "cp")
                    actual = source
                    for layer in layers:
                        actual = layer(
                            actual,
                            hashes[:, layer.layer_hash_index],
                            batch,
                            cp_all_tokens=topology == "cp",
                            prefetch=prefetch,
                        )
                    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
                    if counts == (259, 259) and topology in ("tp", "dp"):

                        def run():
                            prefetch.begin(hashes, batch)
                            value = source
                            for layer in layers:
                                value = layer(
                                    value,
                                    hashes[:, layer.layer_hash_index],
                                    batch,
                                    prefetch=prefetch,
                                )
                            return value

                        for _ in range(2):
                            run()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with graph_capture() as context:
                            with torch.cuda.graph(graph, stream=context.stream):
                                graph_output = run()
                        hashes.random_(0, 1024)
                        graph.replay()
                        reference = source
                        for layer in layers:
                            reference = layer(
                                reference, hashes[:, layer.layer_hash_index], batch
                            )
                        torch.testing.assert_close(
                            graph_output, reference, atol=1e-2, rtol=1e-2
                        )
                    if (
                        topology == "dp"
                        and counts == (259, 131)
                        and layout == "per_rank"
                    ):
                        from sglang.srt.layers import dp_attention

                        with patch.object(dp_attention, "_USE_DP_GATHERV", True):
                            reference = source
                            for layer in layers:
                                reference = layer(
                                    reference, hashes[:, layer.layer_hash_index], batch
                                )
                            prefetch.begin(hashes, batch)
                            value = source
                            for layer in layers:
                                value = layer(
                                    value,
                                    hashes[:, layer.layer_hash_index],
                                    batch,
                                    prefetch=prefetch,
                                )
                            torch.testing.assert_close(
                                value, reference, atol=1e-2, rtol=1e-2
                            )
                    torch.distributed.barrier()
            destroy_model_parallel()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        destroy_distributed_environment()
        reset_context()


class TestEngramPrefetchDistributed(CustomTestCase):
    def test_two_rank_layouts(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        mp.spawn(_worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
