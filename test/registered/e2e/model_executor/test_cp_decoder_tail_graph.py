"""Real CP collectives and decoder-tail CUDA graphs versus eager per bucket.

Synthetic layers isolate transport/count ownership; full-model accuracy is an
independent server test. Recording and collectives use the production runners.
"""

import socket
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.layers.cp.interleave import InterleaveContextParallelMetadata
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_forward_context,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.deepseek_common.moe_transport import MoeRowTransport
from sglang.srt.models.deepseek_v4_mhc import HcState
from sglang.srt.models.deepseek_v4_replay_tail import (
    DecoderReplayTail,
    _replay_tail_scope,
    in_replay_tail_graph,
)
from sglang.srt.runtime_context import SpawnRanks, get_context, get_flags, publish
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="extra", runner_config="4-gpu-h100")


class _CollectiveTail(DecoderReplayTail):
    @contextmanager
    def _break_context(self, batch):
        with (
            _replay_tail_scope(),
            forward_context(get_forward_context(), forward_batch=batch),
        ):
            yield

    def _layer_stack(self, state, *, forward_batch, input_ids, input_ids_global, **kw):
        values = state.streams[:, 0]
        if (
            in_replay_tail_graph()
            and forward_batch.attn_cp_metadata.moe_local_token_count
            is forward_batch.num_token_non_padded
        ):
            # Attention scratch may leave invalid rows dirty on graph replay.
            live = torch.arange(values.shape[0], device=values.device)
            values = torch.where(
                (live < forward_batch.num_token_non_padded)[:, None],
                values,
                torch.full_like(values, float("nan")),
            )
        transport = MoeRowTransport(
            None,
            values,
            forward_batch,
            input_ids,
            input_ids_global,
            use_cp=True,
            cp_return_contiguous=True,
        )
        with transport.gathered_rows:
            output = transport.input * 1.5
            if input_ids_global is not None:
                output = output + input_ids_global[:, None]
            output = output + output.sum() / 1000
        result = transport.return_output(output)
        return result, None, [result + 0.25]


def _inputs(rank, world, physical, step):
    logical = [physical - int(r >= world - step % world) for r in range(world)]
    count = logical[rank]
    ids = (
        torch.arange(physical, device="cuda", dtype=torch.int64) * world
        + rank
        + 100 * step
    )
    values = ids.float()[:, None].expand(physical, 4).clone()
    ids[count:] = 0
    values[count:] = 0
    scalar = torch.tensor(count, device="cuda", dtype=torch.int32)
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=ids,
        req_pool_indices=torch.tensor([0], device="cuda"),
        seq_lens=torch.tensor([physical * world], device="cuda"),
        out_cache_loc=torch.arange(physical, device="cuda"),
        seq_lens_sum=physical * world,
        num_token_non_padded=scalar,
        global_num_token_non_padded=scalar,
        global_num_token_non_padded_cpu=physical,
        is_extend_in_batch=True,
        attn_cp_metadata=InterleaveContextParallelMetadata(
            per_rank_actual_token=[physical] * world,
            per_rank_logical_token=logical,
            local_index=torch.arange(count, device="cuda"),
            gather_index=torch.arange(physical * world, device="cuda"),
        ),
    )
    return dict(
        state=HcState(values[:, None]),
        positions=ids.clone(),
        input_ids=ids,
        input_ids_global=None,
        forward_batch=batch,
        capture_dspark=False,
    )


def _worker(rank, world, port):
    print(f"rank {rank}: selecting CUDA device", flush=True)
    torch.cuda.set_device(rank)
    print(f"rank {rank}: publishing topology", flush=True)
    with (
        get_context().override_server_args(
            tp_size=world,
            ep_size=world,
            attn_cp_size=world,
            enable_prefill_cp=True,
            cp_strategy="interleave",
            moe_a2a_backend="none",
        ) as server_args,
        get_flags().moe.override(a2a_backend=MoeA2ABackend.NONE),
    ):
        publish(
            server_args, role="test", ranks=SpawnRanks(world_rank=rank, gpu_id=rank)
        )
        print(f"rank {rank}: initializing distributed environment", flush=True)
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            timeout=120,
        )
        print(f"rank {rank}: initializing NCCL collective groups", flush=True)
        # This test exercises CP all-gather/reduce-scatter, not the optional
        # custom all-reduce workspace initialized by a full ModelRunner.
        set_custom_all_reduce(False)
        initialize_model_parallel()
        print(f"rank {rank}: capturing buckets", flush=True)
        init_cp_strategy(
            enable_prefill_cp=True, cp_size=world, cp_strategy="interleave"
        )
        try:
            backend = SimpleNamespace(forward_metadata=None)
            # The process-wide graph pool stays live while recording variants.
            tails = []
            with forward_context(ForwardContext(attn_backend=backend)):
                for routing in (False, True):
                    tail = _CollectiveTail(
                        model=SimpleNamespace(late_layer_vl_routing=routing),
                        layer_ids=range(1),
                        row_step=32,
                        max_rows=128,
                    )
                    tails.append(tail)
                    tail._startup_capture = True
                    for rows in (128, 96, 64, 32):
                        tail.run(**_inputs(rank, world, rows, 0))
                    tail._startup_capture = False
                    recorded = len(tail.runner._graphs)
                    assert recorded == 4
                    for step, rows in enumerate(
                        (31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128), 1
                    ):
                        inputs = _inputs(rank, world, rows, step)
                        expected, _, expected_aux = tail._run_layers_eagerly(**inputs)
                        expected = expected.clone()
                        assert torch.isfinite(expected).all()
                        expected_aux = [x.clone() for x in expected_aux]
                        for buffers in tail.runner._static.values():
                            for value in buffers:
                                value.fill_(
                                    float("nan") if value.is_floating_point() else 999
                                )
                        actual, _, aux = tail.run(**inputs)
                        assert tail._last_run_replayed
                        assert len(tail.runner._graphs) == recorded
                        torch.testing.assert_close(actual, expected)
                        torch.testing.assert_close(aux[0], expected_aux[0])
                        torch.cuda.synchronize()
                    print(
                        f"rank {rank}: routing={routing} buckets match eager",
                        flush=True,
                    )
            torch.cuda.synchronize()
            if rank == 0:
                print(
                    "CP4: all 4 buckets match eager, routing on/off, serving never captures",
                    flush=True,
                )
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


class TestDistributedDecoderTailGraphs(CustomTestCase):
    def test_cp4_buckets_match_eager_with_real_collectives(self):
        if torch.cuda.device_count() < 4:
            self.skipTest("Four CUDA devices required")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(_worker, args=(4, port), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main()
