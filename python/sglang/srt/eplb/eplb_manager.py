import logging
import time
import threading
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch.cuda
from torch.distributed import ProcessGroup

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ExpertLocationMetadata

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class EPLBManager:
    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self._model_runner = model_runner
        self._server_args = model_runner.server_args
        self._rebalance_async = self._server_args.enable_eplb_rebalance_async
        self._src_rank = 0
        world_size = self._model_runner.tp_group.world_size
        nnodes = self._server_args.nnodes
        self._num_gpu_per_node = world_size // nnodes
        self._rebalance_layers_per_chunk = (
            self._server_args.eplb_rebalance_layers_per_chunk
        )
        self._rebalance_num_iterations = self._server_args.eplb_rebalance_num_iterations

        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        assert (
            self._server_args.eplb_rebalance_num_iterations
            >= self._server_args.expert_distribution_recorder_buffer_size
        ), "eplb_rebalance_num_iterations must be greater than expert_distribution_recorder_buffer_size"

        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._rebalance_num_iterations} iterations."
        )

        self._main_generator = self._entrypoint()

    def on_forward_pass_end(self):
        next(self._main_generator)
        self._step_counter += 1

    # can be more complex if needed
    def _entrypoint(self):
        while True:
            self.initial_rebalance()

            for _ in range(self._rebalance_num_iterations):
                yield

            yield from self.rebalance()

    def initial_rebalance(self):
        self._step_counter = 0
        if self._rebalance_async:
            self._begin_transfer_step = None
            self._compute_ongoing = False
            self._tp_sync_ongoing = False
            self._rebalance_result = None

    def rebalance(self):
        logger.info("[EPLBManager] rebalance start")
        enable_timing = self._rebalance_layers_per_chunk is None
        if enable_timing:
            torch.cuda.synchronize()
            time_start = time.time()
        if self._rebalance_async:
            # async: overlap eplb computing with model_runner-forward
            yield from self.compute()
            yield from self.tp_sync()
            # ensure all the tp enter transfer stage at the same forward-pass
            while self._begin_transfer_step is None or self._step_counter<self._begin_transfer_step:
                yield
        else:
            # synchronous mode: directly run on the main thread
            logical_count = get_global_expert_distribution_recorder().dump_record(
                output_mode="object"
            )["logical_count"]
            self._rebalance_result = ExpertLocationMetadata.init_by_eplb(
                self._server_args, self._model_runner.model_config, logical_count
            )
        # ------------- real parameter transfer --------------------
        yield from self.transfer_parameter()

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.cuda.synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)
        
    def compute(self):
        logical_count = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )["logical_count"]
        # use rank-src broadcast to make `logical_count_sum` identical
        logical_count_sum = logical_count.sum(dim=0).clone()
        src_rank = self._src_rank
        self._model_runner.tp_group.broadcast(logical_count_sum, src=src_rank)
        yield
        torch.cuda.synchronize()
        self._compute_ongoing = True
        self._compute_thread = threading.Thread(
                target=self._compute_expert_metadata,
                args=(logical_count_sum,),
                daemon=True
            )
        self._compute_thread.start()
        yield
        # spin until the computation completes
        while self._compute_ongoing:
            yield
    
    def _compute_expert_metadata(self, logical_count):
        local_rank = self._model_runner.tp_rank
        num_gpu_per_node = self._num_gpu_per_node
        torch.cuda.set_device(local_rank % num_gpu_per_node)
        expert_location_metadata = ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_runner.model_config, logical_count
        )
        self._rebalance_result = expert_location_metadata
        self._compute_ongoing = False

    def tp_sync(self):
        """Barrier-like handshake among all TP ranks.

        Each rank sends a step counter to every other rank and
        waits for all peers' counters.  When all messages are completed,
        we know that the new expert mapping is globally visible and can
        be applied safely.
        """
        self._send_signal_step = self._step_counter
        send_works, recv_works, buffer_kept = self._gen_rec_send_works()
        self._tp_sync_ongoing = True
        self._tp_sync_thread = threading.Thread(
                target=self._wait_for_tp_sync_signals,
                args=(send_works, recv_works,),
                daemon=True
            )
        self._tp_sync_thread.start()
        yield
        while self._tp_sync_ongoing:
            yield 
        # the largest counter among all ranks indicates the earliest
        # forward-pass index after which the new mapping becomes valid
        max_tensor = torch.stack(buffer_kept).max()
        # Add a buffer of 2 steps to ensure all ranks see the new step before acting on it.
        self._begin_transfer_step = max(max_tensor.item(),self._send_signal_step) + 2

    def _gen_rec_send_works(self):
        world_size = self._model_runner.tp_group.world_size
        local_rank = self._model_runner.tp_group.rank_in_group
        # Using the dedicated CPU(Gloo) process-group avoids touching NCCL streams and
        # therefore never blocks the GPU-side decoding kernels that are running in parallel.
        group = self._model_runner.tp_group.cpu_group
        send_works = []
        recv_buffer_kept = []
        recv_works = []
        for src in range(world_size):
            if src == local_rank:
                continue
            w, t = self._recv_single_signal(src, group)
            recv_works.append(w)
            recv_buffer_kept.append(t)
        for dst in range(world_size):
            if dst == local_rank:
                continue
            w, t = self._send_single_signal(dst, self._send_signal_step, group)
            send_works.append(w)
        return send_works, recv_works, recv_buffer_kept
    
    def _send_single_signal(self, dst_rank_in_group: int, value: int, group: Optional[ProcessGroup] = None,
                            ) -> Tuple[Optional[torch.distributed.Work], torch.Tensor]:
        signal_tensor = torch.tensor([value], dtype=torch.long)
        w = torch.distributed.isend(signal_tensor, dst=dst_rank_in_group, group=group)
        return w, signal_tensor
    
    def _recv_single_signal(self, src_rank_in_group: int, group: Optional[ProcessGroup] = None, 
                            ) -> Tuple[Optional[torch.distributed.Work], torch.Tensor]:
        signal_tensor = torch.empty(1, dtype=torch.long)
        w = torch.distributed.irecv(signal_tensor, src=src_rank_in_group, group=group)
        return w, signal_tensor
    
    def _wait_for_tp_sync_signals(self, send_works,recv_works):
        works = send_works + recv_works
        for w in works:
            w.wait()
        self._tp_sync_ongoing = False

    def transfer_parameter(self):
        expert_location_metadata = self._rebalance_result
        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            if len(update_layer_ids_chunks) > 1:
                yield
            self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        all_layer_ids = sorted(
            list(self._model_runner.model.routed_experts_weights_of_layer.keys())
        )
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))

def _chunk_list(items: List, chunk_size):
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]
