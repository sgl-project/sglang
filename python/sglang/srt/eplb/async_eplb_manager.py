import logging
from multiprocessing import Manager, Process, Queue
from typing import TYPE_CHECKING

import torch

from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ExpertLocationMetadata

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


logger = logging.getLogger(__name__)


class AsyncEPLBManager(EPLBManager):
    def __init__(self, model_runner: "ModelRunner"):
        super().__init__(model_runner)

        self.device = self._server_args.device

        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

        self.planner_q = Queue()
        self.block_q = Queue(maxsize=1)

        self.num_wait_worker_iterations = 40

        self.manager = Manager()
        self.shared_dict = self.manager.dict(
            {
                "moe_load": None,
            }
        )

        self.eplb = EplbProcess(
            shared_dict=self.shared_dict,
            planner_q=self.planner_q,
            block_q=self.block_q,
            server_args=self._server_args,
            model_config=self._model_runner.model_config,
            rank=self.rank,
        )

        self.eplb_process = self.eplb.launch_process()

        logger.info(
            f"[ModelRunner] Launched EPLB process (pid={self.eplb_process.pid})"
        )

    def _entrypoint(self):
        while True:
            for _ in range(self._rebalance_num_iterations):
                yield
            self.forward_eplb_process()
            for _ in range(self.num_wait_worker_iterations):
                yield
            self.take_update_info_from_eplb_process()
            yield from self.forward_update_weight()

    def forward_eplb_process(self):
        logger.info("[EPLBManager] rebalance start")
        logical_count = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )["logical_count"]
        self.shared_dict["moe_load"] = logical_count.cpu()
        self.wakeup_eplb_worker()

    def forward_update_weight(self):
        expert_location_metadata = self.to_device(self.update_info_all)

        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            if len(update_layer_ids_chunks) > 1:
                yield

            yield from self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )

    def to_device(self, metadata):
        fields = (
            "physical_to_logical_map",
            "logical_to_all_physical_map",
            "logical_to_all_physical_map_num_valid",
            "logical_to_rank_dispatch_physical_map",
        )
        for name in fields:
            t = getattr(metadata, name, None)
            if t is None:
                continue
            if hasattr(t, "device") and t.device != self.device:
                setattr(metadata, name, t.to(self.device, non_blocking=True))
        return metadata

    def take_update_info_from_eplb_process(self):
        self.update_info_all = self.block_q.get()

    def wakeup_eplb_worker(self):
        self.planner_q.put(1)


class EplbProcess:
    def __init__(
        self,
        shared_dict,
        planner_q,
        block_q,
        server_args,
        model_config,
        rank,
    ):
        self.shared_dict = shared_dict
        self.planner_q = planner_q
        self.block_q = block_q

        self._server_args = server_args
        self._model_config = model_config
        self.rank = rank
        self._server_args.device = "cpu"

    def do_algorithm(self):
        logical_count = self.shared_dict["moe_load"]
        return ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_config, logical_count, self.rank
        )

    def worker_process(self, planner_q, block_q):
        while True:
            try:
                planner_q.get()
                update_info = self.do_algorithm()

                while True:
                    if not block_q.empty():
                        continue
                    block_q.put(update_info)
                    break

            except Exception as e:
                logger.warning(
                    f"[EPLB process] Exiting due to error:{e}", exc_info=True
                )
                break

    def launch_process(self):
        proc = Process(
            target=self.worker_process, args=(self.planner_q, self.block_q), daemon=True
        )
        proc.start()
        return proc
