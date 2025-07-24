import json
import logging
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, List

import torch
import torch.distributed

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

    # can be more complex if needed
    def _entrypoint(self):
        while True:
            for _ in range(self._rebalance_num_iterations):
                yield

            yield from self.rebalance()

    def rebalance(self):
        logger.info("[EPLBManager] rebalance start")

        enable_timing = self._rebalance_layers_per_chunk is None

        if enable_timing:
            torch.cuda.synchronize()
            time_start = time.time()

        logical_count = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )["logical_count"]
        expert_location_metadata = ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_runner.model_config, logical_count
        )
        # Export the object to disk, with one file per GPU (rank)
        try:
            output_dir = "/tmp/expert_location_metadata"
            os.makedirs(output_dir, exist_ok=True)

            rank = torch.distributed.get_rank()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(output_dir, f"expert_metadata_rank_{rank}_{timestamp}.json")

            data_to_save = {
                "physical_to_logical_map": expert_location_metadata.physical_to_logical_map.cpu()
                .numpy()
                .tolist(),
                "logical_to_all_physical_map": expert_location_metadata.logical_to_all_physical_map.cpu()
                .numpy()
                .tolist(),
                "logical_to_all_physical_map_num_valid": expert_location_metadata.logical_to_all_physical_map_num_valid.cpu()
                .numpy()
                .tolist(),
            }
            if (
                expert_location_metadata.logical_to_rank_dispatch_physical_map
                is not None
            ):
                data_to_save["logical_to_rank_dispatch_physical_map"] = (
                    expert_location_metadata.logical_to_rank_dispatch_physical_map.cpu()
                    .numpy()
                    .tolist()
                )

            with open(file_path, "w") as f:
                json.dump(data_to_save, f, indent=4)

            logger.info(
                f"[EPLBManager] Saved expert location metadata for rank {rank} to {file_path}"
            )
        except ImportError as e:
            logger.warning(
                f"[EPLBManager] Could not save expert metadata, import error: {e}"
            )
        except Exception as e:
            logger.error(
                f"[EPLBManager] Error saving expert metadata: {e}", exc_info=True
            )

        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            if len(update_layer_ids_chunks) > 1:
                yield
            self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.cuda.synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        all_layer_ids = sorted(
            list(self._model_runner.model.routed_experts_weights_of_layer.keys())
        )
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))


def _chunk_list(items: List, chunk_size):
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]
