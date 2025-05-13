import asyncio
import logging
from typing import TYPE_CHECKING

import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers import deepseek_eplb
from sglang.srt.managers.expert_distribution_storage import ExpertDistributionStorage
from sglang.srt.managers.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
)
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class EPLBManager:
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self._model_runner = model_runner

        server_args = model_runner.server_args
        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        assert server_args.eplb_rebalance_num_iterations <= server_args.expert_distribution_recorder_buffer_size, \
            "eplb_rebalance_num_iterations must be less than expert_distribution_recorder_buffer_size"


class _TODO_REMOVE_EPLBManager:
    def bind(self, tokenizer_manager: "TokenizerManager"):
        self._tokenizer_manager = tokenizer_manager
        self._expert_distribution_storage.bind(tokenizer_manager)

    async def handle_loop(self):
        await self._expert_distribution_storage.start()
        TODO_remove_eplb_rebalance_period
        while True:
            sleep_time = self._server_args.eplb_rebalance_period or 1000000000
            logger.info(
                f"EPLBManager: Sleep {sleep_time} seconds before next automatic rebalancing"
            )
            await asyncio.sleep(sleep_time)
            await self.rebalance(EplbRebalanceReqInput())

    async def rebalance(self, obj: EplbRebalanceReqInput):
        await self.save_expert_distribution()
        expert_location_metadata = self.compute_expert_location_metadata(
            debug_use_random_stat=obj.debug_use_random_stat
        )
        await self._tokenizer_manager.update_expert_location(
            UpdateExpertLocationReqInput(
                expert_location_metadata=expert_location_metadata
            )
        )

    async def save_expert_distribution(self):
        await self._expert_distribution_storage.save_current()

    def compute_expert_location_metadata(self, debug_use_random_stat: bool = False):
        snapshot = self._expert_distribution_storage.get_last_snapshot()
        if snapshot is None:
            return ExpertLocationMetadata.init_trivial(self._server_args)

        if debug_use_random_stat:
            logger.warning(
                "EPLBManager.compute_expert_location_metadata use random stat for debugging."
            )
            original_logical_count = torch.tensor(snapshot["logical_count"])
            snapshot = {
                "logical_count": torch.randint_like(original_logical_count, high=100000)
            }

        return ExpertLocationMetadata.init_by_eplb(self._server_args, **snapshot)
