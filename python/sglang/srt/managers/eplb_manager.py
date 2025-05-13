import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers import deepseek_eplb
from sglang.srt.managers.expert_distribution_storage import ExpertDistributionStorage
from sglang.srt.managers.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
)
from sglang.srt.managers.io_struct import (
    EplbRebalanceReqInput,
    UpdateExpertLocationReqInput,
)
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)

TODO_rm


class EPLBManager:
    def __init__(self, server_args: ServerArgs):
        super().__init__()
        self._server_args = server_args
        TODO_remove_eplb_storage_dir
        self._expert_distribution_storage = ExpertDistributionStorage(
            dir_data=Path(self._server_args.eplb_storage_dir)
                     / "expert_distribution_storage"
        )

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
