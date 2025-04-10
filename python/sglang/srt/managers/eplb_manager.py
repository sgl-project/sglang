import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers import deepseek_eplb
from sglang.srt.managers.expert_distribution_storage import ExpertDistributionStorage
from sglang.srt.managers.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
)
from sglang.srt.managers.io_struct import UpdateExpertLocationReqInput
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class EPLBManager:
    def __init__(self, server_args: ServerArgs):
        super().__init__()
        self._server_args = server_args
        self._expert_distribution_storage = ExpertDistributionStorage(
            dir_data=Path(self._server_args.eplb_storage_dir)
                     / "expert_distribution_storage"
        )

    def bind(self, tokenizer_manager: "TokenizerManager"):
        self._tokenizer_manager = tokenizer_manager
        self._expert_distribution_storage.bind(tokenizer_manager)

    async def handle_loop(self):
        await self._expert_distribution_storage.start()
        while True:
            sleep_time = self._server_args.eplb_rebalance_period or 1000000000
            logger.info(f"Sleep {sleep_time} seconds before automatically trigger rebalancing")
            await asyncio.sleep(sleep_time)
            self.save_expert_distribution()
            await self.rebalance()

    async def rebalance(self):
        expert_location_metadata = self.compute_expert_location_metadata()
        await self._tokenizer_manager.update_expert_location(
            UpdateExpertLocationReqInput(expert_location_metadata=expert_location_metadata))

    def save_expert_distribution(self):
        self._expert_distribution_storage.save_current()

    def compute_expert_location_metadata(self):
        logical_count = self._expert_distribution_storage.get_last_snapshot()[
            "logical_count"
        ]
        if logical_count is None:
            return ExpertLocationMetadata.init_trivial(self._server_args)
        return ExpertLocationMetadata.init_by_eplb(
            self._server_args, logical_count=logical_count
        )
