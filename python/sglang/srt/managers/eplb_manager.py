from pathlib import Path
from typing import TYPE_CHECKING

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers import deepseek_eplb
from sglang.srt.managers.expert_distribution_storage import ExpertDistributionStorage
from sglang.srt.managers.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
)
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager


class EPLBManager:
    @staticmethod
    def init_new(server_args: ServerArgs):
        if server_args.enable_eplb:
            return _EPLBManagerReal(server_args)
        else:
            return _EPLBManagerNoop()

    async def initialize(self, tokenizer_manager: TokenizerManager):
        pass

    def compute_expert_location_metadata(self) -> ExpertLocationMetadata:
        raise NotImplementedError


class _EPLBManagerReal(EPLBManager):
    def __init__(self, server_args: ServerArgs):
        super().__init__()
        self._server_args = server_args

    async def initialize(self, tokenizer_manager: TokenizerManager):
        self._expert_distribution_storage = ExpertDistributionStorage(
            dir_data=Path(self._server_args.eplb_storage_dir) / "expert_distribution_storage",
            tokenizer_manager=tokenizer_manager,
        )
        await self._expert_distribution_storage.initialize()

    def compute_expert_location_metadata(self):
        logical_count = self._expert_distribution_storage.get_last_snapshot()["logical_count"]
        if logical_count is None:
            return ExpertLocationMetadata.init_trivial(self._server_args)
        return ExpertLocationMetadata.init_by_eplb(self._server_args, logical_count=logical_count)


class _EPLBManagerNoop(EPLBManager):
    pass
