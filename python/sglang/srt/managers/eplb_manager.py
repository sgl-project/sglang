from typing import TYPE_CHECKING

from sglang.srt.managers.expert_distribution_storage import ExpertDistributionStorage

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager


class EPLBManager:
    def __init__(self, tokenizer_manager: TokenizerManager):
        self._tokenizer_manager = tokenizer_manager
        self._expert_distribution_storage = ExpertDistributionStorage()

    async def rebalance_experts(self):
        TODO_may_or_may_not_save_current
        self._expert_distribution_storage.get_last_snapshot()
        TODO
        await self._tokenizer_manager.update_expert_location_metadata(TODO)
