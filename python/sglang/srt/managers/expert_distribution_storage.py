from pathlib import Path

from sglang.srt.managers.tokenizer_manager import TokenizerManager


class ExpertDistributionStorage:
    def __init__(self, dir_data, tokenizer_manager: TokenizerManager):
        self._dir_data = Path(dir_data)
        self._tokenizer_manager = tokenizer_manager

    async def initialize(self):
        await self._tokenizer_manager.start_expert_distribution_record()

    async def save_current(self):
        data = await self._tokenizer_manager.dump_expert_distribution_record()
        TODO_write_data

    def get_last_snapshot(self):
        return TODO_read_data
