import json
import time
from pathlib import Path
from typing import Any, Optional, Dict

from sglang.srt.managers.tokenizer_manager import TokenizerManager


class ExpertDistributionStorage:
    def __init__(self, dir_data, tokenizer_manager: TokenizerManager):
        self._dir_data = Path(dir_data)
        self._tokenizer_manager = tokenizer_manager

    async def initialize(self):
        await self._tokenizer_manager.start_expert_distribution_record()

    async def save_current(self):
        data = await self._tokenizer_manager.dump_expert_distribution_record()
        (self._dir_data / f"{time.time_ns()}.json").write_text(json.dumps(data))

    def get_last_snapshot(self) -> Optional[Dict[str, Any]]:
        paths = sorted(list(self._dir_data.glob("*.json")), key=lambda p: int(p.stem))
        if len(paths) == 0:
            return None
        path = paths[-1]
        return json.loads(path.read_text())
