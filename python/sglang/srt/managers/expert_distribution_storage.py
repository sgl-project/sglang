import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, Dict

from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class ExpertDistributionStorage:
    def __init__(self, dir_data, tokenizer_manager: TokenizerManager):
        self._dir_data = Path(dir_data)
        self._tokenizer_manager = tokenizer_manager
        if not self._dir_data.exists():
            self._dir_data.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        await self._tokenizer_manager.start_expert_distribution_record()

    async def save_current(self):
        data = await self._tokenizer_manager.dump_expert_distribution_record()
        path = self._dir_data / f"{time.time_ns()}.json"
        logger.info(f"save_current to path {path}")
        path.write_text(json.dumps(data))

    def get_last_snapshot(self) -> Optional[Dict[str, Any]]:
        paths = sorted(list(self._dir_data.glob("*.json")), key=lambda p: int(p.stem))
        if len(paths) == 0:
            return None
        path = paths[-1]
        logger.info(f"get_last_snapshot choose path {path}")
        return json.loads(path.read_text())
