import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class ExpertDistributionStorage:
    def __init__(self, dir_data):
        self._dir_data = Path(dir_data)
        if not self._dir_data.exists():
            self._dir_data.mkdir(parents=True, exist_ok=True)

    def bind(self, tokenizer_manager: "TokenizerManager"):
        self._tokenizer_manager = tokenizer_manager

    async def start(self):
        await self._tokenizer_manager.start_expert_distribution_record()

    async def save_current(self):
        data = await self._tokenizer_manager.dump_expert_distribution_record()
        path = self._dir_data / f"{time.time_ns()}.json"
        logger.info(f"save_current to path {path}")
        path.write_text(json.dumps(data))

    # Most vanilla method since I do not have production environment data to test what algorithm is better
    def get_last_snapshot(self) -> Optional[Dict[str, Any]]:
        path = self.get_last_snapshot_path(self._dir_data)
        if path is None:
            return None
        logger.info(f"get_last_snapshot choose path {path}")
        return json.loads(path.read_text())

    @staticmethod
    def get_last_snapshot_path(dir_data) -> Optional[Path]:
        paths = sorted(list(Path(dir_data).glob("*.json")), key=lambda p: int(p.stem))
        if len(paths) == 0:
            return None
        return paths[-1]
