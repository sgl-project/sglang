from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkConfig:
    request_rate: float = float("inf")
    max_concurrency: Optional[int] = None
    ignore_request_timestamp: bool = False
