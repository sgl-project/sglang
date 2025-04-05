from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class Config:
    server_args: Dict[str, Any]
    bench_serving_args_list: List[Dict[str, Any]]
