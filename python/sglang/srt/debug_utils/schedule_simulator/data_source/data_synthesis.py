import random
from typing import List, Optional

from sglang.srt.debug_utils.schedule_simulator.request import SimRequest


def generate_random_requests(
    num_requests: int,
    input_len: int,
    output_len: int,
    range_ratio: float = 1.0,
    seed: Optional[int] = None,
) -> List[SimRequest]:
    """
    Generate random requests with random input/output lengths.

    Args:
        num_requests: Number of requests to generate
        input_len: Target input length (max if range_ratio < 1)
        output_len: Target output length (max if range_ratio < 1)
        range_ratio: Ratio to determine min length. E.g., 0.5 means lengths
                     are sampled from [input_len * 0.5, input_len]
        seed: Random seed for reproducibility

    Returns:
        List of SimRequest objects
    """
    if seed is not None:
        random.seed(seed)

    requests = []
    for i in range(num_requests):
        isl = _random_len(input_len, range_ratio)
        osl = _random_len(output_len, range_ratio)
        requests.append(
            SimRequest(
                request_id=f"syn{i}",
                input_len=isl,
                output_len=osl,
            )
        )

    return requests


def _random_len(full_len: int, range_ratio: float) -> int:
    min_len = max(int(full_len * range_ratio), 1)
    return random.randint(min_len, full_len)
