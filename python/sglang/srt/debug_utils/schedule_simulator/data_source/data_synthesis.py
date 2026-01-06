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

    print(
        f"Generated {len(requests)} random requests "
        f"(input_len={input_len}, output_len={output_len}, range_ratio={range_ratio})"
    )
    return requests


def generate_gsp_requests(
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    range_ratio: float = 1.0,
    seed: Optional[int] = None,
) -> List[SimRequest]:
    if seed is not None:
        random.seed(seed)

    requests = []
    idx = 0
    for group_idx in range(num_groups):
        group_id = f"g{group_idx}"
        prefix_len = _random_len(system_prompt_len, range_ratio)
        for _ in range(prompts_per_group):
            q_len = _random_len(question_len, range_ratio)
            osl = _random_len(output_len, range_ratio)
            requests.append(
                SimRequest(
                    request_id=f"gsp{idx}",
                    input_len=prefix_len + q_len,
                    output_len=osl,
                    group_id=group_id,
                    prefix_len=prefix_len,
                )
            )
            idx += 1

    random.shuffle(requests)
    print(
        f"Generated {len(requests)} GSP requests "
        f"({num_groups} groups x {prompts_per_group} prompts, "
        f"system_prompt_len={system_prompt_len}, question_len={question_len}, "
        f"output_len={output_len})"
    )
    return requests


def _random_len(full_len: int, range_ratio: float) -> int:
    min_len = max(int(full_len * range_ratio), 1)
    return random.randint(min_len, full_len)
