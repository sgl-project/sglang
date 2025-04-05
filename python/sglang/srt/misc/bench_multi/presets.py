from typing import List, Any, Dict

from sglang.srt.misc.bench_multi.configs import Config


def get_configs_debug():
    return TODO


def get_configs_scan_DeepSeekV3_8xH200():
    return TODO


def get_configs_scan_DeepSeekV3_2x8xH100():
    return TODO


def get_configs_scan_DeepSeekV3_4x8xH100():
    return TODO


def _compute_scan_config(
    server_args: Dict[str, Any],
    random_input_lens: List[int],
    random_output_lens: List[int],
):
    return Config(
        server_args=dict(
            stream_output=True,
            disable_radix_cache=True,
            **server_args,
        ),
        bench_serving_args_list=[
            dict(
                random_input_len=random_input_len,
                random_output_len=random_output_len,
                flush_cache=True,
                dataset_name="random",  # TODO
                random_range_ratio=0.0,  # TODO
                max_concurrency=128,  # TODO
                num_prompts=512,  # TODO
                request_rate=float("inf"),  # TODO
            )
            for random_input_len in random_input_lens
            for random_output_len in random_output_lens
        ],
    )
