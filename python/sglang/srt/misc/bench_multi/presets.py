from functools import partial
from typing import List

from sglang.srt.misc.bench_multi.configs import Config


def get_configs_debug():
    return [
        _compute_scan_config(model_path=TODO, tp_size=1, random_input_lens=[100, 1000], random_output_lens=[10]),
        _compute_scan_config(model_path=TODO, tp_size=2, random_input_lens=[1000], random_output_lens=[10, 100]),
    ]


def get_configs_scan_DeepSeekV3_8xH200():
    compute = partial(_compute_scan_config, random_input_lens=_SCAN_INPUT_LENS, random_output_lens=_SCAN_OUTPUT_LENS)
    return [
        compute(tp_size=8),
        compute(tp_size=8, dp_size=8, enable_dp_attention=True),
        compute(tp_size=8, enable_ep_moe=True),
        compute(tp_size=8, dp_size=8, enable_dp_attention=True, enable_ep_moe=True),
        # TODO
    ]


def get_configs_scan_DeepSeekV3_2x8xH100():
    compute = partial(_compute_scan_config, random_input_lens=_SCAN_INPUT_LENS, random_output_lens=_SCAN_OUTPUT_LENS)
    return [
        compute(tp_size=16),
        compute(tp_size=16, dp_size=16, enable_dp_attention=True),
        compute(tp_size=16, enable_ep_moe=True),
        compute(tp_size=16, dp_size=16, enable_dp_attention=True, enable_ep_moe=True),
        # TODO
    ]


def get_configs_scan_DeepSeekV3_4x8xH100():
    compute = partial(_compute_scan_config, random_input_lens=_SCAN_INPUT_LENS, random_output_lens=_SCAN_OUTPUT_LENS)
    return [
        compute(tp_size=32),
        compute(tp_size=32, dp_size=32, enable_dp_attention=True),
        compute(tp_size=32, enable_ep_moe=True),
        compute(tp_size=32, dp_size=32, enable_dp_attention=True, enable_ep_moe=True),
        # TODO
    ]


def _compute_deepseekv3_scan_config():
    return _compute_scan_config(
        random_input_lens=[100, 1000, 10000, 100000],
        random_output_lens=[100, 1000, 10000],  # TODO
    )


def _compute_scan_config(
    random_input_lens: List[int],
    random_output_lens: List[int],
    num_repeat: int = 2,
    **kwargs,
):
    return Config(
        server_args=dict(
            stream_output=True,
            disable_radix_cache=True,
            **kwargs,
            # TODO attn backend
            # TODO blockwise fp8
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
            for _ in range(num_repeat)
            for random_input_len in random_input_lens
            for random_output_len in random_output_lens
        ],
    )
