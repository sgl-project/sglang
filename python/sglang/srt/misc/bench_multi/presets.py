from typing import List

from sglang.srt.misc.bench_multi.configs import Config


def get_configs(preset_name: str) -> List[Config]:
    if preset_name == "scan_8xH200":
        return TODO
    if preset_name == "scan_2x8xH100":
        return TODO
    if preset_name == "scan_4x8xH100":
        return TODO
    raise NotImplementedError(f"Unknown {preset_name=}")


def _compute_scan_config():
    return Config(
        server_args=dict(
            stream_output=True,
            disable_radix_cache=True,
        ),
        bench_serving_args_list=[
            dict(
                dataset_name="random",  # TODO
                flush_cache=True,
                random_range_ratio=0.0,  # TODO
            )
            for what in TODO
        ],
    )
