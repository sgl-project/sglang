import functools
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from sglang.srt.utils import get_device_name

logger = logging.getLogger(__name__)


def get_config_file_name(attention_backend, num_local_heads) -> str:
    device_name = get_device_name().replace(" ", "_")
    return (
        f"H={num_local_heads},device_name={device_name}_attn={attention_backend}.json"
    )


def get_config_file_with_dir(attention_backend, num_local_heads) -> str:
    json_file_name = get_config_file_name(attention_backend, num_local_heads)
    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
        json_file_name,
    )
    return config_file_path


@functools.lru_cache
def get_mha_chunk_configs(
    attention_backend: str,
    num_local_heads: int,
) -> Optional[Dict[int, Dict[int, Dict[int, int]]]]:
    # return: {bs: {prefix_len: {extend_len: min prefill_len*seq_len}}}
    attention_backend = attention_backend or "flashinfer"

    config_file_path = get_config_file_with_dir(attention_backend, num_local_heads)
    if not os.path.exists(config_file_path) and attention_backend != "flashinfer":
        logger.info(
            f"Mha prefix config file {config_file_path} not found, try to load flashinfer backend config"
        )
        attention_backend = "flashinfer"
        config_file_path = get_config_file_with_dir(attention_backend, num_local_heads)

    if not os.path.exists(config_file_path):
        # If no optimized configuration is available, we will use the default
        # configuration
        logger.warning(
            (
                "Not using tuned mha chunk prefix configs. Performance might be sub-optimal! Config file not found at %s, "
                "you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/mha_chunk_prefix/"
            ),
            config_file_path,
        )
        return None
    with open(config_file_path) as f:
        logger.info(f"Using mha chunk prefix config from {config_file_path}.")
        # {bs: {prefix_len: {extend_len: min prefill_len*seq_len}}}
        rst_configs = {}
        for bs, prefix_dict in json.load(f).items():
            prefix_configs = {}
            for prefix, extend_dict in prefix_dict.items():
                extend_configs = {
                    int(extend): int(se_mul) for extend, se_mul in extend_dict.items()
                }
                prefix_configs[int(prefix)] = dict(
                    sorted(extend_configs.items(), reverse=True)
                )
            rst_configs[int(bs)] = dict(sorted(prefix_configs.items(), reverse=True))
        return rst_configs


def tuned_dispatch_mha_chunk(
    attention_backend: str, num_local_heads, prefix_lens: List[int], seq_lens: List[int]
) -> Optional[bool]:
    rst_configs = get_mha_chunk_configs(attention_backend, num_local_heads)
    if rst_configs is None:
        return None
    bs = len(prefix_lens)
    assert bs >= 1
    assert bs == len(
        seq_lens
    ), f"Length of prefix_lens and seq_lens should be the same, but got {len(prefix_lens)} and {len(seq_lens)}"
    if bs > max(rst_configs.keys()):
        bs = max(rst_configs.keys())
    if bs not in rst_configs:
        return None
    configs = rst_configs[bs]
    extend_lens = [s - p for p, s in zip(prefix_lens, seq_lens)]
    prefix_sum = sum(prefix_lens)
    extend_sum = sum(extend_lens)
    seq_extend_mul = sum([e * s for e, s in zip(extend_lens, seq_lens)])

    for cfg_prefill, cfg_extend_items in configs.items():
        if prefix_sum >= cfg_prefill:
            for cfg_extend, cfg_min_seq_extend_mul in cfg_extend_items.items():
                if extend_sum >= cfg_extend:
                    return seq_extend_mul >= cfg_min_seq_extend_mul
    return False
