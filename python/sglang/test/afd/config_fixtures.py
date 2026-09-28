"""Minimal argument builders shared by AFD CPU contracts."""

from types import SimpleNamespace

from sglang.srt.afd import config


def _server_args(**overrides):
    values = {
        "afd_execution_mode": "attention",
        "afd_config": config.AFDConfig(),
        "tp_size": 1,
        "dp_size": 1,
        "ep_size": 1,
        "pp_size": 1,
        "nnodes": 1,
        "disable_overlap_schedule": True,
        "disable_cuda_graph": True,
        "enable_two_batch_overlap": False,
        "enable_single_batch_overlap": False,
        "moe_a2a_backend": "none",
        "speculative_algorithm": None,
        "speculative_num_steps": None,
        "enable_dp_attention": False,
        "enable_dp_lm_head": False,
        "moe_runner_backend": "auto",
        "context_length": None,
        "page_size": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _lane_server_args(*, role, lanes, attention_lanes=None, **overrides):
    return _server_args(
        afd_execution_mode=role,
        afd_config=config.AFDConfig(lanes=lanes, attention_lanes=attention_lanes),
        **overrides,
    )
