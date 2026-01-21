"""
Integration tests for ParallelismContext with real sglang servers.

Tests that ParallelismContext can instantiate models with correct tensor parallel
sharding by comparing parameter names and sizes against a running sglang server.

Run with:
    pytest test/srt/test_parallelism_context_integration.py -v

Full test suite (non-CI):
    - TP=2 small model (Qwem2.5-1.5B-Instruct)
    - EP=2 small MOE model (DeepSeek-Coder-V2-Lite-Instruct)
    - MLA model with hybrid dp attention (DeepSeek-Coder-V2-Lite-Instruct)

CI test (reduced):
    - TP=2 small model only
"""

import dataclasses
import gc
from typing import Dict, List, Tuple

import pytest
import requests
import torch

from sglang.srt.distributed.parallel_state import RankParallelismConfig
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
)
from sglang.utils import terminate_process


def get_transfer_engine_info(url: str, rank: int) -> Dict:
    """Get transfer engine info (parameter names and sizes) for a rank."""
    response = requests.get(
        f"{url}/get_remote_instance_transfer_engine_info",
        params={"rank": rank},
    )
    response.raise_for_status()
    return response.json()


def get_parallelism_config(url: str, rank: int) -> Dict:
    """Get parallelism config for a rank."""
    response = requests.get(f"{url}/parallelism_config", params={"rank": rank})
    response.raise_for_status()
    return response.json()


def get_server_info(url: str) -> Dict:
    """Get server info."""
    response = requests.get(f"{url}/server_info")
    response.raise_for_status()
    return response.json()


def verify_model_params_match_for_rank(
    url: str,
    rank: int,
    server_info: Dict,
    test_gpu_id: int,
):
    """Verify model parameters match for a specific rank by recreating a model shard.

    Args:
        url: Server URL
        rank: The rank to verify
        server_info: Server info dict
        test_gpu_id: GPU ID to use for instantiating the test model
    """
    transfer_info = get_transfer_engine_info(url, rank)
    server_weights_info = transfer_info["remote_instance_transfer_engine_info"][1]

    # Get parallelism config from running server
    parallelism_config_data = get_parallelism_config(url, rank)
    parallelism_config = RankParallelismConfig.from_dict(parallelism_config_data)
    # Get server args from server info
    from sglang.srt.server_args import ServerArgs

    valid_fields = {f.name for f in dataclasses.fields(ServerArgs)}
    filtered_info = {k: v for k, v in server_info.items() if k in valid_fields}
    filtered_info.pop("model_config", None)
    server_args = ServerArgs(**filtered_info)

    from sglang.srt import server_args as server_args_module
    from sglang.srt.distributed.parallel_state import ParallelismContext

    original_global_server_args = server_args_module._global_server_args

    try:
        # In a Mock ParallelismContext, instantiate the model for this rank.
        # Use a separate GPU (test_gpu_id) to avoid memory conflicts with the running server.
        server_args_module._global_server_args = server_args
        with ParallelismContext(parallelism_config):
            from sglang.srt.configs.device_config import DeviceConfig
            from sglang.srt.configs.load_config import LoadConfig
            from sglang.srt.configs.model_config import ModelConfig
            from sglang.srt.model_loader import get_model

            model_config = ModelConfig.from_server_args(server_args)
            load_config = LoadConfig(load_format="dummy")
            device_config = DeviceConfig(device="cuda", gpu_id=test_gpu_id)

            torch.cuda.set_device(test_gpu_id)
            model = get_model(
                model_config=model_config,
                load_config=load_config,
                device_config=device_config,
            )
            model_params = {}
            for name, param in model.named_parameters():
                model_params[name] = param.numel() * param.element_size()

            # Verify all server parameters exist in model with same size
            mismatches = []
            missing = []
            for param_name, (ptr, numel, elem_size) in server_weights_info.items():
                expected_size = numel * elem_size
                if param_name not in model_params:
                    missing.append(param_name)
                elif model_params[param_name] != expected_size:
                    mismatches.append(
                        f"{param_name}: model={model_params[param_name]}, server={expected_size}"
                    )

            assert not missing, f"Rank {rank}: Missing parameters: {missing}"
            assert not mismatches, f"Rank {rank}: Size mismatches: {mismatches}"
            del model
            torch.cuda.empty_cache()

    finally:
        server_args_module._global_server_args = original_global_server_args


TEST_CONFIGS: List[Tuple[str, str, int, List[str], int, bool]] = [
    # Basic TP=2 test (CI only)
    (
        "tp2_small",
        DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
        2,
        [],
        2,
        False,
    ),
    # EP=2: MoE experts split across 2 groups, moe_tp=1 per group
    (
        "mla_ep2",
        DEFAULT_MLA_MODEL_NAME_FOR_TEST,
        2,
        ["--ep-size", "2"],
        2,
        True,
    ),
    (
        "mla_dp2_tp4",
        DEFAULT_MLA_MODEL_NAME_FOR_TEST,
        4,
        ["--enable-dp-attention", "--dp", "2"],
        4,
        True,
    ),
    (
        "mla_dp2_ep2_tp4",
        DEFAULT_MLA_MODEL_NAME_FOR_TEST,
        4,
        ["--enable-dp-attention", "--dp", "2", "--ep-size", "2"],
        4,
        True,
    ),
    (
        "mla_dp2_ep4_tp4",
        DEFAULT_MLA_MODEL_NAME_FOR_TEST,
        4,
        ["--enable-dp-attention", "--dp", "2", "--ep-size", "4"],
        4,
        True,
    ),
    (
        "mla_dp4_ep2_tp4",
        DEFAULT_MLA_MODEL_NAME_FOR_TEST,
        4,
        ["--enable-dp-attention", "--dp", "4", "--ep-size", "2"],
        4,
        True,
    ),
]


def get_test_configs():
    if is_in_ci():
        return [TEST_CONFIGS[0]]
    else:
        return TEST_CONFIGS


def _get_test_params():
    """Generate pytest parameters based on test configs."""
    configs = get_test_configs()
    params = []
    ids = []
    for (
        test_id,
        model_name,
        tp_size,
        extra_args,
        min_gpus,
        trust_remote_code,
    ) in configs:
        params.append(
            pytest.param(
                (model_name, tp_size, extra_args, min_gpus, trust_remote_code),
                id=test_id,
            )
        )
    return params


class TestParallelismContextIntegration:
    """
    Test that ParallelismContext can instantiate models with the same
    parameter names and sizes as the sglang server engine.
    """

    @pytest.mark.parametrize("config", _get_test_params())
    def test_model_instantiation_matches_server(self, config):
        """
        Test that a model instantiated with ParallelismContext has the same
        parameter names and sizes as the model in the sglang server.

        This test:
        1. Starts a server with specified parallelism config
        2. Gets transfer_engine_info for all ranks (contains param names and sizes)
        3. Gets parallelism_config and server_info
        4. Uses ParallelismContext to instantiate a model for each rank
        5. Compares the parameter names and sizes
        """
        model_name, tp_size, extra_args, min_gpus, trust_remote_code = config
        url = DEFAULT_URL_FOR_TEST

        # Need min_gpus for server + 1 extra GPU for test model instantiation
        required_gpus = min_gpus + 1
        if torch.cuda.device_count() < required_gpus:
            pytest.skip(
                f"Need at least {required_gpus} GPUs (server={min_gpus} + test=1), have {torch.cuda.device_count()}"
            )

        # The test model will be instantiated on GPU after the server's GPUs
        test_gpu_id = min_gpus  # e.g., if server uses 0-1, test uses 2

        # Build server args
        other_args = [
            "--tp-size",
            str(tp_size),
            "--remote-instance-weight-loader-start-seed-via-transfer-engine",
        ]
        if trust_remote_code:
            other_args.append("--trust-remote-code")
        other_args.extend(extra_args)

        process = None
        try:
            process = popen_launch_server(
                model_name,
                url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
            )
            server_info = get_server_info(url)

            for rank in range(tp_size):
                verify_model_params_match_for_rank(url, rank, server_info, test_gpu_id)

        finally:
            if process is not None:
                terminate_process(process)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
