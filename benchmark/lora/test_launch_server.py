import argparse
import itertools
import shlex
from unittest.mock import patch

from benchmark.lora.launch_server import launch_server


def test_optional_flags_remain_separate_arguments():
    flags = (
        ("disable_custom_all_reduce", "--disable-custom-all-reduce"),
        ("enable_mscclpp", "--enable-mscclpp"),
        ("enable_torch_symm_mem", "--enable-torch-symm-mem"),
    )
    for enabled in itertools.product((False, True), repeat=len(flags)):
        args = argparse.Namespace(
            base_model_path="base",
            lora_path="adapter",
            num_loras=1,
            base_only=False,
            max_loras_per_batch=8,
            max_running_requests=8,
            lora_backend="csgmv",
            tp_size=1,
            **{name: value for (name, _), value in zip(flags, enabled)},
        )
        with patch("benchmark.lora.launch_server.os.system") as system:
            launch_server(args)

        system.assert_called_once()
        tokens = shlex.split(system.call_args.args[0])
        optional_tokens = [
            token for token in tokens if any(flag in token for _, flag in flags)
        ]
        assert optional_tokens == [
            flag for (_, flag), value in zip(flags, enabled) if value
        ]
