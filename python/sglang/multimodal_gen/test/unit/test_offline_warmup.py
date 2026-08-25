"""Synthetic warmup on entrypoints that host no HTTP startup request.

`--warmup-mode server` used to run no warmup at all offline, so the first real
request paid the full `torch.compile` cost.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.server_warmup import (
    run_sync_client_warmup,
    should_run_client_warmup,
)


def _make_server_args(
    *, warmup_mode: str, warmup_resolutions: list[str] | None
) -> MagicMock:
    server_args = MagicMock()
    server_args.warmup_mode = warmup_mode
    server_args.warmup_resolutions = warmup_resolutions
    server_args.warmup_steps = 1
    server_args.enable_cfg_parallel = False
    server_args.enable_torch_compile = False
    server_args.enable_breakable_cuda_graph = False
    server_args.backend = "native"
    server_args.pipeline_class_name = None
    server_args.is_arg_explicitly_set.return_value = False
    server_args.pipeline_config = SimpleNamespace(
        task_type=ModelTaskType.T2I,
        vae_stride=None,
        vae_scale_factor=None,
        vae_config=None,
    )
    return server_args


def _not_realtime():
    return patch(
        "sglang.multimodal_gen.runtime.server_warmup.is_realtime_serving",
        return_value=False,
    )


class TestClientWarmupSelection(unittest.TestCase):
    def test_server_mode_warms_up_on_the_client(self):
        # `_adjust_warmup` resolves --enable-torch-compile to this mode, and
        # offline runs have no HTTP startup task to execute it.
        server_args = _make_server_args(warmup_mode="server", warmup_resolutions=None)
        with _not_realtime():
            self.assertTrue(should_run_client_warmup(server_args))

    def test_request_mode_defers_to_the_scheduler(self):
        # The scheduler clones the first real request instead. Warming up here
        # too would run two warmups for one request.
        server_args = _make_server_args(warmup_mode="request", warmup_resolutions=None)
        with _not_realtime():
            self.assertFalse(should_run_client_warmup(server_args))


class TestSyncClientWarmupFailurePolicy(unittest.TestCase):
    def test_fail_open_only_swallows_failures_when_requested(self):
        server_args = _make_server_args(warmup_mode="server", warmup_resolutions=None)
        forward = MagicMock(side_effect=RuntimeError("warmup forward exploded"))

        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=SamplingParams(width=512, height=512),
        ):
            run_sync_client_warmup(server_args, forward, fail_open=True)
            with self.assertRaises(RuntimeError):
                run_sync_client_warmup(server_args, forward, fail_open=False)


if __name__ == "__main__":
    unittest.main()
