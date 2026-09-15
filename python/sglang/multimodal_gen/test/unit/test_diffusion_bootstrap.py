# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.distributed import bootstrap
from sglang.srt.utils.network import NetworkAddress


class TestDiffusionBootstrap(unittest.TestCase):
    def test_worker_and_daemon_share_globals_groups_and_rendezvous(self):
        for role in ("diffusion_gpu_worker", "diffusion_weight_cache_daemon"):
            with self.subTest(role=role):
                args = SimpleNamespace(
                    num_gpus=4,
                    nnodes=2,
                    tp_size=2,
                    cfg_parallel_degree=1,
                    ulysses_degree=1,
                    ring_degree=1,
                    sp_degree=1,
                    dp_size=2,
                    dist_timeout=99,
                )
                context = SimpleNamespace(_server_args=None)
                events = []
                with (
                    patch.dict(os.environ),
                    patch.object(
                        bootstrap.current_platform, "is_mps", return_value=False
                    ),
                    patch.object(
                        bootstrap.current_platform, "is_cuda", return_value=True
                    ),
                    patch.object(
                        bootstrap.current_platform, "get_device", return_value="cuda:3"
                    ),
                    patch.object(
                        bootstrap.current_platform,
                        "set_device",
                        side_effect=lambda device: events.append("device"),
                    ),
                    patch.object(
                        bootstrap,
                        "set_global_server_args",
                        side_effect=lambda value: events.append("globals"),
                    ) as set_args,
                    patch.object(
                        bootstrap, "worker_cpu_intra_op_threads", return_value=8
                    ) as cpu_budget,
                    patch.object(bootstrap.torch, "set_num_threads") as threads,
                    patch.object(bootstrap, "configure_persistent_torch_compile_cache"),
                    patch.object(
                        bootstrap,
                        "maybe_init_distributed_environment_and_model_parallel",
                        side_effect=lambda **kwargs: events.append("groups"),
                    ) as groups,
                    patch(
                        "sglang.srt.runtime_context.get_context", return_value=context
                    ),
                    patch(
                        "sglang.srt.runtime_context.publish",
                        side_effect=lambda *args, **kwargs: events.append("publish"),
                    ) as publish,
                    patch("sglang.srt.server_args.ServerArgs", return_value=object()),
                    patch(
                        "sglang.srt.utils.patch_torch.monkey_patch_torch_reductions"
                    ) as reductions,
                ):
                    bootstrap.bootstrap_diffusion_runtime(
                        args,
                        local_rank=3,
                        rank=2,
                        rendezvous=NetworkAddress("127.0.0.1", 19437),
                        role=role,
                    )
                    self.assertEqual(events, ["device", "globals", "groups", "publish"])
                    self.assertEqual(os.environ["MASTER_PORT"], "19437")
                    self.assertEqual(os.environ["LOCAL_RANK"], "3")
                    self.assertEqual(os.environ["RANK"], "2")
                    self.assertEqual(os.environ["WORLD_SIZE"], "4")
                    set_args.assert_called_once_with(args)
                    cpu_budget.assert_called_once_with(2)
                    threads.assert_called_once_with(8)
                    self.assertEqual(
                        groups.call_args.kwargs["distributed_init_method"],
                        "tcp://127.0.0.1:19437",
                    )
                    self.assertEqual(groups.call_args.kwargs["tp_size"], 2)
                    self.assertEqual(publish.call_args.kwargs["role"], role)
                    reductions.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
