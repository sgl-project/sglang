"""User/runtime boundaries and Qwen head conversion for DeepEP Engram."""

import json
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt import runtime_context
from sglang.srt.mem_cache.engram import _parse_config
from sglang.srt.mem_cache.engram.remote import (
    configure_nccl,
    make_remote_config,
    select_rdma_device,
)
from sglang.srt.mem_cache.storage.mooncake_engram import MooncakeEngramStorage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestEngramRemoteConfig(CustomTestCase):
    def setUp(self):
        self.options = dict(backend="deepep_rdma", assignment="auto", group_size=2)

    def test_runtime_owned_fields_cannot_be_supplied_in_json(self):
        for name, value in (
            ("caller_id", 0),
            ("num_callers", 4),
            ("node_id", "node"),
            ("model_key", "weights"),
            ("master_addr", "host:50051"),
            ("session_id", "old"),
            ("local_dir", "/tmp"),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unknown"):
                _parse_config(json.dumps(dict(self.options, **{name: value})))

    def test_auto_manual_and_invalid_identity(self):
        self.assertEqual(_parse_config(json.dumps(self.options)), self.options)
        for extra in ({"group_id": "g"}, {"shard_id": 0}, {"group_size": True}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                _parse_config(json.dumps(dict(self.options, **extra)))
        manual = dict(self.options, assignment="manual", group_id="g", shard_id=1)
        self.assertEqual(_parse_config(json.dumps(manual)), manual)
        with self.assertRaises(ValueError):
            _parse_config(json.dumps(dict(manual, shard_id=2)))

    def test_rdma_device_mapping_and_validation(self):
        devices = ["mlx5_7", "mlx5_7", "mlx5_5", "mlx5_5"]
        self.assertEqual(select_rdma_device("mlx5_{local_rank}", 2), "mlx5_2")
        self.assertEqual(select_rdma_device("mlx5_7", 2), "mlx5_7")
        self.assertEqual(select_rdma_device(devices, 2), "mlx5_5")
        self.assertEqual(
            _parse_config(json.dumps(dict(self.options, rdma_device=devices)))[
                "rdma_device"
            ],
            devices,
        )
        for device in ([], ["mlx5_0", ""], ["mlx5_0", 1], " ", 1, None):
            with self.subTest(device=device), self.assertRaises(ValueError):
                _parse_config(json.dumps(dict(self.options, rdma_device=device)))
        for rank in (-1, 4):
            with self.subTest(rank=rank), self.assertRaisesRegex(ValueError, "rank"):
                select_rdma_device(devices, rank)

    def test_nccl_preparation_preserves_default_and_local_settings(self):
        with mock.patch.dict(
            os.environ, {"NCCL_IB_HCA": "=existing", "NCCL_CUMEM_ENABLE": "1"}
        ):
            configure_nccl(None, 0)
            configure_nccl(json.dumps(self.options), 0)
            self.assertEqual(os.environ["NCCL_IB_HCA"], "=existing")

    def test_nccl_preparation_does_not_override_disabled_cumem(self):
        with mock.patch.dict(os.environ, {"NCCL_CUMEM_ENABLE": "0"}):
            with self.assertRaisesRegex(ValueError, "CUMEM"):
                configure_nccl(json.dumps(self.options), 0)
            self.assertEqual(os.environ["NCCL_CUMEM_ENABLE"], "0")

    def test_nccl_preparation_rejects_late_device_selection(self):
        options = dict(self.options, rdma_device="mlx5_0")
        with (
            mock.patch.dict(os.environ, {"NCCL_CUMEM_ENABLE": "1"}),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            self.assertRaisesRegex(RuntimeError, "before NCCL"),
        ):
            configure_nccl(json.dumps(options), 0)

    def test_bootstrap_selects_device_before_first_process_group(self):
        from sglang.srt.distributed import bootstrap

        options = dict(self.options, rdma_device=["mlx5_7", "mlx5_7", "mlx5_5"])
        parallel = SimpleNamespace(
            tp_size=4,
            pp_size=1,
            tp_rank=2,
            pp_rank=0,
            dist_timeout=None,
            max_ep_size=None,
        )
        execution = SimpleNamespace(
            moe=SimpleNamespace(
                is_ep_joiner=False, is_ep_scale_joiner=False, moe_a2a_backend=None
            ),
            comm=SimpleNamespace(enable_symm_mem=False),
        )
        args = SimpleNamespace(
            enable_hi_engram=True, hi_engram_config=json.dumps(options)
        )

        def init_group(**kwargs):
            self.assertEqual(kwargs["local_rank"], 2)
            self.assertEqual(os.environ["NCCL_IB_HCA"], "=mlx5_5")

        with (
            mock.patch.dict(
                os.environ, {"NCCL_CUMEM_ENABLE": "1", "NCCL_IB_HCA": "=existing"}
            ),
            mock.patch("torch.distributed.is_initialized", return_value=False),
            mock.patch.object(bootstrap, "get_parallel", return_value=parallel),
            mock.patch.object(bootstrap, "get_exec", return_value=execution),
            mock.patch.object(
                bootstrap, "init_distributed_environment", side_effect=init_group
            ) as initialize,
            mock.patch.object(bootstrap, "initialize_model_parallel"),
            mock.patch.object(bootstrap, "_tag_groups_for_flashinfer_allreduce_only"),
            mock.patch.object(bootstrap, "initialize_dp_attention"),
            mock.patch.object(bootstrap, "is_npu", return_value=False),
        ):
            bootstrap._init_parallel_groups(
                backend="nccl", dist_init_method="env://", server_args=args, gpu_id=2
            )
            initialize.assert_called_once()
            args.enable_hi_engram = False
            with mock.patch(
                "sglang.srt.mem_cache.engram.remote.configure_nccl"
            ) as prepare:
                bootstrap._init_parallel_groups(
                    backend="nccl",
                    dist_init_method="env://",
                    server_args=args,
                    gpu_id=2,
                )
                prepare.assert_not_called()

    def test_callers_identity_and_capacity_are_derived_from_runtime(self):
        module = ModuleType("mooncake.engram")
        module.EngramStoreConfig = lambda **kwargs: SimpleNamespace(**kwargs)
        parallel = SimpleNamespace(
            pp_size=1,
            dp_size=1,
            launch_world_size=8,
            nnodes=2,
            tp_rank=5,
            attn_dp_size=1,
        )
        graph = SimpleNamespace(
            decode=SimpleNamespace(max_bs=16), prefill=SimpleNamespace(max_bs=2048)
        )
        patches = dict(
            get_parallel=parallel,
            get_schedule=SimpleNamespace(
                chunked_prefill_size=1024,
                max_prefill_tokens=16384,
                max_running_requests=8,
                enable_mixed_chunk=False,
            ),
            get_model=SimpleNamespace(
                model_path="checkpoint", revision="r1", context_length=4096
            ),
            get_device=SimpleNamespace(base_gpu_id=0, gpu_id_step=1),
            get_exec=SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=graph)),
            get_spec=SimpleNamespace(speculative_num_draft_tokens=5),
            process_model_config=SimpleNamespace(context_len=131072),
        )
        with (
            mock.patch.dict(sys.modules, {"mooncake.engram": module}),
            mock.patch.dict(os.environ, {"NCCL_CUMEM_ENABLE": "1"}),
        ):
            for name, value in patches.items():
                self.enterContext(
                    mock.patch.object(runtime_context, name, return_value=value)
                )
            options = dict(self.options)
            options.pop("backend")
            first = make_remote_config(options, 24)
            self.assertEqual((first.caller_id, first.num_callers), (1, 4))
            self.assertEqual(first.max_lookup_rows, 2048 * 24)
            parallel.tp_rank = 6
            second = make_remote_config(options, 24)
            self.assertEqual(second.caller_id, 2)
            self.assertEqual(first.node_id, second.node_id)
            with mock.patch("torch.cuda.current_device", return_value=2):
                selected = make_remote_config(
                    dict(options, rdma_device=["mlx5_7", "mlx5_7", "mlx5_5", "mlx5_5"]),
                    24,
                )
            self.assertEqual(selected.rdma_device, "mlx5_5")
            with self.assertRaisesRegex(ValueError, "cover"):
                make_remote_config(dict(options, max_lookup_rows=1), 24)
            patches["get_schedule"].chunked_prefill_size = -1
            self.assertEqual(
                make_remote_config(options, 24).max_lookup_rows, 131072 * 24
            )
            patches["get_schedule"].chunked_prefill_size = 2048
            patches["get_schedule"].enable_mixed_chunk = True
            self.assertEqual(
                make_remote_config(options, 24).max_lookup_rows, (2048 + 16 * 5) * 24
            )
            parallel.dp_size = 2
            with self.assertRaisesRegex(ValueError, "one model replica"):
                make_remote_config(options, 24)

    def test_qwen_conversion_supplies_only_node_owned_heads(self):
        stores = []

        class Store:
            shard_id, group_size, group_id = 1, 2, "g"

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                stores.append(self)

            def populate_and_sync(self, local_tables, *, table_owners):
                self.local, self.owners = local_tables, table_owners

            def close(self):
                pass

        module = ModuleType("mooncake.engram")
        module.EngramStore = Store
        with (
            mock.patch.dict(sys.modules, {"mooncake.engram": module}),
            mock.patch(
                "sglang.srt.mem_cache.engram.remote.make_remote_config",
                return_value=SimpleNamespace(group_size=2),
            ),
        ):
            storage = MooncakeEngramStorage(
                remote=self.options,
                head_bounds=[0, 4, 10, 15, 21],
                namespace="layer0",
                row_bytes=8,
            )
            self.assertEqual((storage.row_start, storage.row_end), (10, 21))
            rows = torch.arange(88, dtype=torch.uint8).reshape(11, 8)
            storage.initialize(rows)
            self.assertEqual(set(stores[0].local), {"head2", "head3"})
            torch.testing.assert_close(stores[0].local["head2"], rows[:5])
            torch.testing.assert_close(stores[0].local["head3"], rows[5:])
            self.assertEqual(
                stores[0].owners, {"head0": 0, "head1": 0, "head2": 1, "head3": 1}
            )
            storage.close()


if __name__ == "__main__":
    unittest.main()
