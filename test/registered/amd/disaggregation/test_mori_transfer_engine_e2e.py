import os
import struct
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import requests

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import build_dcp_token_transfer_plan
from sglang.srt.disaggregation.mori.conn import (
    MORI_DCP_GUARD,
    KVArgsRegisterInfo,
    MoriKVManager,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
)

register_amd_ci(est_time=900, suite="stage-b-test-large-8-gpu-mi35x-disaggregation-amd")


class TestMoriDCPTransfer(unittest.TestCase):
    def test_register_info_parses_dcp_geometry(self):
        kv_descs = [object(), object()]
        payload = [
            b"None",
            b"10.0.0.2",
            b"23456",
            b"engine",
            b"kv",
            b"aux",
            b"state",
            b"3",
            b"4",
            b"1",
            b"1024",
            b"",
            b"",
            struct.pack("2Q", 1024, 2048),
            struct.pack("2I", 2, 7),
            b"4",
            b"3",
        ]
        fake_engine_desc = SimpleNamespace(key="engine")

        with (
            patch(
                "sglang.srt.disaggregation.mori.conn.EngineDesc.unpack",
                return_value=fake_engine_desc,
            ),
            patch(
                "sglang.srt.disaggregation.mori.conn._unpack_mem_desc_list",
                side_effect=[kv_descs, []],
            ),
            patch(
                "sglang.srt.disaggregation.mori.conn._unpack_mem_desc_lists",
                return_value=[],
            ),
        ):
            info = KVArgsRegisterInfo.from_zmq(payload)

        self.assertEqual(info.dst_kv_item_lens, [1024, 2048])
        self.assertEqual(info.dst_kv_layer_ids, [2, 7])
        self.assertEqual(info.dst_dcp_size, 4)
        self.assertEqual(info.dst_dcp_rank, 3)

    def test_register_info_defaults_for_legacy_peer(self):
        kv_descs = [object(), object()]
        payload = [
            b"None",
            b"10.0.0.2",
            b"23456",
            b"engine",
            b"kv",
            b"aux",
            b"state",
            b"0",
            b"1",
            b"0",
            b"256",
        ]

        with (
            patch(
                "sglang.srt.disaggregation.mori.conn.EngineDesc.unpack",
                return_value=SimpleNamespace(key="engine"),
            ),
            patch(
                "sglang.srt.disaggregation.mori.conn._unpack_mem_desc_list",
                side_effect=[kv_descs, []],
            ),
            patch(
                "sglang.srt.disaggregation.mori.conn._unpack_mem_desc_lists",
                return_value=[],
            ),
        ):
            info = KVArgsRegisterInfo.from_zmq(payload)

        self.assertEqual(info.dst_kv_item_lens, [256, 256])
        self.assertEqual(info.dst_kv_layer_ids, [])
        self.assertEqual(info.dst_dcp_size, 1)
        self.assertEqual(info.dst_dcp_rank, 0)

    def test_send_kvcache_dcp_uses_token_byte_offsets(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.kv_mem_descs = ["src0", "src1"]
        manager._submit_batch_transfer_plan = MagicMock(
            side_effect=[["status0"], ["status1"]]
        )
        peer_info = SimpleNamespace(
            dst_kv_mem_descs=["dst0", "dst1"],
            dcp_dst_region_indices=[1, 0],
            dcp_token_item_lens=[8, 16],
        )
        plan = build_dcp_token_transfer_plan(
            np.array([2], dtype=np.int32),
            np.array([7], dtype=np.int32),
            physical_page_size=4,
            dcp_size=2,
            dcp_rank=0,
            num_kv_tokens=4,
        )

        statuses = manager.send_kvcache_dcp(peer_info, plan)

        self.assertEqual(statuses, ["status0", "status1"])
        first_plan = manager._submit_batch_transfer_plan.call_args_list[0].args[2]
        self.assertEqual(first_plan.local_offsets, [64, 80])
        self.assertEqual(first_plan.remote_offsets, [224, 232])
        self.assertEqual(first_plan.sizes, [8, 8])
        second_plan = manager._submit_batch_transfer_plan.call_args_list[1].args[2]
        self.assertEqual(second_plan.local_offsets, [128, 160])
        self.assertEqual(second_plan.remote_offsets, [448, 464])
        self.assertEqual(second_plan.sizes, [16, 16])

    def test_add_remote_peer_resolves_dcp_destination_layers(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.decode_kv_args_table = {}
        manager.engine = SimpleNamespace(register_remote_engine=MagicMock())
        manager.dcp_size = 1
        manager.dcp_rank = 0
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.kv_mem_descs = ["src2", "src7"]
        manager.kv_args = SimpleNamespace(
            kv_layer_ids=[2, 7],
            kv_item_lens=[512, 1024],
            page_size=64,
            num_draft_entries=0,
        )
        peer_info = KVArgsRegisterInfo(
            endpoint="10.0.0.2",
            dst_port=23456,
            engine_desc=SimpleNamespace(key="peer"),
            dst_kv_mem_descs=["dst7", "dst2"],
            dst_aux_mem_descs=[],
            dst_state_mem_descs=[],
            gpu_id=3,
            decode_tp_size=4,
            decode_tp_rank=3,
            dst_kv_item_len=1024,
            dst_state_item_lens=[],
            dst_state_dim_per_tensor=[],
            dst_kv_item_lens=[1024, 512],
            dst_kv_layer_ids=[7, 2],
            dst_dcp_size=4,
            dst_dcp_rank=3,
        )

        manager._add_remote_peer(peer_info)

        self.assertTrue(peer_info.requires_dcp_relayout)
        self.assertEqual(peer_info.dcp_dst_region_indices, [1, 0])
        self.assertEqual(peer_info.dcp_token_item_lens, [8, 16])
        self.assertIs(manager.decode_kv_args_table["peer"], peer_info)
        manager.engine.register_remote_engine.assert_called_once_with(
            peer_info.engine_desc
        )

    def test_add_remote_peer_slices_regular_mla_destination_for_pp(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.decode_kv_args_table = {}
        manager.engine = SimpleNamespace(register_remote_engine=MagicMock())
        manager.dcp_size = 1
        manager.dcp_rank = 0
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.kv_mem_descs = ["src2", "src3"]
        manager.kv_args = SimpleNamespace(
            kv_layer_ids=[],
            kv_item_lens=[512, 512],
            page_size=64,
            prefill_start_layer=2,
            mla_compression_ratios=None,
            num_draft_entries=0,
        )
        peer_info = KVArgsRegisterInfo(
            endpoint="10.0.0.2",
            dst_port=23456,
            engine_desc=SimpleNamespace(key="peer"),
            dst_kv_mem_descs=["dst0", "dst1", "dst2", "dst3"],
            dst_aux_mem_descs=[],
            dst_state_mem_descs=[],
            gpu_id=3,
            decode_tp_size=4,
            decode_tp_rank=3,
            dst_kv_item_len=512,
            dst_state_item_lens=[],
            dst_state_dim_per_tensor=[],
            dst_kv_item_lens=[512, 512, 512, 512],
            dst_dcp_size=4,
            dst_dcp_rank=3,
        )

        manager._add_remote_peer(peer_info)

        self.assertEqual(peer_info.dcp_dst_region_indices, [2, 3])
        self.assertEqual(peer_info.dcp_token_item_lens, [8, 8])

    def test_dcp_registration_guard_is_accepted(self):
        manager = MoriKVManager.__new__(MoriKVManager)

        self.assertEqual(
            manager._validate_message([MORI_DCP_GUARD, b"None"]),
            [b"None"],
        )

    def test_submit_routes_nonzero_dcp_chunk_with_full_destination_pages(self):
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.request_status = {23: KVPoll.WaitingForInput}
        manager.transfer_lock = threading.Lock()
        info = SimpleNamespace(
            engine_key="peer",
            is_dummy=False,
            dst_kv_indices=np.array([7], dtype=np.int32),
            decode_prefix_len=0,
            dst_state_indices=[],
            dst_aux_index=-1,
        )
        peer_info = SimpleNamespace(
            requires_dcp_relayout=True,
            dst_dcp_size=2,
            dst_dcp_rank=0,
        )
        manager.transfer_infos = {23: {"peer": info}}
        manager.decode_kv_args_table = {"peer": peer_info}
        manager.kv_args = SimpleNamespace(page_size=4)
        manager.state_mem_descs = []
        manager.update_status = MagicMock()
        manager.send_kvcache_dcp = MagicMock(return_value=["status"])

        statuses = manager._submit_kv_transfer(
            23,
            np.array([3], dtype=np.int32),
            slice(1, 2),
            False,
            num_kv_tokens=4,
        )

        self.assertEqual(statuses, ["status"])
        plan = manager.send_kvcache_dcp.call_args.args[1]
        np.testing.assert_array_equal(plan.target_src_token_indices, [12, 14])
        np.testing.assert_array_equal(plan.target_dst_token_indices, [30, 31])


class MoriTransferEngineBase(PDDisaggregationServerBase):
    port_delta = 0
    prefill_tp = 1
    decode_tp = 1
    decode_base_gpu_id = 1
    required_gpus = 2

    # Subclasses can override to pick a different model or pass extra args.
    model_default = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    model_env_var = "SGLANG_MORI_E2E_TEST_MODEL"
    extra_prefill_args: list = []
    extra_decode_args: list = []

    @classmethod
    def setUpClass(cls):
        try:
            import torch

            if not torch.cuda.is_available():
                raise unittest.SkipTest("torch.cuda is not available.")
            if torch.cuda.device_count() < cls.required_gpus:
                raise unittest.SkipTest(
                    f"MORI PD check requires >= {cls.required_gpus} visible GPUs."
                )
        except Exception as e:
            raise unittest.SkipTest(f"torch is not available/usable: {e}")

        super().setUpClass()

        cls._old_use_aiter = os.environ.get("SGLANG_USE_AITER")
        os.environ["SGLANG_USE_AITER"] = "1"

        # The shared fixture defaults to Mooncake in CI; pin Mori explicitly here.
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mori"]

        rdma_env = os.environ.get("SGLANG_TEST_RDMA_DEVICE")
        if rdma_env:
            cls.rdma_devices = ["--disaggregation-ib-device", rdma_env]
            print(f"Found RDMA devices in env: {rdma_env}")
        else:
            print("SGLANG_TEST_RDMA_DEVICE is not set! Running without RDMA.")
            cls.rdma_devices = []

        cls._shift_ports()
        cls.model = try_cached_model(
            os.environ.get(cls.model_env_var, cls.model_default)
        )

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(
            cls.prefill_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process_prefill,
        )
        cls.wait_server_ready(
            cls.decode_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process_decode,
        )
        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_old_use_aiter", None) is None:
            os.environ.pop("SGLANG_USE_AITER", None)
        else:
            os.environ["SGLANG_USE_AITER"] = cls._old_use_aiter
        super().tearDownClass()

    @classmethod
    def _shift_ports(cls):
        if cls.port_delta == 0:
            return

        cls.lb_port = str(int(cls.lb_port) + cls.port_delta)
        cls.prefill_port = str(int(cls.prefill_port) + cls.port_delta)
        cls.decode_port = str(int(cls.decode_port) + cls.port_delta)
        cls.bootstrap_port = str(int(cls.bootstrap_port) + cls.port_delta)
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        cls.base_url = cls.lb_url

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            str(cls.prefill_tp),
            "--attention-backend",
            "aiter",
        ] + list(cls.extra_prefill_args)
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            str(cls.decode_tp),
            "--base-gpu-id",
            str(cls.decode_base_gpu_id),
            "--attention-backend",
            "aiter",
        ] + list(cls.extra_decode_args)
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def _assert_generate_smoke(self):
        resp = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        out = resp.json()
        self.assertIn("text", out)
        self.assertIsInstance(out["text"], str)
        self.assertGreater(len(out["text"]), 0)


class TestMoriTransferEngineE2E(MoriTransferEngineBase):
    def test_generate_smoke(self):
        self._assert_generate_smoke()


class TestMoriTransferEngineTPMismatchE2E(MoriTransferEngineBase):
    port_delta = 10
    prefill_tp = 2
    decode_tp = 4
    decode_base_gpu_id = 2
    required_gpus = 6

    def test_generate_smoke_tp_mismatch(self):
        self._assert_generate_smoke()


if __name__ == "__main__":
    unittest.main()
