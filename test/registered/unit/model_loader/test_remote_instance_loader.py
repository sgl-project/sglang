"""Unit tests for RemoteInstanceModelLoader construction - no server, no weights."""

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

import sglang.srt.connector.remote_instance as connector_mod
import sglang.srt.model_executor.model_runner_components.weight_exporter as exporter_mod
import sglang.srt.model_loader.loader as loader_mod
import sglang.srt.model_loader.remote_instance_weight_loader_utils as rfork_utils
import sglang.srt.utils.device as device_mod
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

EXTRA_CONFIG = {"enable_multithread_load": True, "num_threads": 64}

# ServerArgs forwards the backend as a plain str, so every case covers both that
# and the enum member; the loader's comparison holds only via the str mixin.
MODELEXPRESS_BACKENDS = (
    RemoteInstanceWeightLoaderBackend.MODELEXPRESS,
    "modelexpress",
)
OTHER_BACKENDS = (
    RemoteInstanceWeightLoaderBackend.NCCL,
    "nccl",
    RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE,
    "transfer_engine",
)
ALL_BACKENDS = MODELEXPRESS_BACKENDS + OTHER_BACKENDS


def _load_config(backend, extra_config=None):
    return LoadConfig(
        load_format=LoadFormat.REMOTE_INSTANCE,
        model_loader_extra_config=extra_config or {},
        remote_instance_weight_loader_backend=backend,
    )


class TestRemoteInstanceModelLoaderExtraConfig(CustomTestCase):
    def test_modelexpress_backend_accepts_extra_config(self):
        for backend in MODELEXPRESS_BACKENDS:
            with self.subTest(backend=backend):
                loader = loader_mod.RemoteInstanceModelLoader(
                    _load_config(backend, EXTRA_CONFIG)
                )
                # ModelExpress hands the extra config to the DefaultModelLoader
                # it falls back to, so it must survive construction unmodified.
                self.assertEqual(
                    loader.load_config.model_loader_extra_config, EXTRA_CONFIG
                )

    def test_modelexpress_backend_rejects_unknown_extra_config_key(self):
        for backend in MODELEXPRESS_BACKENDS:
            with self.subTest(backend=backend):
                with self.assertRaises(ValueError):
                    loader_mod.RemoteInstanceModelLoader(
                        _load_config(backend, {"num_thread": 64})
                    )

    def test_other_backends_reject_extra_config(self):
        for backend in OTHER_BACKENDS:
            with self.subTest(backend=backend):
                with self.assertRaises(ValueError):
                    loader_mod.RemoteInstanceModelLoader(
                        _load_config(backend, EXTRA_CONFIG)
                    )

    def test_all_backends_construct_without_extra_config(self):
        for backend in ALL_BACKENDS:
            with self.subTest(backend=backend):
                loader = loader_mod.RemoteInstanceModelLoader(_load_config(backend))
                self.assertFalse(loader.load_config.model_loader_extra_config)


def _client_group_kwargs(device_type, *, xpu):
    """The process-group kwargs the client-side connector passes on ``device_type``."""
    connector = connector_mod.RemoteInstanceConnector(
        "tcp://127.0.0.1:29500", torch.device(device_type)
    )
    with (
        patch.object(connector_mod, "is_xpu", return_value=xpu),
        patch.object(connector_mod, "is_npu", return_value=False),
        patch.object(connector_mod, "init_custom_process_group") as init_group,
        patch.object(connector_mod.dist, "barrier"),
    ):
        ok, _ = connector.build_group(gpu_id=1, tp_rank=0, instance_ip="10.0.0.2")
    assert ok
    return init_group.call_args.kwargs


def _seed_group_kwargs(*, xpu):
    """The process-group kwargs the seed-side exporter passes."""
    exporter = exporter_mod.WeightExporter(
        tp_rank=0,
        tp_size=1,
        gpu_id=1,
        get_model_path=lambda: "",
        get_model=lambda: nn.Linear(2, 2),
    )
    with (
        patch.object(exporter_mod, "is_xpu", return_value=xpu),
        patch.object(exporter_mod, "is_npu", return_value=False),
        patch.object(exporter_mod, "init_custom_process_group") as init_group,
        patch.object(exporter_mod.dist, "barrier"),
        patch.object(
            exporter_mod.torch.distributed, "is_initialized", return_value=True
        ),
        patch.object(exporter_mod.current_platform, "empty_cache"),
    ):
        ok, _ = exporter.init_weights_send_group_for_remote_instance(
            master_address="10.0.0.1",
            ports="29500",
            group_rank=0,
            world_size=2,
            group_name="send_weights_10.0.0.2",
        )
    assert ok
    return init_group.call_args.kwargs


class TestRemoteInstanceNcclGroupDevice(CustomTestCase):
    def test_xpu_seed_and_client_build_an_xccl_group(self):
        """Seed and client must agree on the backend, or the group init hangs."""
        client = _client_group_kwargs("xpu", xpu=True)
        seed = _seed_group_kwargs(xpu=True)
        self.assertEqual((client["backend"], seed["backend"]), ("xccl", "xccl"))
        self.assertEqual(client["device_id"], torch.device("xpu", 1))
        self.assertEqual(seed["device_id"], torch.device("xpu", 1))

    def test_cuda_keeps_nccl(self):
        client = _client_group_kwargs("cuda", xpu=False)
        seed = _seed_group_kwargs(xpu=False)
        self.assertEqual((client["backend"], seed["backend"]), ("nccl", "nccl"))
        self.assertEqual(seed["device_id"], torch.device("cuda", 1))


class TestRemoteInstanceTransferEngineRegistration(CustomTestCase):
    def test_xpu_registers_each_weight_without_the_cuda_allocator(self):
        """The merged path reads torch.cuda's allocator snapshot, absent on XPU."""
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2, bias=False))
        engine = MagicMock()
        engine.register_memory.return_value = 0
        with (
            patch.object(device_mod, "is_xpu", return_value=True),
            patch.object(
                torch.cuda.memory,
                "memory_snapshot",
                side_effect=AssertionError("read the CUDA allocator"),
            ),
        ):
            info = rfork_utils.register_memory_region(model, engine)
        registered = {c.args for c in engine.register_memory.call_args_list}
        expected = {
            (p.data_ptr(), p.numel() * p.element_size()) for p in model.parameters()
        }
        self.assertEqual(registered, expected)
        self.assertEqual(set(info), {name for name, _ in model.named_parameters()})


def _seed_exporter(*, tp_size):
    return exporter_mod.WeightExporter(
        tp_rank=0,
        tp_size=tp_size,
        gpu_id=1,
        get_model_path=lambda: "",
        get_model=lambda: nn.Linear(2, 2),
    )


class TestRemoteInstanceSeedRejectsBadRequests(CustomTestCase):
    def test_bad_client_request_is_reported_not_raised(self):
        """A TP2 client against a TP1 seed, or a send before the group exists,
        used to raise in the seed's scheduler and take the seed down."""
        exporter = _seed_exporter(tp_size=1)
        with (
            patch.object(exporter_mod, "init_custom_process_group") as init_group,
            patch.object(
                exporter_mod.torch.distributed, "is_initialized", return_value=True
            ),
        ):
            requests = {
                "init, TP-mismatched ports": lambda: (
                    exporter.init_weights_send_group_for_remote_instance(
                        master_address="10.0.0.1",
                        ports="29500,29501",
                        group_rank=0,
                        world_size=2,
                        group_name="send_weights_10.0.0.2",
                    )
                ),
                "send, TP-mismatched ports": lambda: (
                    exporter.send_weights_to_remote_instance(
                        master_address="10.0.0.1",
                        ports="29500,29501",
                        group_name="send_weights_10.0.0.2",
                    )
                ),
                "send, group never initialized": lambda: (
                    exporter.send_weights_to_remote_instance(
                        master_address="10.0.0.1",
                        ports="29500",
                        group_name="send_weights_10.0.0.2",
                    )
                ),
            }
            for case, request in requests.items():
                with self.subTest(case=case):
                    ok, message = request()
                    self.assertFalse(ok)
                    self.assertTrue(message)
        init_group.assert_not_called()


def _trigger(post):
    with (
        patch.object(rfork_utils.requests, "post", post),
        patch.object(rfork_utils, "_seed_admin_headers", return_value={}),
    ):
        return rfork_utils.trigger_init_weights_send_group_for_remote_instance_request(
            "10.0.0.1", 30000, [29500, 29501], "10.0.0.2"
        )


class TestRemoteInstanceClientAbortsOnSeedRejection(CustomTestCase):
    def test_trigger_reports_a_rejected_or_unreachable_seed(self):
        """The trigger ignored the seed's answer, so the client kept waiting in
        its group rendezvous after the seed had refused."""
        rejected = MagicMock(status_code=400)
        rejected.json.return_value = {"success": False, "message": "Expected 1 ports"}
        ok, message = _trigger(MagicMock(return_value=rejected))
        self.assertFalse(ok)
        self.assertIn("Expected 1 ports", message)

        ok, _ = _trigger(MagicMock(side_effect=ConnectionError("refused")))
        self.assertFalse(ok)

        ok, _ = _trigger(MagicMock(return_value=MagicMock(status_code=200)))
        self.assertTrue(ok)

    def test_rejection_fails_the_client_server(self):
        import sglang.srt.model_executor.model_runner_components.load_model_utils as lmu

        for ok in (False, True):
            with (
                self.subTest(seed_accepts=ok),
                patch.object(
                    lmu,
                    "trigger_init_weights_send_group_for_remote_instance_request",
                    return_value=(ok, "" if ok else "rejected"),
                ),
                patch.object(lmu.psutil, "Process") as process,
            ):
                lmu._init_send_group_or_abort("10.0.0.1", 30000, [29500], "10.0.0.2")
                sent = process.return_value.parent.return_value.send_signal
                if ok:
                    sent.assert_not_called()
                else:
                    sent.assert_called_once_with(lmu.signal.SIGQUIT)


def _resolved_load_format(**kwargs):
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.arg_groups.model_path_hook import handle_load_format
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(model_path="dummy", enable_memory_saver=True, **kwargs)
    handle_load_format(args)
    return resolving_view(args).load_format


_CLIENT = dict(
    load_format="remote_instance",
    remote_instance_weight_loader_seed_instance_ip="10.0.0.1",
    remote_instance_weight_loader_seed_instance_service_port=30000,
)


class TestRemoteInstanceMemorySaver(CustomTestCase):
    def test_transfer_engine_seed_refuses_memory_saver(self):
        """A seed asked to serve transfer_engine clients under memory saver used to
        start without that service, so every client failed later."""
        with self.assertRaisesRegex(
            ValueError, "cannot be used with --enable-memory-saver"
        ):
            _resolved_load_format(
                remote_instance_weight_loader_start_seed_via_transfer_engine=True
            )

    def test_transfer_engine_client_keeps_its_disk_fallback(self):
        """The seed refusal must not catch a client that also passes the seed flag."""
        load_format = _resolved_load_format(
            remote_instance_weight_loader_backend="transfer_engine",
            remote_instance_weight_loader_start_seed_via_transfer_engine=True,
            **_CLIENT,
        )
        self.assertEqual(load_format, "auto")


def _receive_over_xccl(*, xpu, memory_saver):
    """Run the XCCL client receive with a fake broadcast; returns the model and,
    per broadcast, (ran on the loading thread, received into a parameter)."""
    import threading

    model = nn.Linear(4, 4)
    param_ptrs = {p.data.data_ptr() for p in model.parameters()}
    calls = []

    def broadcast(tensor, src, group):
        calls.append(
            (
                threading.current_thread() is threading.main_thread(),
                tensor.data_ptr() in param_ptrs,
            )
        )
        tensor.fill_(7.0)

    loader = loader_mod.RemoteInstanceModelLoader(_load_config("nccl"))
    dist = loader_mod.torch.distributed
    with (
        patch.object(loader_mod, "is_xpu", return_value=xpu),
        patch.object(loader_mod, "get_exec") as get_exec,
        patch.object(loader_mod.torch.xpu, "set_device"),
        patch.object(dist, "broadcast", side_effect=broadcast),
        patch.object(dist.distributed_c10d, "destroy_process_group"),
        patch.object(loader_mod, "trigger_transferring_weights_request"),
        patch.object(loader_mod.current_platform, "synchronize"),
        patch.object(loader_mod.current_platform, "empty_cache"),
        patch.object(loader_mod, "post_load_weights"),
    ):
        get_exec.return_value.features.enable_memory_saver = memory_saver
        loader.load_model_from_remote_instance_by_nccl(
            model,
            MagicMock(),
            MagicMock(dtype=torch.float32),
            MagicMock(gpu_id=0),
        )
    return model, calls


class TestRemoteInstanceXcclMemorySaver(CustomTestCase):
    def test_xpu_memory_saver_receives_off_thread_into_staging(self):
        """Under memory saver on XPU the weights are Level Zero virtual-memory
        mappings, which oneCCL read on the CPU (SIGSEGV): the receive must leave
        the loading thread, whose allocations go to the memory-saver pool, and
        must not target a parameter directly."""
        model, calls = _receive_over_xccl(xpu=True, memory_saver=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(set(calls), {(False, False)})
        for param in model.parameters():
            self.assertTrue(bool((param.data == 7.0).all()))

    def test_other_configs_receive_into_the_parameters(self):
        for xpu, memory_saver in ((True, False), (False, True), (False, False)):
            with self.subTest(xpu=xpu, memory_saver=memory_saver):
                _, calls = _receive_over_xccl(xpu=xpu, memory_saver=memory_saver)
                self.assertEqual(set(calls), {(True, True)})


if __name__ == "__main__":
    unittest.main()
