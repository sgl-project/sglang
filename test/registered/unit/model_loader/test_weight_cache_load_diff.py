"""The weight cache serves a load-diff snapshot: the daemon describes its
loaded model, an engine checks the response before mapping anything, builds
the model on the meta device and installs the snapshot. Both sides run here
on CPU, over a socket pair, with a transport that copies the bytes."""

import contextlib
import json
import socket
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.eplb.expert_location import (
    ModelConfigForExpertLocation,
    compute_initial_expert_location_metadata,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.runtime_context import get_context, get_parallel, get_resources
from sglang.srt.weight_cache import daemon as daemon_module
from sglang.srt.weight_cache import ipc_loader, load_diff, load_diff_format
from sglang.srt.weight_cache.daemon import WeightCacheDaemon
from sglang.srt.weight_cache.protocol import (
    PROTOCOL_VERSION,
    ProtocolVersionError,
    handshake_with_daemon,
    recv_msg,
    send_msg,
)
from sglang.srt.weight_cache.transport import (
    TORCH_IPC_BACKEND,
    TorchIpcTransportBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4), requires_grad=False)
        self.register_buffer("table", torch.zeros(4), persistent=False)
        self.w_kc = None


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = Layer(), Layer()
        self.b.table = self.a.table  # one cached buffer under two modules

    def forward(self, x):
        return (x @ self.a.weight + self.a.table) @ self.a.w_kc + self.b.table


def load(model):
    torch.manual_seed(0)
    with torch.no_grad():
        model.a.weight.copy_(torch.randn(4, 4))
        model.a.table.copy_(torch.arange(4.0))
    model.a.w_kc = model.a.weight.t().contiguous()


class CopyingDaemonTransport(TorchIpcTransportBackend):
    """Exports by copying each storage, as vmm_fd does; entries carry no handle."""

    def prepare_export(self, state_tensors):
        self.storages = {name: t.clone() for name, (t, _) in state_tensors.items()}
        state_tensors.clear()
        return {
            name: {"shape": [t.numel()], "dtype": "uint8", "name": name}
            for name, t in self.storages.items()
        }


class CopyingClientTransport(TorchIpcTransportBackend):
    def __init__(self, storages):
        self.storages = storages
        self.closed = False

    def import_tensor(self, entry):
        return self.storages[entry["name"]].clone()

    def close(self):
        self.closed = True


def daemon_with_snapshot(device=torch.device("cpu"), model=Model, loading=load):
    owner = model()
    constructed = load_diff.record_construction(owner)
    loading(owner)
    daemon = WeightCacheDaemon.__new__(WeightCacheDaemon)
    daemon.gpu_id, daemon.tp_rank = 0, 0
    daemon.model = owner
    daemon.manifest = None
    daemon.state_entries = {}
    daemon.preloaded_weights_bytes = 0
    daemon.transport_name = TORCH_IPC_BACKEND
    daemon.transport_backend = CopyingDaemonTransport()
    daemon.config = mock.Mock(matches=lambda other: True, to_dict=lambda: {})
    with on(device):
        daemon._export_state(daemon._snapshot(constructed))
    return owner, daemon


# The engine's fingerprint: the daemon's (a Mock giving {}) matches it.
ENGINE = SimpleNamespace(to_dict=lambda: {})


def on(device):
    """The daemon's device (the CPU here, so that the transport moves the
    storages; another one, so that they travel in the response)."""
    return mock.patch.object(
        daemon_module.current_platform, "get_device", return_value=device
    )


def fetch(daemon, version=PROTOCOL_VERSION, handshake=True):
    client, server = socket.socketpair()
    with client, server:
        client.settimeout(5)
        server.settimeout(5)
        with mock.patch("sglang.srt.weight_cache.daemon.CacheConfig.from_dict"):
            thread = threading.Thread(target=daemon._handle_connection, args=(server,))
            thread.start()
            if handshake:
                handshake_with_daemon(client)
            request = {"type": "fetch_state", "config": {}}
            if version is not None:
                request["protocol_version"] = version
            send_msg(client, request)
            response = recv_msg(client)
            thread.join(5)
    return response


def loader_for(daemon):
    loader = ipc_loader.WeightCacheModelLoader(
        load_config=LoadConfig(load_format=LoadFormat.IPC_CACHE)
    )
    loader._transport_backend = CopyingClientTransport(
        daemon.transport_backend.storages
    )
    return loader


def install(loader, response, model=Model):
    with torch.device("meta"):
        reader = model()
    with mock.patch.object(ipc_loader, "_initialize_model", return_value=reader):
        return loader._install(
            SimpleNamespace(dtype=torch.float32),
            SimpleNamespace(device_type="cpu", gpu_id=-1),
            response,
            None,
        )


class TestServeSnapshot(CustomTestCase):
    def test_engine_serves_what_the_daemon_loaded(self):
        owner, daemon = daemon_with_snapshot()
        response = fetch(daemon)
        loader = loader_for(daemon)
        loader._check_response(response, ENGINE)
        model = install(loader, response)
        try:
            x = torch.ones(2, 4)
            torch.testing.assert_close(model(x), owner(x))
            self.assertIs(model.b.table, model.a.table)
            self.assertIn(
                loader._transport_backend, ipc_loader._LOADED_MODEL_TRANSPORTS
            )
            # A parameter outliving the model stays usable: nothing unmaps it.
            weight = model.a.weight
            del model
            self.assertFalse(loader._transport_backend.closed)
            torch.testing.assert_close(weight, owner.a.weight)
        finally:
            ipc_loader._LOADED_MODEL_TRANSPORTS.remove(loader._transport_backend)

    def test_response_carries_the_manifest_and_the_version(self):
        _, daemon = daemon_with_snapshot()
        response = fetch(daemon)
        self.assertEqual(response["protocol_version"], PROTOCOL_VERSION)
        manifest = response["manifest"]
        load_diff_format.decode_manifest(manifest["load_diff"])
        self.assertEqual(set(manifest["storages"]), set(response["entries"]))
        self.assertEqual(manifest["host"], {})

    def test_cpu_storages_travel_in_the_response(self):
        owner, daemon = daemon_with_snapshot(device=torch.device("cuda", 0))
        response = fetch(daemon)
        self.assertEqual(response["entries"], {})
        self.assertEqual(
            set(response["manifest"]["host"]), set(daemon.manifest["storages"])
        )
        loader = loader_for(daemon)
        loader._check_response(response, ENGINE)
        model = install(loader, response)
        try:
            x = torch.ones(2, 4)
            torch.testing.assert_close(model(x), owner(x))
        finally:
            ipc_loader._LOADED_MODEL_TRANSPORTS.remove(loader._transport_backend)

    def load(self, loader, response):
        with (
            mock.patch.object(loader, "_fetch_from_cache", return_value=response),
            mock.patch.object(loader, "_resolve_engine_quant", return_value=("", None)),
            mock.patch(
                "sglang.srt.model_loader.loader._get_quantization_config",
                return_value=None,
            ),
        ):
            with torch.device("meta"):
                reader = Model()
            with mock.patch.object(
                ipc_loader, "_initialize_model", return_value=reader
            ):
                loader.load_model(
                    model_config=SimpleNamespace(dtype=torch.float32),
                    device_config=None,
                )

    def test_a_failed_install_keeps_the_mappings(self):
        _, daemon = daemon_with_snapshot()
        response = fetch(daemon)
        next(iter(daemon.transport_backend.storages.values())).view(-1)[0] ^= 1
        loader = loader_for(daemon)
        with self.assertRaises(load_diff.SnapshotError):
            self.load(loader, response)
        try:
            self.assertFalse(loader._transport_backend.closed)
            self.assertIn(
                loader._transport_backend, ipc_loader._LOADED_MODEL_TRANSPORTS
            )
        finally:
            ipc_loader._LOADED_MODEL_TRANSPORTS.remove(loader._transport_backend)

    def test_a_failed_import_closes_the_transport(self):
        _, daemon = daemon_with_snapshot()
        response = fetch(daemon)
        loader = loader_for(daemon)
        with mock.patch.object(
            loader._transport_backend,
            "import_tensor",
            side_effect=RuntimeError("no mapping"),
        ):
            with self.assertRaisesRegex(RuntimeError, "no mapping"):
                self.load(loader, response)
        self.assertTrue(loader._transport_backend.closed)
        self.assertNotIn(loader._transport_backend, ipc_loader._LOADED_MODEL_TRANSPORTS)

    def test_a_failure_before_the_install_closes_the_transport(self):
        _, daemon = daemon_with_snapshot()
        response = {**fetch(daemon), "preloaded_weights_bytes": -1}
        loader = loader_for(daemon)
        with self.assertRaisesRegex(RuntimeError, "weight-memory"):
            self.load(loader, response)
        self.assertTrue(loader._transport_backend.closed)
        self.assertNotIn(loader._transport_backend, ipc_loader._LOADED_MODEL_TRANSPORTS)


# Initial expert layouts (--init-expert-location) of two MoE layers of four
# experts: the logical expert in each physical slot.
LAYOUT_A = [[3, 2, 1, 0], [1, 0, 3, 2]]
LAYOUT_B = [[0, 1, 3, 2], [2, 3, 0, 1]]


class Experts(nn.Module):
    """Stands in for a model's MoE layers: row l holds, for each physical
    slot of layer l, the id of the expert whose weights are there."""

    def __init__(self):
        super().__init__()
        self.rows = nn.Parameter(torch.full((2, 4), -1.0), requires_grad=False)


def load_experts(model):
    """Place each checkpoint expert as FusedMoE.weight_loader does: in the
    slots the process's expert map gives it, and without a map in the slot of
    its own number."""
    metadata = get_global_expert_location_metadata()
    with torch.no_grad():
        for layer_id in range(2):
            for expert_id in range(4):
                slots = [expert_id]
                if metadata is not None:
                    slots = metadata.logical_to_all_physical(layer_id, expert_id)
                model.rows[layer_id, slots] = float(expert_id)


@contextlib.contextmanager
def process(layout, **settings):
    """An emulated process started with --init-expert-location ``layout`` and
    without --enable-eplb: its published config and EP rank, and no expert
    map yet. The model registry lookup of the layout's shape is patched."""
    with (
        get_context().override_server_args(
            device="cpu",
            moe_a2a_backend="none",
            init_expert_location=json.dumps({"physical_to_logical_map": layout}),
            **settings,
        ),
        get_parallel().override(moe_ep_rank=0),
        get_resources().override(expert_location_metadata=None),
        mock.patch.object(
            ModelConfigForExpertLocation,
            "from_model_config",
            return_value=ModelConfigForExpertLocation(
                num_layers=2, num_logical_experts=4
            ),
        ),
    ):
        yield


def daemon_with_layout(layout):
    """A daemon started with ``layout``: it sets its expert map, loads by it
    and snapshots its model, in the order of WeightCacheDaemon.load."""
    with process(layout):
        daemon = WeightCacheDaemon.__new__(WeightCacheDaemon)
        daemon._initialize_expert_location_metadata(None)
        return daemon_with_snapshot(model=Experts, loading=load_experts)


@contextlib.contextmanager
def engine_with_layout(layout, dispatch="dynamic"):
    """An engine started with ``layout``, up to its load: the runner set its
    expert map (ModelRunner.maybe_init_expert_location_metadata). With a
    layout the argument pipeline resolves --ep-dispatch-algorithm for a real
    model; ``dispatch`` states it here."""
    with process(layout, ep_dispatch_algorithm=dispatch):
        set_global_expert_location_metadata(
            compute_initial_expert_location_metadata(model_config=None, moe_ep_rank=0)
        )
        yield


class TestExpertMap(CustomTestCase):
    """The daemon's expert rows are in the layout of the daemon's expert map,
    which its snapshot carries (weight_cache/expert_map.py)."""

    def process_map(self):
        metadata = get_global_expert_location_metadata()
        return metadata.physical_to_logical_map_cpu.tolist()

    def runs_the_experts_asked_for(self, model):
        """Every token runs in a slot whose installed weights are those of
        the expert the router picked for it."""
        held = model.rows.data.to(torch.int64)
        for layer_id in range(2):
            logical = torch.arange(4).repeat(3, 1)
            physical = topk_ids_logical_to_physical(
                logical, ExpertLocationDispatchInfo.init_new(layer_id)
            )
            self.assertTrue(
                torch.equal(held[layer_id][physical], logical),
                (layer_id, physical.tolist()),
            )

    def test_a_daemon_without_eplb_loads_by_its_layout_and_exports_it(self):
        """An engine sets its expert map from --init-expert-location with or
        without EPLB, and dispatches by it. The daemon used to set its map
        only with --enable-eplb: without it, it loaded every expert into the
        slot of its own number, where the engine looks for another expert."""
        owner, daemon = daemon_with_layout(LAYOUT_A)
        self.assertEqual(owner.rows.to(torch.int64).tolist(), LAYOUT_A)
        manifest = load_diff_format.decode_manifest(daemon.manifest["load_diff"])
        self.assertEqual(manifest.expert_map.rows, LAYOUT_A)

    def test_an_engine_with_another_layout_dispatches_by_the_daemons(self):
        _, daemon = daemon_with_layout(LAYOUT_A)
        cases = ((LAYOUT_B, True), (LAYOUT_A, False))
        for layout, adopts in cases:
            with self.subTest(adopts=adopts), engine_with_layout(layout):
                response = fetch(daemon)
                loader = loader_for(daemon)
                loader._check_response(response, ENGINE)
                logs = self.assertLogs if adopts else self.assertNoLogs
                with logs("sglang.srt.weight_cache.expert_map", "WARNING"):
                    model = install(loader, response, model=Experts)
                try:
                    self.assertEqual(self.process_map(), LAYOUT_A)
                    self.runs_the_experts_asked_for(model)
                finally:
                    ipc_loader._LOADED_MODEL_TRANSPORTS.remove(
                        loader._transport_backend
                    )

    def test_an_engine_that_cannot_follow_the_daemons_layout_is_refused(self):
        # Without a dispatch algorithm its MoE layers run each token in the
        # slot of its expert's number, whatever map the process holds.
        _, daemon = daemon_with_layout(LAYOUT_A)
        with engine_with_layout(LAYOUT_B, dispatch=None):
            response = fetch(daemon)
            loader = loader_for(daemon)
            loader._check_response(response, ENGINE)
            with torch.device("meta"):
                reader = Experts()
            try:
                with mock.patch.object(
                    ipc_loader, "_initialize_model", return_value=reader
                ):
                    with self.assertRaisesRegex(
                        load_diff.SnapshotError, "cannot take the owner's expert map"
                    ):
                        loader._install(
                            SimpleNamespace(dtype=torch.float32),
                            SimpleNamespace(device_type="cpu", gpu_id=-1),
                            response,
                            None,
                        )
                # Nothing was attached, and the engine keeps its own map.
                self.assertTrue(reader.rows.is_meta)
                self.assertEqual(self.process_map(), LAYOUT_B)
            finally:
                ipc_loader._LOADED_MODEL_TRANSPORTS.remove(loader._transport_backend)


class TestKernelScratch(CustomTestCase):
    def test_cutlass_and_marlin_scratch_is_not_shared(self):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        method = Fp8MoEMethod.__new__(Fp8MoEMethod)  # buffers as create_weights makes
        method.ab_strides1 = torch.full((2,), 4)
        method.workspace = torch.empty(16, dtype=torch.uint8)
        method.expert_offsets = torch.empty(3, dtype=torch.int32)
        linear = LinearBase.__new__(LinearBase)
        nn.Module.__init__(linear)
        linear.workspace = torch.zeros(4, dtype=torch.int32)  # Marlin's lock buffer
        model = nn.Module()
        model.experts = nn.Module()
        model.experts.quant_method = method
        model.linear = linear
        found = load_diff_format.walk_model(model)
        method_path = ("experts", ("a", "quant_method"))
        self.assertEqual(
            set(found.local),
            {
                method_path + (("a", "workspace"),),
                method_path + (("a", "expert_offsets"),),
                ("linear", ("a", "workspace")),
            },
        )
        self.assertIn(method_path + (("a", "ab_strides1"),), found.tensors)

    def test_the_vision_cudnn_workspace_is_declared(self):
        from sglang.srt.layers.attention.vision import (
            VisionAttention,
            VisionFlashInferAttention,
        )

        for kind in (VisionAttention, VisionFlashInferAttention):
            self.assertIn(
                "workspace_buffer", load_diff_format.process_local_fields(kind)
            )

    def test_the_fp8_moe_dispatcher_enum_needs_no_aiter_import(self):
        # FP8 MoE post-processing stores an AiterQuantType member in every
        # dispatcher's config; an engine decodes it only if the class's module
        # is imported, and the engine does not post-process.
        from sglang.srt.layers.moe.moe_runner.aiter import AiterQuantType

        self.assertEqual(AiterQuantType.__module__, "sglang.srt.layers.moe.utils")


class TestRefuseBeforeMapping(CustomTestCase):
    def setUp(self):
        _, self.daemon = daemon_with_snapshot()
        self.response = fetch(self.daemon)

    def refused(self, response, error=Exception):
        with self.assertRaises(error):
            ipc_loader.WeightCacheModelLoader._check_response(response, ENGINE)

    def test_another_protocol_version(self):
        self.refused({**self.response, "protocol_version": 1}, ProtocolVersionError)
        self.refused({**self.response, "protocol_version": None}, ProtocolVersionError)
        mismatch = {"status": "mismatch", "daemon_config": {}}
        self.refused(mismatch, ProtocolVersionError)

    def test_a_reply_for_another_config(self):
        self.refused({**self.response, "config": {"tp_rank": 1}}, RuntimeError)

    def test_a_response_without_a_manifest(self):
        response = dict(self.response)
        del response["manifest"]
        self.refused(response, load_diff.SnapshotError)

    def test_entries_that_are_not_the_manifests_storages(self):
        entries = dict(self.response["entries"])
        name = next(iter(entries))
        self.refused({**self.response, "entries": {**entries, "99": entries[name]}})
        del entries[name]
        self.refused({**self.response, "entries": entries})
        bigger = {**self.response["entries"][name], "shape": [1 << 30]}
        self.refused(
            {**self.response, "entries": {**self.response["entries"], name: bigger}}
        )

    def test_entries_on_two_devices(self):
        # A storage off the daemon's device that came without its bytes.
        manifest = self.response["manifest"]
        name = next(iter(manifest["storages"]))
        storages = {
            **manifest["storages"],
            name: [manifest["storages"][name][0], "cuda"],
        }
        self.refused(
            {**self.response, "manifest": {**manifest, "storages": storages}},
            load_diff.SnapshotError,
        )

    def test_host_bytes_that_are_not_the_manifests(self):
        manifest = self.response["manifest"]
        name = next(iter(manifest["storages"]))
        nbytes = manifest["storages"][name][0]
        entries = {k: v for k, v in self.response["entries"].items() if k != name}
        for host in ({name: bytes(nbytes + 1)}, {name: "text"}, {"99": bytes(4)}):
            with self.subTest(host=host):
                self.refused(
                    {**self.response, "entries": entries,
                     "manifest": {**manifest, "host": host}},
                    load_diff.SnapshotError,
                )  # fmt: skip


class TestProtocolVersion(CustomTestCase):
    def test_daemon_refuses_a_fetch_of_another_version_before_exporting(self):
        _, daemon = daemon_with_snapshot()
        daemon.transport_backend.send_fetch_state_response = mock.Mock()
        versions = (None, 1, PROTOCOL_VERSION - 1, PROTOCOL_VERSION + 1, "2", True)
        for version in versions:
            with self.subTest(version=version):
                response = fetch(daemon, version=version, handshake=False)
                self.assertEqual(response["status"], "protocol_mismatch")
        daemon.transport_backend.send_fetch_state_response.assert_not_called()

    def test_the_protocol_version_follows_the_load_diff_format(self):
        # The response carries load_diff bytes. An engine that reads another
        # format than its daemon writes must be refused at the handshake,
        # before anything is mapped: a change of one version needs the other.
        self.assertEqual((PROTOCOL_VERSION, load_diff_format.FORMAT_VERSION), (3, 2))

    def test_engine_refuses_a_daemon_without_the_handshake(self):
        client, server = socket.socketpair()
        with client, server:
            client.settimeout(5)
            send_msg(server, {"status": "error", "message": "Unknown request type"})
            with self.assertRaises(ProtocolVersionError):
                handshake_with_daemon(client)


class TestDaemonReports(CustomTestCase):
    def test_no_record_is_refused(self):
        _, daemon = daemon_with_snapshot()
        with self.assertRaisesRegex(RuntimeError, "_initialize_model"):
            daemon._snapshot(None)


if __name__ == "__main__":
    unittest.main()
