# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for shared, opt-in weight-cache correctness primitives."""

import json
import multiprocessing as mp
import os
import socket
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.weight_cache_common.checkpoint import (
    MANIFEST_FILENAME,
    CheckpointManifest,
    build_manifest,
    check_verified_stats,
    verify_manifest,
    write_manifest,
)
from sglang.weight_cache_common.descriptors import StateManifest
from sglang.weight_cache_common.identity import socket_path, source_digest
from sglang.weight_cache_common.liveness import (
    ProcessIdentity,
    ProducerDiedError,
    ProducerWatchdog,
)
from sglang.weight_cache_common.mapping import import_state
from sglang.weight_cache_common.traversal import snapshot_module, storage_byte_views

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _model(device="cpu"):
    module = nn.Module()
    module.child = nn.Module()
    base = torch.arange(24, dtype=torch.float32, device=device).reshape(4, 6)
    module.weight = nn.Parameter(base[1:3, 1:5], requires_grad=False)
    module.weight.input_dim = 1
    module.child.tied = module.weight
    buffer = base[:, ::2]
    module.register_buffer("persistent", buffer)
    module.child.register_buffer("scratch", buffer, persistent=False)
    module.register_buffer("other_view", base.t(), persistent=False)
    module.register_buffer("raw_bytes", base.view(torch.uint8), persistent=False)
    module.register_buffer("empty", torch.empty(0, device=device))
    module.register_buffer("scalar", torch.tensor(3.0, device=device))
    module.eval()
    module.child.train()
    return module


def _hold_process(conn):
    conn.send(ProcessIdentity.read(os.getpid()))
    conn.recv()
    conn.close()


class TestStateMapping(unittest.TestCase):
    def setUp(self):
        self.source = _model()
        self.snapshot = snapshot_module(self.source)

    def test_roundtrip_aliases_layout_persistence_and_training(self):
        manifest = StateManifest.from_dict(
            json.loads(json.dumps(self.snapshot.manifest.to_dict()))
        )
        self.assertEqual(manifest, self.snapshot.manifest)
        self.assertEqual(manifest.digest, self.snapshot.manifest.digest)
        self.assertEqual(manifest.unique_storage_bytes, 24 * 4 + 4)
        target = _model("meta")
        target.train()
        import_state(target, manifest, storage_byte_views(self.snapshot))
        self.assertIs(target.weight, target.child.tied)
        self.assertIs(target.persistent, target.child.scratch)
        self.assertIsNot(target.persistent, target.other_view)
        self.assertEqual(
            target.weight.untyped_storage()._cdata,
            target.other_view.untyped_storage()._cdata,
        )
        self.assertEqual(target.weight.storage_offset(), 7)
        self.assertEqual(target.weight.stride(), (6, 1))
        self.assertEqual(target.weight.input_dim, 1)
        self.assertFalse(target.weight.requires_grad)
        self.assertFalse(target.training)
        self.assertTrue(target.child.training)
        self.assertEqual(set(target.state_dict()), set(self.source.state_dict()))
        for name, tensor in snapshot_module(target).tensors.items():
            torch.testing.assert_close(tensor, self.snapshot.tensors[name])
        self.assertEqual(snapshot_module(target).manifest, manifest)

    def test_names_validation_is_transactional(self):
        target = _model("meta")
        target.register_buffer("unexpected", torch.empty(1, device="meta"))
        old_weight = target.weight
        with self.assertRaisesRegex(ValueError, "Tensor names differ"):
            import_state(
                target, self.snapshot.manifest, storage_byte_views(self.snapshot)
            )
        self.assertIs(target.weight, old_weight)
        self.assertEqual(target.weight.device.type, "meta")

    def test_bad_storage_view_does_not_replace_parameters(self):
        target = _model("meta")
        views = storage_byte_views(self.snapshot)
        group = self.snapshot.manifest.tensors[0].storage_group
        views[group] = torch.zeros(1, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "Invalid full-storage"):
            import_state(target, self.snapshot.manifest, views)
        self.assertEqual(target.weight.device.type, "meta")

    def test_manifest_rejects_out_of_bounds_and_bad_stride(self):
        manifest = self.snapshot.manifest
        first = manifest.tensors[0]
        for bad in (
            replace(first, storage_offset=10**9),
            replace(first, stride=(-1,) * len(first.shape)),
            replace(first, shape=(True,)),
            replace(first, kind="unknown"),
            replace(first, dtype="not_a_dtype"),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(manifest, tensors=(bad, *manifest.tensors[1:])).validate()

    def test_manifest_rejects_wrong_abi_and_duplicate_names(self):
        manifest = self.snapshot.manifest
        for bad in (
            replace(manifest, cache_abi=999),
            replace(manifest, tensors=(*manifest.tensors, manifest.tensors[0])),
            replace(manifest, training=(("", False), ("", True))),
            replace(manifest, storages=(*manifest.storages, manifest.storages[0])),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bad.validate()

    def test_exact_tie_cannot_change_kind_or_layout(self):
        manifest = self.snapshot.manifest
        tied = next(tensor for tensor in manifest.tensors if tensor.name == "weight")
        tensors = tuple(
            replace(tensor, storage_offset=tensor.storage_offset + 1)
            if tensor.name == "child.tied"
            else tensor
            for tensor in manifest.tensors
        )
        self.assertEqual(
            tied.tensor_group,
            next(t.tensor_group for t in tensors if t.name == "child.tied"),
        )
        with self.assertRaisesRegex(ValueError, "Conflicting exact tensor tie"):
            replace(manifest, tensors=tensors).validate()

    def test_empty_storages_stay_distinct(self):
        module = nn.Module()
        module.register_buffer("a", torch.empty(0))
        module.register_buffer("b", torch.empty(0))
        manifest = snapshot_module(module).manifest
        self.assertEqual(len(manifest.storages), 2)
        self.assertEqual(manifest.unique_storage_bytes, 0)

    def test_nested_tensor_metadata_cannot_leave_stale_meta_views(self):
        target = _model("meta")
        target.weight.extra = {"nested": [torch.empty(2, device="meta")]}
        with self.assertRaisesRegex(ValueError, "metadata needs an adapter"):
            import_state(
                target, self.snapshot.manifest, storage_byte_views(self.snapshot)
            )
        self.assertEqual(target.weight.device.type, "meta")

    def test_cross_kind_tie_and_cycles_rejected(self):
        module = nn.Module()
        module.weight = nn.Parameter(torch.ones(2), requires_grad=False)
        module.register_buffer("alias", module.weight)
        with self.assertRaisesRegex(ValueError, "cross-kind"):
            snapshot_module(module)
        cyclic = nn.Module()
        cyclic.child = cyclic
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            snapshot_module(cyclic)

    def test_non_meta_import_and_parameter_subclasses_rejected(self):
        with self.assertRaisesRegex(ValueError, "meta-initialized"):
            import_state(
                _model(), self.snapshot.manifest, storage_byte_views(self.snapshot)
            )

        class SpecialParameter(nn.Parameter):
            pass

        target = _model("meta")
        target.weight = SpecialParameter(target.weight, requires_grad=False)
        target.child.tied = target.weight
        with self.assertRaisesRegex(ValueError, "Parameter subclass"):
            import_state(
                target, self.snapshot.manifest, storage_byte_views(self.snapshot)
            )


class TestIdentity(unittest.TestCase):
    def test_socket_path_binds_and_full_identity_is_not_the_locator(self):
        with tempfile.TemporaryDirectory(prefix="wc-") as directory:
            root = Path(directory)
            digest = "a" * 64
            path = socket_path("GPU-" + "0" * 36, digest, runtime_dir=root)
            path.parent.mkdir()
            with (
                socket.socket(socket.AF_UNIX) as server,
                socket.socket(socket.AF_UNIX) as client,
            ):
                server.bind(str(path))
                server.listen()
                client.connect(str(path))
                peer, _ = server.accept()
                peer.close()
            collision = "a" * 32 + "b" * 32
            self.assertEqual(
                path, socket_path("GPU-" + "0" * 36, collision, runtime_dir=root)
            )
            self.assertNotEqual(
                digest, collision
            )  # handshake must compare these, not paths

    def test_default_xdg_and_encoded_path_budget(self):
        with patch.dict(os.environ, {}, clear=True):
            path = socket_path("GPU-test", "a" * 64)
            self.assertEqual(len(os.fsencode(path)), 89)
            self.assertTrue(str(path).startswith("/tmp/sglang_diffusion_weight_cache/"))
        with patch.dict(os.environ, {"XDG_RUNTIME_DIR": "/run/user/1000"}, clear=True):
            self.assertLessEqual(
                len(os.fsencode(socket_path("GPU-test", "a" * 64))), 107
            )
        for root in (Path("/tmp") / ("x" * 100), Path("/tmp") / ("中" * 20)):
            with self.assertRaisesRegex(ValueError, "107 bytes"):
                socket_path("GPU-test", "a" * 64, runtime_dir=root)
        with self.assertRaises(ValueError):
            socket_path("GPU-test", "a" * 16)

    def test_source_digest_covers_reused_code_not_just_diffusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "multimodal_gen").mkdir()
            (root / "srt").mkdir()
            (root / "multimodal_gen" / "__init__.py").write_text("x = 1\n")
            dependency = root / "srt" / "helper.py"
            dependency.write_text("x = 1\n")
            before = source_digest(root)
            dependency.write_text("x = 2\n")
            self.assertNotEqual(before, source_digest(root))

    def test_checkpoint_content_and_fast_stat_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            weights = root / "weights.bin"
            weights.write_bytes(b"weights")
            names = ["weights.bin", "config.json", MANIFEST_FILENAME]
            manifest = build_manifest(root, names)
            destination = write_manifest(root, manifest)
            self.assertEqual(CheckpointManifest.read(destination), manifest)
            receipt = verify_manifest(root, manifest, ["weights.bin", "config.json"])
            with patch(
                "sglang.weight_cache_common.checkpoint.hash_file",
                side_effect=AssertionError("Client must not hash weights"),
            ):
                check_verified_stats(root, manifest, receipt)
            weights.write_bytes(b"changed")  # same size, but different content and stat
            # Some filesystems coalesce immediate writes into the same timestamp
            # tick. Exercise a changed stat deterministically, without sleeping.
            stamp = weights.stat()
            os.utime(weights, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))
            with self.assertRaisesRegex(ValueError, "changed since producer"):
                check_verified_stats(root, manifest, receipt)
            with self.assertRaisesRegex(ValueError, "content does not match"):
                verify_manifest(root, manifest, ["weights.bin", "config.json"])

    def test_republished_manifest_invalidates_even_when_stat_checks_are_inconclusive(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "weights"
            path.write_bytes(b"before")
            original = build_manifest(root, ["weights"])
            receipt = verify_manifest(root, original, ["weights"])
            path.write_bytes(b"after!")
            republished = build_manifest(root, ["weights"])
            with patch(
                "sglang.weight_cache_common.checkpoint.FileStamp.read",
                return_value=receipt.stamps[0][1],
            ):
                with self.assertRaisesRegex(
                    ValueError, "differs from verified producer"
                ):
                    check_verified_stats(root, republished, receipt)

    def test_checkpoint_rejects_recipe_mismatch_duplicate_and_escaping_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights").write_bytes(b"test")
            manifest = build_manifest(root, ["weights"])
            with self.assertRaisesRegex(ValueError, "Duplicate normalized"):
                build_manifest(root, ["weights", "./weights"])
            with self.assertRaises(ValueError):
                build_manifest(root, ["../outside"])
            (root / "escape").symlink_to("/etc/hosts")
            with self.assertRaisesRegex(ValueError, "symlink escapes"):
                build_manifest(root, ["escape"])
            (root / "config").write_text("{}")
            with self.assertRaisesRegex(ValueError, "file set differs"):
                verify_manifest(root, manifest, ["weights", "config"])


class TestProducerIdentity(unittest.TestCase):
    def test_invalid_intervals_do_not_start_a_watchdog(self):
        identity = ProcessIdentity.read(os.getpid())
        for interval in (0, -1, 11, float("nan"), float("inf"), True):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                ProducerWatchdog(identity, poll_interval=interval)

    def test_start_identity_checked_before_watchdog_starts(self):
        identity = ProcessIdentity.read(os.getpid())
        self.assertTrue(identity.is_alive())
        bad = replace(identity, start_ticks=identity.start_ticks + 1)
        self.assertFalse(bad.is_alive())
        with self.assertRaises(ProducerDiedError):
            ProducerWatchdog(bad)

    def test_pidfd_and_polling_detect_producer_exit(self):
        for pidfd in (True, False):
            with self.subTest(pidfd=pidfd):
                parent, child = mp.get_context("spawn").Pipe()
                producer = mp.get_context("spawn").Process(
                    target=_hold_process, args=(child,)
                )
                producer.start()
                child.close()
                watcher = None
                try:
                    self.assertTrue(parent.poll(20), "producer did not start")
                    identity = parent.recv()
                    lost = threading.Event()
                    watcher = ProducerWatchdog(
                        identity, on_death=lost.set, use_pidfd=pidfd, poll_interval=0.02
                    )
                    watcher.check_alive()
                    parent.send("exit")
                    producer.join(10)
                    self.assertEqual(producer.exitcode, 0)
                    self.assertTrue(lost.wait(3))
                    with self.assertRaises(ProducerDiedError):
                        watcher.check_alive()
                finally:
                    if watcher is not None:
                        watcher.close()
                    if producer.is_alive():
                        producer.kill()
                        producer.join(5)
                    parent.close()


if __name__ == "__main__":
    unittest.main()
