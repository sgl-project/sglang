"""Receiver preparation, plan reuse and decode/apply lifecycle contracts."""

import copy
import sys
import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.weight_sync.gpu_delta import bindings as byte_layout
from sglang.srt.weight_sync.gpu_delta import layout, models, receiver
from sglang.srt.weight_sync.gpu_delta.host_worker import TENSOR_DTYPE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def publication_view(catalog):
    return SimpleNamespace(
        header={"plan_digest": catalog["plan_digest"]},
        read_catalog=lambda: catalog,
    )


class TestCanonicalPlanCache(unittest.TestCase):
    def test_nvtx_phase_balances_failure_without_cuda_events(self):
        events = []
        nvtx = SimpleNamespace(
            range_push=lambda name: events.append(("push", name)),
            range_pop=lambda: events.append(("pop",)),
        )

        def work(fail=False):
            if fail:
                raise RuntimeError("worker failure")
            return 42

        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                patch.object(receiver.profiling, "ENABLED", enabled),
                patch.object(receiver.profiling, "nvtx", nvtx, create=True),
                patch.object(
                    torch.cuda, "Event", side_effect=AssertionError("new event")
                ),
            ):
                events.clear()
                wrapped = receiver.profiling.annotate("gpu_delta.worker")(work)
                if not enabled:
                    self.assertIs(wrapped, work)
                self.assertEqual(wrapped(), 42)
                with self.assertRaisesRegex(RuntimeError, "worker failure"):
                    wrapped(True)
                with self.assertRaisesRegex(RuntimeError, "enqueue failed"):
                    with receiver.PreparedDelta._phase(SimpleNamespace(), "decode"):
                        raise RuntimeError("enqueue failed")
                self.assertEqual(
                    events,
                    (
                        [("push", "gpu_delta.worker"), ("pop",)] * 2
                        + [("push", "gpu_delta.pause.decode"), ("pop",)]
                        if enabled
                        else []
                    ),
                )

    def publication(self):
        entries = [
            {
                "name": name,
                "dtype": "U8",
                "shape": [2, 4],
                "encoding": "xor_bytes",
                "byte_order": "little",
                "nbytes": 8,
                "views": [
                    {"id": "a", "slices": [[0, 2], [0, 2]]},
                    {"id": "b", "slices": [[0, 2], [2, 4]]},
                ],
                "frames": [],
            }
            for name in ("local", "foreign")
        ]
        backend = SimpleNamespace(
            _canonical_plan_digest=None,
            _canonical_tensors=None,
            _canonical_tensor_ids=(),
            batch_plan=None,
            layers_per_batch=1,
            layout=SimpleNamespace(
                inventory={
                    name: {"dtype": "U8", "shape": [2, 4]}
                    for name in ("local", "foreign")
                },
                excluded={"foreign": "expert projection owned by another EP/TP rank"},
                bindings=[
                    SimpleNamespace(name="local", view_id="a", slices=[[0, 2], [0, 2]])
                ],
            ),
        )
        definitions = [
            {key: entry[key] for key in ("name", "dtype", "shape", "encoding")}
            | {"views": sorted(entry["views"], key=lambda view: view["id"])}
            for entry in sorted(entries, key=lambda entry: entry["name"])
        ]
        return backend, {
            "tensors": sorted(entries, key=lambda entry: entry["name"]),
            "plan_digest": receiver._digest(definitions),
        }

    def test_warm_plan_reuses_admitted_digest_with_fresh_payloads(self):
        backend, publication = self.publication()
        admitted, reused = receiver._qualify_canonical_plan(
            backend, publication_view(publication)
        )
        self.assertFalse(reused)
        self.assertEqual(admitted, {"local": (1, 8)})
        self.assertEqual(backend._canonical_tensor_ids, (1,))
        updated = publication_view(publication)
        with (
            patch.object(receiver, "_digest", side_effect=AssertionError("cold only")),
            patch.object(
                updated, "read_catalog", side_effect=AssertionError("cold only")
            ),
        ):
            entries, reused = receiver._qualify_canonical_plan(backend, updated)
        self.assertTrue(reused)
        self.assertIs(entries, admitted)
        self.assertEqual(backend._canonical_plan_digest, publication["plan_digest"])
        with self.assertRaisesRegex(ValueError, "plan changed"):
            receiver._qualify_canonical_plan(
                backend, publication_view(publication | {"plan_digest": "changed"})
            )

    def test_compact_scale_format_is_negotiated_without_duplicate_wire_fields(self):
        backend, publication = self.publication()
        definitions = []
        for entry in publication["tensors"]:
            entry["encoding"] = "raw_bytes"
            canonical = backend.layout.inventory[entry["name"]]
            canonical.update(
                encoding="raw_bytes",
                format="fp8-block128-ue8m0",
                checkpoint_dtype="F32",
            )
            definition = {
                key: entry[key]
                for key in ("name", "dtype", "shape", "encoding", "views")
            }
            definition.update(format="fp8-block128-ue8m0", checkpoint_dtype="F32")
            definitions.append(definition)
        publication["plan_digest"] = receiver._digest(
            sorted(definitions, key=lambda x: x["name"])
        )
        receiver._qualify_canonical_plan(backend, publication_view(publication))
        backend._canonical_plan_digest = None
        backend.layout.inventory["local"]["checkpoint_dtype"] = "U8"
        with self.assertRaisesRegex(ValueError, "negotiated canonical view plan"):
            receiver._qualify_canonical_plan(backend, publication_view(publication))

    def test_failed_cold_qualification_does_not_admit_a_cache(self):
        backend, publication = self.publication()
        invalid = copy.deepcopy(publication)
        invalid["plan_digest"] = "invalid"
        with self.assertRaisesRegex(ValueError, "negotiated canonical view plan"):
            receiver._qualify_canonical_plan(backend, publication_view(invalid))
        self.assertFalse(
            receiver._qualify_canonical_plan(backend, publication_view(publication))[1]
        )
        backend, publication = self.publication()
        invalid = copy.deepcopy(publication)
        invalid["tensors"][1]["views"][0]["slices"][0][1] = 1
        with self.assertRaisesRegex(ValueError, "conflicting rank view"):
            receiver._qualify_canonical_plan(backend, publication_view(invalid))
        self.assertIsNone(backend._canonical_plan_digest)
        self.assertFalse(
            receiver._qualify_canonical_plan(backend, publication_view(publication))[1]
        )
        backend, publication = self.publication()
        backend.layout.excluded["local"] = "static activation calibration"
        with self.assertRaisesRegex(ValueError, "unadmitted tensor"):
            receiver._qualify_canonical_plan(backend, publication_view(publication))


class TestReceiverPreparation(unittest.TestCase):
    def test_independent_mapping_uses_shared_apply_and_identity_contract(self):
        root = torch.nn.Module()
        root.config = SimpleNamespace(architectures=["IndependentTestModel"])
        root.runtime = torch.nn.Module()
        root.runtime.weight = torch.nn.Parameter(torch.zeros(6), requires_grad=False)
        root.cache = torch.zeros(3, 2)

        class IndependentMapping:
            def __init__(self, model):
                self.model = model
                self.parameters = byte_layout.ParameterBindings(model)
                self.derived = [
                    byte_layout.DerivedImage(
                        "cache", model.cache, model.runtime.weight.view(2, 3).t()
                    )
                ]
                self.consumers = [byte_layout.ConsumerSnapshot(lambda: (model.cache,))]

            def bind(self, name, meta):
                return self.parameters.bind(name, meta, "runtime.weight")

            def finish(self):
                pass

            @staticmethod
            def aliases(a, b):
                return False

        with (
            patch.dict(models._MODEL_MAPPINGS, IndependentTestModel=IndependentMapping),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.runtime_context": SimpleNamespace(
                        get_exec=lambda: self.fail(
                            "dense mapping accessed MoE topology"
                        )
                    )
                },
            ),
        ):
            plan = layout.GpuDeltaLayout(
                root, {"canonical.vector": {"dtype": "F32", "shape": [6]}}
            )
        prepared = receiver.PreparedDelta.__new__(receiver.PreparedDelta)
        prepared.backend = SimpleNamespace(layout=plan)
        description = plan.description
        self.assertEqual(
            description["rank_plan_digest"],
            byte_layout._digest(
                {key: description[key] for key in ("layouts", "tensors", "excluded")}
            ),
        )
        with patch.object(
            plan.bindings[0], "describe", side_effect=AssertionError("rebuild")
        ):
            backend = SimpleNamespace(layout=plan)
            self.assertIs(receiver.GpuDeltaBackend.describe(backend), description)
        prepared.device = torch.device("cpu")
        prepared.stream = SimpleNamespace(wait_stream=lambda _: None)
        prepared.batches, prepared.status_checks = [], []
        prepared.apply_bindings, prepared.apply_source_patches = [], []
        prepared.apply_fallback_groups = set()
        prepared.matrix_tensor_count, prepared.raw_tensor_count = 0, 1
        prepared.derived = plan.derived
        prepared.timings, prepared.h2d_bytes, prepared.target_version = {}, 0, 1
        pointer = root.cache.data_ptr()
        with (
            patch.object(prepared, "_allocate_paused"),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
            patch.object(torch.cuda, "stream", return_value=nullcontext()),
            patch.object(torch.cuda, "default_stream", return_value=prepared.stream),
            patch.object(
                torch.cuda,
                "Event",
                return_value=SimpleNamespace(
                    record=lambda _: None, synchronize=lambda: None
                ),
            ),
        ):
            for offset in (1, 7):
                prepared.derived = plan.derived
                prepared.error = torch.tensor([0])
                target = torch.arange(6, dtype=torch.float32) + offset
                prepared.raw_copies = {
                    torch.float32: ([plan.bindings[0].storage[0]], [target])
                }
                self.assertTrue(prepared.apply()["applied"])
                torch.testing.assert_close(
                    root.cache, target.view(2, 3).t(), rtol=0, atol=0
                )
                self.assertEqual(root.cache.data_ptr(), pointer)
        root.cache = root.cache.clone()
        with self.assertRaisesRegex(RuntimeError, "consumer storage changed"):
            plan.check_identity()

    def test_cuda_pipeline_prefetches_and_fences_two_input_slots(self):
        events = []

        class Stream:
            def __init__(self, name):
                self.name = name

            def wait_event(self, event):
                events.append((self.name, "wait", event.name))

        class Event:
            def __init__(self, name):
                self.name = name

            def record(self, stream):
                events.append((stream.name, "record", self.name))

        prepared = receiver.PreparedDelta.__new__(receiver.PreparedDelta)
        prepared.stream, prepared.io_stream = Stream("compute"), Stream("h2d")
        prepared.decoded_ready = [Event(f"ready{i}") for i in range(2)]
        prepared.decoded_free = [Event(f"free{i}") for i in range(2)]
        prepared.batches = [
            SimpleNamespace(
                decoder=SimpleNamespace(
                    copy_input=lambda stream, i=i: events.append(("copy", i)),
                    enqueue=lambda i=i: events.append(("decode", i)),
                ),
                zero_ranges=[],
                check_status=lambda i=i: events.append(("status", i)),
                apply=lambda i=i: events.append(("apply", i)),
            )
            for i in range(4)
        ]
        with patch.object(torch.cuda, "stream", return_value=nullcontext()):
            prepared._copy_batch(prepared.batches[0], 0)
            prepared._apply_cuda_batches()
        for i in range(4):
            self.assertLess(events.index(("copy", i)), events.index(("decode", i)))
            self.assertLess(events.index(("decode", i)), events.index(("status", i)))
            self.assertLess(events.index(("status", i)), events.index(("apply", i)))
            if i < 3:
                self.assertLess(
                    events.index(("copy", i + 1)), events.index(("decode", i))
                )
            if i >= 2:
                wait = events.index(("h2d", "wait", f"free{i % 2}"))
                self.assertLess(
                    events.index(("compute", "record", f"free{i % 2}")), wait
                )
                self.assertLess(wait, events.index(("copy", i)))

    def test_decode_table_preserves_sparse_rows_and_natural_zero_ranges(self):
        plans = [
            (np.array([[0, 1], [0, 64]], dtype="<i8"), 80, None),
            (np.empty((2, 0), dtype="<i8"), 0, None),
            (np.array([[2], [0]], dtype="<i8"), 32, None),
            (np.array([[3], [0]], dtype="<i8"), 24, None),
        ]
        tensors = np.zeros(4, dtype=TENSOR_DTYPE)
        tensors["frame_start"] = [0, 2, 2, 3]
        tensors["frame_count"] = [2, 0, 1, 0]
        tensors["gap_start"] = [0, 3, 4, 6]
        tensors["gap_count"] = [3, 1, 2, 1]
        publication = SimpleNamespace(
            tensors=tensors,
            frames=np.array(
                [[0, 16, 8], [3, 5, 7], [8, 8, 8], [8, 24, 16]], dtype="<i8"
            ),
            gaps=np.array(
                [[0, 8], [16, 8], [32, 16], [0, 16], [0, 16], [24, 8], [0, 24]],
                dtype="<i8",
            ),
        )
        # DE offsets differ from the shared prepared image; both must produce
        # the same decoded geometry without changing the shared metadata.
        original_frames, original_gaps = (
            publication.frames.copy(),
            publication.gaps.copy(),
        )
        for offsets in (np.array([64, 192, 128, 256]), np.array([0, 64, 80, 112])):
            table, counts, gaps = receiver._plan_decode(plans, publication, offsets)
            self.assertEqual(table.dtype, np.int64)
            self.assertTrue(table.flags.c_contiguous)
            np.testing.assert_array_equal(
                table,
                [
                    [offsets[0], offsets[0] + 16, offsets[2] + 8],
                    [3, 5, 7],
                    [8, 8, 8],
                    [8, 24, 16],
                ],
            )
            self.assertEqual(counts, [2, 0, 1, 0])
            for actual, expected in zip(
                gaps,
                (
                    [[0, 8], [16, 8], [32, 16], [64, 16]],
                    [],
                    [[0, 16], [24, 8]],
                    [[0, 24]],
                ),
            ):
                np.testing.assert_array_equal(
                    actual, np.asarray(expected, dtype="<i8").reshape(-1, 2)
                )
        np.testing.assert_array_equal(publication.frames, original_frames)
        np.testing.assert_array_equal(publication.gaps, original_gaps)
        empty, counts, gaps = receiver._plan_decode(plans[-1:], publication, offsets)
        self.assertEqual(empty.shape, (4, 0))
        self.assertEqual(counts, [0])
        np.testing.assert_array_equal(gaps[0], [[0, 24]])
        empty, counts, gaps = receiver._plan_decode([], publication, offsets)
        self.assertEqual(empty.shape, (4, 0))
        self.assertEqual((counts, gaps), ([], []))

    def test_prepare_uses_shared_publication_and_only_local_ordinals(self):
        backend, catalog = TestCanonicalPlanCache().publication()
        backend.device = torch.device("cpu")
        backend.decoders = {"lz4": SimpleNamespace(backend="hardware")}
        refs = []

        class TrackedEntry(dict):
            pass

        def read_catalog():
            value = copy.deepcopy(catalog)
            foreign = TrackedEntry(value["tensors"][0])
            value["tensors"][0] = foreign
            refs.append(weakref.ref(foreign))
            return value

        publication = SimpleNamespace(
            header={
                "codec": "lz4-zstd",
                "backend": "hardware",
                "target_version": 7,
                "plan_digest": catalog["plan_digest"],
            },
            read_catalog=read_catalog,
        )

        def prepare_local(actual, tensor_ids, timings, hardware):
            self.assertIs(actual, publication)
            self.assertTrue(hardware)
            self.assertIsNone(refs[-1]())
            self.assertEqual(tuple(tensor_ids), (1,))
            raise RuntimeError("stop at local preparation boundary")

        backend.host_arena = SimpleNamespace(
            prepare_publication=lambda *args: publication, prepare_local=prepare_local
        )
        with self.assertRaisesRegex(RuntimeError, "local preparation boundary"):
            receiver.PreparedDelta(backend, "manifest.json", "digest", {})

    def test_layer_batch_cache_tracks_active_names_and_grouping(self):
        names = [
            "model.layers.0.a",
            "model.layers.0.b",
            "model.layers.1.a",
            "model.layers.2.a",
            "model.layers.3.a",
            "standalone",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ]
        bindings = [
            byte_layout._direct_binding(
                name,
                {"dtype": "U8", "shape": [2, 4]},
                torch.zeros(2, 4, dtype=torch.uint8),
            )
            for name in names
        ]
        entries = {b.name: (i, 8) for i, b in enumerate(bindings)}
        backend = SimpleNamespace(batch_plan=None)
        # This test checks grouping/cache ownership, not Triton launch plans.
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.weight_sync.gpu_delta.apply": SimpleNamespace(
                    plan_apply=lambda outputs: None
                )
            },
        ):
            previous = receiver._plan_layers(backend, bindings, entries)
            self.assertIs(receiver._plan_layers(backend, bindings, entries), previous)
            reduced = receiver._plan_layers(backend, bindings[1:], entries)
            self.assertIsNot(reduced, previous)
            self.assertEqual(reduced[3][0][0, 0], 1)
            for count, expected in ((1, [[0], [1], [2], [3]]), (2, [[0, 1], [2, 3]])):
                grouped = receiver._plan_layers(backend, bindings, entries, count)
                self.assertEqual(
                    [
                        list(dict.fromkeys(bindings[i].layer for i in plan[0][0]))
                        for plan in grouped
                    ],
                    [[None], [None], [None], *expected],
                )
                self.assertIs(
                    receiver._plan_layers(backend, bindings, entries, count), grouped
                )

    def test_backend_reads_inventory_once_and_drains_all_streams_on_failure(self):
        fake_plan = SimpleNamespace(
            check_identity=lambda: None,
            description={"rank_plan_digest": "digest", "tensors": []},
            rank_plan_digest="digest",
            bindings=[],
            excluded={},
        )
        fake_model = SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda", 0))])
        )
        with (
            patch.dict(receiver.os.environ, GPU_DELTA_LAYERS_PER_BATCH="3"),
            patch.object(receiver, "GpuDeltaLayout", return_value=fake_plan),
            patch("sglang.srt.weight_sync.gpu_delta.host.HostArena"),
            patch(
                "sglang.srt.weight_sync.gpu_delta.checkpoint.read_canonical_checkpoint_inventory",
                return_value={"weight": {"shape": [1], "dtype": "U8"}},
            ) as read_inventory,
        ):
            backend = receiver.GpuDeltaBackend(
                SimpleNamespace(model=fake_model), {"engine_id": "test-engine"}
            )
            self.assertEqual(backend.decoders, {})
            self.assertIsNone(backend.apply_stream)
            self.assertIsNone(backend.io_stream)
            self.assertEqual(backend.layers_per_batch, 3)
            with patch.dict(receiver.os.environ, GPU_DELTA_LAYERS_PER_BATCH="0"):
                backend.describe()
                backend.describe()
                self.assertEqual(backend.layers_per_batch, 3)
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    receiver.GpuDeltaBackend(
                        SimpleNamespace(model=fake_model), {"engine_id": "invalid"}
                    )
            read_inventory.assert_called_once()
            backend.host_arena.prepare_publication.side_effect = ValueError(
                "bad manifest"
            )
            with patch.object(torch.cuda, "device", return_value=nullcontext()):
                with self.assertRaisesRegex(ValueError, "bad manifest"):
                    backend.prepare("manifest.json", "digest", {})
            backend.host_arena.close.assert_called_once()

        drains = []
        backend = receiver.GpuDeltaBackend.__new__(receiver.GpuDeltaBackend)
        backend.host_arena = SimpleNamespace(close=lambda: drains.append("backend"))

        def decode_failure():
            drains.append("decode")
            raise RuntimeError("decode stream failed")

        prepared = receiver.PreparedDelta.__new__(receiver.PreparedDelta)
        prepared.io_stream = SimpleNamespace(synchronize=decode_failure)
        prepared.stream = SimpleNamespace(synchronize=lambda: drains.append("compute"))
        prepared._release_gpu = lambda: drains.append("release-gpu")
        prepared.host_snapshot = SimpleNamespace(
            close=lambda: drains.append("snapshot")
        )
        prepared.raw_pinned = object()
        with self.assertRaisesRegex(RuntimeError, "decode stream failed"):
            backend.discard(prepared)
        self.assertEqual(drains, ["decode", "compute"])
        self.assertIsNotNone(prepared.raw_pinned)
        self.assertIsNotNone(prepared.host_snapshot)

        drains.clear()
        prepared.io_stream.synchronize = lambda: drains.append("decode")
        backend.discard(prepared)
        self.assertEqual(
            drains, ["decode", "compute", "release-gpu", "snapshot", "backend"]
        )
        self.assertIsNone(prepared.raw_pinned)
        self.assertIsNone(prepared.host_snapshot)


if __name__ == "__main__":
    unittest.main()
