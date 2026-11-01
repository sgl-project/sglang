"""Receiver preparation, plan reuse and decode/apply lifecycle contracts."""

import copy
import json
import sys
import tempfile
import unittest
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.weight_sync.gpu_delta import bindings as byte_layout
from sglang.srt.weight_sync.gpu_delta import layout, models, receiver
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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
                    [("push", "gpu_delta.worker"), ("pop",)] * 2
                    + [("push", "gpu_delta.pause.decode"), ("pop",)]
                    if enabled
                    else [],
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
            batch_plan=None,
            layers_per_batch=1,
            layout=SimpleNamespace(
                inventory={
                    name: {"dtype": "U8", "shape": [2, 4]}
                    for name in ("local", "foreign")
                },
                excluded={"foreign": "expert owned by another EP rank"},
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
            "tensors": entries,
            "plan_digest": receiver._digest(definitions),
        }

    def test_warm_plan_reuses_admitted_digest_with_fresh_payloads(self):
        backend, publication = self.publication()
        _, reused = receiver._qualify_canonical_plan(backend, publication)
        self.assertFalse(reused)
        updated = copy.deepcopy(publication)
        for entry in updated["tensors"]:
            entry["frames"] = [{"new": "per-publication payload geometry"}]
        with patch.object(receiver, "_digest", side_effect=AssertionError("cold only")):
            entries, reused = receiver._qualify_canonical_plan(backend, updated)
        self.assertTrue(reused)
        self.assertIs(entries["local"], updated["tensors"][0])
        self.assertEqual(backend._canonical_plan_digest, publication["plan_digest"])
        with self.assertRaisesRegex(ValueError, "plan changed"):
            receiver._qualify_canonical_plan(
                backend, updated | {"plan_digest": "changed"}
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
        receiver._qualify_canonical_plan(backend, publication)
        backend._canonical_plan_digest = None
        backend.layout.inventory["local"]["checkpoint_dtype"] = "U8"
        with self.assertRaisesRegex(ValueError, "negotiated canonical view plan"):
            receiver._qualify_canonical_plan(backend, publication)

    def test_failed_cold_qualification_does_not_admit_a_cache(self):
        backend, publication = self.publication()
        invalid = copy.deepcopy(publication)
        invalid["plan_digest"] = "invalid"
        with self.assertRaisesRegex(ValueError, "negotiated canonical view plan"):
            receiver._qualify_canonical_plan(backend, invalid)
        self.assertFalse(receiver._qualify_canonical_plan(backend, publication)[1])
        backend, publication = self.publication()
        invalid = copy.deepcopy(publication)
        invalid["tensors"][0]["views"][0]["slices"][0][1] = 1
        with self.assertRaisesRegex(ValueError, "conflicting rank view"):
            receiver._qualify_canonical_plan(backend, invalid)
        self.assertIsNone(backend._canonical_plan_digest)
        self.assertFalse(receiver._qualify_canonical_plan(backend, publication)[1])
        backend, publication = self.publication()
        backend.layout.excluded["local"] = "static activation calibration"
        with self.assertRaisesRegex(ValueError, "unadmitted tensor"):
            receiver._qualify_canonical_plan(backend, publication)


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
        prepared.device, prepared.stream = torch.device("cpu"), object()
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

    def test_decode_table_preserves_sparse_rows_and_natural_zero_ranges(self):
        sparse = SimpleNamespace(name="sparse")
        tail = SimpleNamespace(name="tail")
        omitted = SimpleNamespace(name="omitted")
        plans = [
            ([(sparse, 0, 48), (omitted, 64, 16)], 80, None),
            ([], 0, None),
            ([(tail, 0, 32)], 32, None),
            ([(omitted, 0, 24)], 24, None),
        ]
        entries = {
            "sparse": {
                "frames": [
                    dict(
                        encoded_offset=0,
                        encoded_bytes=3,
                        decoded_offset=8,
                        decoded_bytes=8,
                    ),
                    dict(
                        encoded_offset=16,
                        encoded_bytes=5,
                        decoded_offset=24,
                        decoded_bytes=8,
                    ),
                ]
            },
            "tail": {
                "frames": [
                    dict(
                        encoded_offset=8,
                        encoded_bytes=7,
                        decoded_offset=16,
                        decoded_bytes=8,
                    )
                ]
            },
            "omitted": {"frames": []},
        }
        records = {
            "sparse": {"offset": 64},
            "tail": {"offset": 128},
            "omitted": {"offset": 192},
        }
        table, counts, gaps = receiver._plan_decode(plans, entries, records)
        self.assertEqual(table.dtype, np.int64)
        self.assertTrue(table.flags.c_contiguous)
        # Native rows are input, encoded size, decoded size, output; batches
        # retain their index even when empty and output offsets restart at zero.
        np.testing.assert_array_equal(
            table,
            [[64, 80, 136], [3, 5, 7], [8, 8, 8], [8, 24, 16]],
        )
        self.assertEqual(counts, [2, 0, 1, 0])
        self.assertEqual(
            gaps,
            [
                [(0, 8), (16, 8), (32, 16), (64, 16)],
                [],
                [(0, 16), (24, 8)],
                [(0, 24)],
            ],
        )  # Tensor alignment padding at [48,64) is deliberately excluded.
        # Exhausting the iterator is required to emit the last trailing gap
        # and the later fully omitted batch, even with no frame yields at all.
        empty, counts, gaps = receiver._plan_decode(plans[-1:], entries, records)
        self.assertEqual(empty.shape, (4, 0))
        self.assertEqual(counts, [0])
        self.assertEqual(gaps, [[(0, 24)]])
        empty, counts, gaps = receiver._plan_decode([], entries, records)
        self.assertEqual(empty.shape, (4, 0))
        self.assertEqual((counts, gaps), ([], []))

    def test_prepare_releases_foreign_metadata_before_local_work(self):
        backend, manifest = TestCanonicalPlanCache().publication()
        backend.device = torch.device("cpu")
        backend.payload_pool = object()
        manifest.update(
            codec="lz4-zstd",
            frame_bytes=1 << 20,
            base_version=0,
            target_version=7,
        )
        # A saved HF-base delta may jump directly to a later committed version.
        refs = []
        parse = receiver.orjson.loads

        class TrackedEntry(dict):
            pass

        def tracked_parse(content):
            value = parse(content)
            foreign = TrackedEntry(value["tensors"][1])
            value["tensors"][1] = foreign
            refs.append(weakref.ref(foreign))
            return value

        def admit(*args):
            self.assertIsNotNone(refs[-1]())
            return {}, {}

        def prepare_local(index, files, entries, pool, timings):
            self.assertIsNone(refs[-1]())
            self.assertEqual([entry["name"] for entry in entries], ["local"])
            raise RuntimeError("stop at local preparation boundary")

        backend.host_arena = SimpleNamespace(
            prepare_encoded=admit, prepare_local=prepare_local
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            content = json.dumps(manifest).encode()
            path.write_bytes(content)
            with patch.object(receiver.orjson, "loads", side_effect=tracked_parse):
                with self.assertRaisesRegex(RuntimeError, "local preparation boundary"):
                    receiver.PreparedDelta(
                        backend,
                        path,
                        receiver.hashlib.sha256(content).hexdigest(),
                        {},
                    )

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
        entries = {b.name: {"nbytes": 8} for b in bindings}
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
            self.assertEqual(reduced[3][0][0][0].name, names[1])
            for count, expected in ((1, [[0], [1], [2], [3]]), (2, [[0, 1], [2, 3]])):
                grouped = receiver._plan_layers(backend, bindings, entries, count)
                self.assertEqual(
                    [
                        list(dict.fromkeys(b.layer for b, _, _ in plan[0]))
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
            self.assertIsNone(backend.de_stream)
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
            backend.close()

        drains = []

        def decode_failure():
            drains.append("decode")
            raise RuntimeError("decode stream failed")

        prepared = receiver.PreparedDelta.__new__(receiver.PreparedDelta)
        prepared.de_stream = SimpleNamespace(synchronize=decode_failure)
        prepared.stream = SimpleNamespace(synchronize=lambda: drains.append("compute"))
        with self.assertRaisesRegex(RuntimeError, "decode stream failed"):
            prepared.close()
        self.assertEqual(drains, ["decode", "compute"])


if __name__ == "__main__":
    unittest.main()
