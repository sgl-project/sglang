"""Byte-layout tests require PyTorch only; no FlashInfer import or GPU JIT."""

import copy
import importlib.util
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

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# Keep this algebra test runnable in CPU development environments without
# importing serving-time CUDA dependencies through sglang's package root.
_path = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/weight_sync/gpu_delta/layout.py"
)
_spec = importlib.util.spec_from_file_location("gpu_delta_layout_under_test", _path)
layout = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = layout
_spec.loader.exec_module(layout)

from sglang.srt.weight_sync.gpu_delta import bindings as byte_layout
from sglang.srt.weight_sync.gpu_delta import models
from sglang.srt.weight_sync.gpu_delta.flashinfer import FlashInferLayout


def _nvfp4_method():
    cls = type("ModelOptNvFp4FusedMoEMethod", (), {})
    method = cls()
    method.runner = None
    method._is_cutedsl_v2_standard = True
    method.quant_config = SimpleNamespace(is_checkpoint_nvfp4_serialized=True)
    return method


def _bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def _apply_binding(binding, mask):
    for view in binding.apply_bindings():
        selected = view.selected_bytes(mask.reshape(-1)) if binding.segments else mask
        for destination in view.destinations:
            destination.bitwise_xor_(selected.reshape(destination.shape))


class TestCanonicalPlanCache(unittest.TestCase):
    def test_nvtx_phase_balances_failure_without_cuda_events(self):
        events = []
        nvtx = SimpleNamespace(
            range_push=lambda name: events.append(("push", name)),
            range_pop=lambda: events.append(("pop",)),
        )
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                patch.object(layout.profiling, "ENABLED", enabled),
                patch.object(layout.profiling, "nvtx", nvtx, create=True),
                patch.object(
                    torch.cuda, "Event", side_effect=AssertionError("new event")
                ),
            ):
                events.clear()
                with self.assertRaisesRegex(RuntimeError, "enqueue failed"):
                    with layout.PreparedDelta._phase(SimpleNamespace(), "decode"):
                        raise RuntimeError("enqueue failed")
                self.assertEqual(
                    events,
                    [("push", "gpu_delta.pause.decode"), ("pop",)] if enabled else [],
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
        return backend, {"tensors": entries, "plan_digest": layout._digest(definitions)}

    def test_warm_plan_reuses_admitted_digest_with_fresh_payloads(self):
        backend, publication = self.publication()
        _, reused = layout._qualify_canonical_plan(backend, publication)
        self.assertFalse(reused)
        updated = copy.deepcopy(publication)
        for entry in updated["tensors"]:
            entry["frames"] = [{"new": "per-publication payload geometry"}]
        with patch.object(layout, "_digest", side_effect=AssertionError("cold only")):
            entries, reused = layout._qualify_canonical_plan(backend, updated)
        self.assertTrue(reused)
        self.assertIs(entries["local"], updated["tensors"][0])
        self.assertEqual(backend._canonical_plan_digest, publication["plan_digest"])
        with self.assertRaisesRegex(ValueError, "plan changed"):
            layout._qualify_canonical_plan(
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
        publication["plan_digest"] = layout._digest(
            sorted(definitions, key=lambda x: x["name"])
        )
        layout._qualify_canonical_plan(backend, publication)
        backend._canonical_plan_digest = None
        backend.layout.inventory["local"]["checkpoint_dtype"] = "U8"
        with self.assertRaisesRegex(ValueError, "negotiated canonical view plan"):
            layout._qualify_canonical_plan(backend, publication)

    def test_failed_cold_qualification_does_not_admit_a_cache(self):
        backend, publication = self.publication()
        invalid = copy.deepcopy(publication)
        invalid["plan_digest"] = "invalid"
        with self.assertRaisesRegex(ValueError, "negotiated canonical view plan"):
            layout._qualify_canonical_plan(backend, invalid)
        self.assertFalse(layout._qualify_canonical_plan(backend, publication)[1])
        backend, publication = self.publication()
        invalid = copy.deepcopy(publication)
        invalid["tensors"][0]["views"][0]["slices"][0][1] = 1
        with self.assertRaisesRegex(ValueError, "conflicting rank view"):
            layout._qualify_canonical_plan(backend, invalid)
        self.assertIsNone(backend._canonical_plan_digest)
        self.assertFalse(layout._qualify_canonical_plan(backend, publication)[1])
        backend, publication = self.publication()
        backend.layout.excluded["local"] = "static activation calibration"
        with self.assertRaisesRegex(ValueError, "unadmitted tensor"):
            layout._qualify_canonical_plan(backend, publication)


class TestFlashInferDeltaLayout(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(
            byte_layout,
            "_identify_moe_layout",
            side_effect=lambda layer: byte_layout.identify_layout(
                layer, "flashinfer_cutedsl"
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        consumer = patch.object(
            layout,
            "consumer_objects",
            side_effect=lambda layer, _: (
                layer._cutedsl_wrapper,
                layer._cutedsl_scales,
                *layer._cutedsl_scales,
            ),
        )
        consumer.start()
        self.addCleanup(consumer.stop)

    def test_tp_selection_reads_reused_decoder_scratch(self):
        for dtype, name in (
            (torch.uint8, "U8"),
            (torch.bfloat16, "BF16"),
            (torch.float32, "F32"),
        ):
            with self.subTest(dtype=name):
                size = 12 * 20 * torch.empty((), dtype=dtype).element_size()
                canonical = torch.randint(256, (size,), dtype=torch.uint8)
                target = canonical.view(dtype).reshape(12, 20)[2:10, 5:13].clone()
                binding = byte_layout._direct_binding(
                    "weight",
                    {"dtype": name, "shape": [12, 20]},
                    target,
                    [[2, 10], [5, 13]],
                )
                decoded = torch.zeros_like(canonical)
                payload = binding.selected_bytes(decoded)
                self.assertEqual(
                    payload.untyped_storage().data_ptr(),
                    decoded.untyped_storage().data_ptr(),
                )
                self.assertFalse(payload.is_contiguous())
                # A prepared TP view observes later decoder writes without a
                # cached contiguous copy. Native tests cover the actual XOR gate.
                for _ in range(2):
                    decoded.random_(256)
                    expected = decoded.view(dtype).reshape(12, 20)[2:10, 5:13]
                    torch.testing.assert_close(_bytes(payload), _bytes(expected))

    def test_unfused_bf16_shared_experts_use_ordinary_tp8_slices(self):
        tp_rank, intermediate, hidden = 3, 32, 16
        shard = intermediate // 8
        before = {
            key: torch.randn(shape).bfloat16()
            for key, shape in (
                ("gate", (intermediate, hidden)),
                ("up", (intermediate, hidden)),
                ("down", (hidden, intermediate)),
            )
        }
        after = {key: torch.randn_like(value) for key, value in before.items()}
        root = torch.nn.Module()
        root.config = SimpleNamespace(
            num_hidden_layers=1, architectures=["GlmMoeDsaForCausalLM"]
        )
        root.mutate_weight_preload = lambda name: name
        root.stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        root.model = torch.nn.Module()
        root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = root.model.layers[0]
        layer.mlp = torch.nn.Module()
        shared = layer.mlp.shared_experts = torch.nn.Module()
        shared.gate_up_proj = torch.nn.Module()
        # The mapper follows the ordinary loader's input_dim for row parallel.
        shared.down_proj = type("RowParallelLinear", (torch.nn.Module,), {})()
        selection = slice(tp_rank * shard, (tp_rank + 1) * shard)
        shared.gate_up_proj.weight = torch.nn.Parameter(
            torch.cat([before["gate"][selection], before["up"][selection]]),
            requires_grad=False,
        )
        shared.down_proj.weight = torch.nn.Parameter(
            before["down"][:, selection].clone(), requires_grad=False
        )
        shared.gate_up_proj.weight.output_dim = 0
        shared.down_proj.weight.input_dim = 1
        for module in (shared.gate_up_proj, shared.down_proj):
            module.tp_rank, module.tp_size = tp_rank, 8
        inventory = {
            f"model.layers.0.mlp.shared_experts.{key}_proj.weight": {
                "dtype": "BF16",
                "shape": list(value.shape),
            }
            for key, value in before.items()
        }
        plan = layout.GpuDeltaLayout(root, inventory)
        for binding in plan.bindings:
            key = binding.name.split(".")[-2].removesuffix("_proj")
            mask = _bytes(before[key]) ^ _bytes(after[key])
            _apply_binding(binding, binding.selected_bytes(mask))
        plan.check_identity()
        torch.testing.assert_close(
            _bytes(shared.gate_up_proj.weight),
            _bytes(torch.cat([after["gate"][selection], after["up"][selection]])),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            _bytes(shared.down_proj.weight),
            _bytes(after["down"][:, selection]),
            rtol=0,
            atol=0,
        )
        shared.down_proj.weight = torch.nn.Parameter(
            shared.down_proj.weight.clone(), requires_grad=False
        )
        with self.assertRaisesRegex(RuntimeError, "parameter identity changed"):
            plan.check_identity()

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
        prepared = layout.PreparedDelta.__new__(layout.PreparedDelta)
        prepared.backend = SimpleNamespace(layout=plan)
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

    def test_derived_geometry_rejected_at_admission(self):
        for source in (torch.zeros(3), torch.zeros(2, dtype=torch.bfloat16)):
            with self.assertRaisesRegex(ValueError, "derived delta buffer geometry"):
                byte_layout.DerivedImage("consumer", torch.zeros(2), source)

    def test_identity_checks_share_geometry_but_preserve_each_live_binding(self):
        root = torch.nn.Module()
        root.weight = torch.nn.Parameter(torch.zeros(2, 3), requires_grad=False)
        root.tied = root.weight
        root.view = root.weight.detach().view(3, 2)
        root.wrapper = SimpleNamespace(config="unchanged")
        plan = layout.GpuDeltaLayout.__new__(layout.GpuDeltaLayout)
        plan._parameter_roots = [
            (root, name, tensor, byte_layout._tensor_identity(tensor))
            for name, tensor in root._parameters.items()
        ]
        plan._consumers = [
            byte_layout.ConsumerSnapshot(
                lambda: (root.weight, root.view, root.wrapper)
            ),
            byte_layout.ConsumerSnapshot(lambda: (root.tied, root.view)),
        ]
        plan._prepare_identity_checks()
        storage = torch.Tensor.untyped_storage
        reads = []

        def read_storage(tensor):
            reads.append(id(tensor))
            return storage(tensor)

        with patch.object(torch.Tensor, "untyped_storage", read_storage):
            plan.check_identity()
            plan.check_identity()
        self.assertEqual(reads, [id(root.weight), id(root.view)] * 2)

        # The tied parameter and consumer can be rebound to objects with exactly
        # the same storage and geometry; every live field still has to match.
        root.tied = torch.nn.Parameter(root.weight.detach(), requires_grad=False)
        with self.assertRaisesRegex(RuntimeError, "parameter identity changed"):
            plan.check_identity()
        root.tied = root.weight
        original = root.view
        root.view = original.view_as(original)
        with self.assertRaisesRegex(RuntimeError, "consumer storage changed"):
            plan.check_identity()
        root.view = original
        wrapper = root.wrapper
        root.wrapper = SimpleNamespace(config="unchanged")
        with self.assertRaisesRegex(RuntimeError, "consumer storage changed"):
            plan.check_identity()
        root.wrapper = wrapper

        # A shared storage pointer cannot hide distinct view geometry. Each new
        # check must also observe in-place changes after an earlier success.
        root.view.transpose_(0, 1)
        with self.assertRaisesRegex(RuntimeError, "consumer storage changed"):
            plan.check_identity()
        with self.assertRaisesRegex(RuntimeError, "consumer storage changed"):
            plan._consumers[0].check()
        root.view.transpose_(0, 1)
        plan.check_identity()
        plan._consumers[0].check()
        root.weight.transpose_(0, 1)
        with self.assertRaisesRegex(RuntimeError, "parameter identity changed"):
            plan.check_identity()

        # Never discard a different expectation for a repeated object. If a
        # consumer was captured after geometry changed, admission must fail.
        plan._consumers.append(byte_layout.ConsumerSnapshot(lambda: (root.weight,)))
        with self.assertRaisesRegex(RuntimeError, "changed during admission"):
            plan._prepare_identity_checks()

    def test_w4a16_calibration_is_static_but_weight_scales_remain_mutable(self):
        prefix = "model.layers.0.mlp.experts"
        layer = SimpleNamespace(
            moe_tp_size=1,
            use_presharded_weights=False,
            quant_method=_nvfp4_method(),
            moe_runner_config=SimpleNamespace(is_gated=True),
            _map_global_expert_id_to_local_expert_id=lambda expert: (
                expert if expert < 2 else -1
            ),
            w13_input_scale=torch.full((4, 2), 7.0),
            w2_input_scale=torch.full((4,), 11.0),
            w13_weight_scale_2=torch.ones(2, 2),
            w2_weight_scale_2=torch.ones(2),
        )
        plan = byte_layout.ParameterBindings.__new__(byte_layout.ParameterBindings)
        plan.modules, plan.moe_layers, plan.excluded = {prefix: layer}, {}, {}
        plan.moe_layouts = {}
        plan.moe_layers[prefix] = layer
        plan.moe_layouts[prefix] = FlashInferLayout("nvfp4", "cutedsl", True, "w4a16")
        meta = {"dtype": "F32", "shape": []}
        for projection in ("gate", "up", "down"):
            for expert in (0, 3):
                name = f"{prefix}.{expert}.{projection}_proj.input_scale"
                self.assertIsNone(plan.bind(name, meta, name))
                self.assertEqual(plan.excluded[name], "static activation calibration")
            name = f"{prefix}.0.{projection}_proj.weight_scale_2"
            binding = plan.bind(name, meta, name)
            after = torch.tensor(0.5)
            self.assertEqual(binding.encoding, "raw_bytes")
            torch._foreach_copy_([binding.storage[0]], [after])
            torch.testing.assert_close(_bytes(binding.storage[0]), _bytes(after))
        self.assertTrue(torch.all(layer.w13_input_scale == 7))
        self.assertTrue(torch.all(layer.w2_input_scale == 11))

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
        table, counts, gaps = layout._plan_decode(plans, entries, records)
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
        empty, counts, gaps = layout._plan_decode(plans[-1:], entries, records)
        self.assertEqual(empty.shape, (4, 0))
        self.assertEqual(counts, [0])
        self.assertEqual(gaps, [[(0, 24)]])
        empty, counts, gaps = layout._plan_decode([], entries, records)
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
        parse = layout.orjson.loads

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
            with patch.object(layout.orjson, "loads", side_effect=tracked_parse):
                with self.assertRaisesRegex(RuntimeError, "local preparation boundary"):
                    layout.PreparedDelta(
                        backend,
                        path,
                        layout.hashlib.sha256(content).hexdigest(),
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
            previous = layout._plan_layers(backend, bindings, entries)
            self.assertIs(layout._plan_layers(backend, bindings, entries), previous)
            reduced = layout._plan_layers(backend, bindings[1:], entries)
            self.assertIsNot(reduced, previous)
            self.assertEqual(reduced[3][0][0][0].name, names[1])
            for count, expected in ((1, [[0], [1], [2], [3]]), (2, [[0, 1], [2, 3]])):
                grouped = layout._plan_layers(backend, bindings, entries, count)
                self.assertEqual(
                    [
                        list(dict.fromkeys(b.layer for b, _, _ in plan[0]))
                        for plan in grouped
                    ],
                    [[None], [None], [None], *expected],
                )
                self.assertIs(
                    layout._plan_layers(backend, bindings, entries, count), grouped
                )

    def test_backend_reads_inventory_once_and_drains_all_streams_on_failure(self):
        fake_plan = SimpleNamespace(
            check_identity=lambda: None,
            rank_plan_digest="digest",
            bindings=[],
            excluded={},
        )
        fake_model = SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda", 0))])
        )
        with (
            patch.dict(layout.os.environ, GPU_DELTA_LAYERS_PER_BATCH="3"),
            patch.object(layout, "GpuDeltaLayout", return_value=fake_plan),
            patch("sglang.srt.weight_sync.gpu_delta.host.HostArena"),
            patch(
                "sglang.srt.weight_sync.gpu_delta.checkpoint.read_canonical_checkpoint_inventory",
                return_value={"weight": {"shape": [1], "dtype": "U8"}},
            ) as read_inventory,
        ):
            backend = layout.GpuDeltaBackend(
                SimpleNamespace(model=fake_model), {"engine_id": "test-engine"}
            )
            self.assertEqual(backend.decoders, {})
            self.assertIsNone(backend.apply_stream)
            self.assertIsNone(backend.de_stream)
            self.assertEqual(backend.layers_per_batch, 3)
            with patch.dict(layout.os.environ, GPU_DELTA_LAYERS_PER_BATCH="0"):
                backend.describe()
                backend.describe()
                self.assertEqual(backend.layers_per_batch, 3)
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    layout.GpuDeltaBackend(
                        SimpleNamespace(model=fake_model), {"engine_id": "invalid"}
                    )
            read_inventory.assert_called_once()
            backend.close()

        drains = []

        def decode_failure():
            drains.append("decode")
            raise RuntimeError("decode stream failed")

        prepared = layout.PreparedDelta.__new__(layout.PreparedDelta)
        prepared.de_stream = SimpleNamespace(synchronize=decode_failure)
        prepared.stream = SimpleNamespace(synchronize=lambda: drains.append("compute"))
        with self.assertRaisesRegex(RuntimeError, "decode stream failed"):
            prepared.close()
        self.assertEqual(drains, ["decode", "compute"])

    def test_glm_mapping_matches_loader_views_and_preserves_consumers(self):
        root = torch.nn.Module()
        root.config = SimpleNamespace(
            num_hidden_layers=1,
            architectures=["GlmMoeDsaForCausalLM"],
            q_lora_rank=3,
            kv_lora_rank=2,
            qk_rope_head_dim=2,
        )
        root.mutate_weight_preload = lambda name: name
        root.stacked_params_mapping = [
            ("fused_qkv_a_proj_with_mqa", "q_a_proj", 0),
            ("fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", 1),
        ]
        root.model = torch.nn.Module()
        root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = root.model.layers[0]
        layer.self_attn = torch.nn.Module()
        layer.self_attn.indexer = torch.nn.Module()
        norm = layer.self_attn.indexer.k_norm = torch.nn.LayerNorm(128)
        inventory = {
            f"model.layers.0.self_attn.indexer.k_norm.{key}": {
                "dtype": "BF16",
                "shape": [128],
            }
            for key in ("weight", "bias")
        }
        attn = layer.self_attn
        attn.fused_qkv_a_proj_with_mqa = torch.nn.Linear(8, 7, bias=False).bfloat16()
        attn.indexer.wk_weights_proj = torch.nn.Linear(8, 9, bias=False).bfloat16()
        attn.kv_b_proj = torch.nn.Linear(8, 20, bias=False).bfloat16()
        attn.qk_nope_head_dim, attn.v_head_dim = 4, 6
        key, value = attn.kv_b_proj.weight.unflatten(0, (2, 10)).split([4, 6], dim=1)
        attn.w_kc = key.transpose(1, 2).contiguous().transpose(1, 2).detach()
        attn.w_vc = value.contiguous().transpose(1, 2).detach()
        expected_views = {
            "q_a_proj.weight": attn.fused_qkv_a_proj_with_mqa.weight[:3],
            "kv_a_proj_with_mqa.weight": attn.fused_qkv_a_proj_with_mqa.weight[3:],
            "indexer.wk.weight": attn.indexer.wk_weights_proj.weight[:5],
            "indexer.weights_proj.weight": attn.indexer.wk_weights_proj.weight[5:],
            "kv_b_proj.weight": attn.kv_b_proj.weight,
        }
        prefix = "model.layers.0.self_attn."
        inventory.update(
            {
                prefix + name: {"dtype": "BF16", "shape": list(target.shape)}
                for name, target in expected_views.items()
            }
        )
        inventory.update(
            {
                "model.layers.1.weight": {"dtype": "BF16", "shape": [2, 2]},
                prefix + "rotary_emb.inv_freq": {"dtype": "F32", "shape": [2]},
            }
        )
        layer.mlp = torch.nn.Module()
        experts = layer.mlp.experts = torch.nn.Module()
        experts.moe_tp_size, experts.use_presharded_weights = 1, False
        experts.quant_method = _nvfp4_method()
        experts.moe_runner_config = SimpleNamespace(
            is_gated=True, activation="silu", gemm1_clamp_limit=None, swiglu_limit=None
        )
        experts.num_local_experts = experts.num_experts = 1
        experts.moe_ep_rank = 0
        experts.w13_input_scale_quant = experts.w2_input_scale_quant = torch.ones(1)
        experts._map_global_expert_id_to_local_expert_id = lambda expert: expert
        experts.w13_weight_scale_2 = torch.nn.Parameter(
            torch.ones(1, 2), requires_grad=False
        )
        experts.w2_weight_scale_2 = torch.nn.Parameter(
            torch.ones(1), requires_grad=False
        )
        experts.g1_alphas, experts.g1_alphas_up, experts.g2_alphas = [
            torch.ones(1) for _ in range(3)
        ]
        experts._cutedsl_wrapper = SimpleNamespace(quant_mode="w4a16")
        experts._cutedsl_scales = (torch.ones(1), torch.ones(1), torch.ones(1))
        experts._cutedsl_input_scale = torch.ones(1)
        experts.quant_config = SimpleNamespace(use_per_token_activation=False)
        inventory["model.layers.0.mlp.experts.0.gate_proj.weight_scale_2"] = {
            "dtype": "F32",
            "shape": [],
        }
        moe = SimpleNamespace(
            enable_eplb=False,
            elastic_ep_backend=None,
            init_expert_location="trivial",
            ep_num_redundant_experts=0,
            moe_a2a_backend="none",
            moe_runner_backend="flashinfer_cutedsl",
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.runtime_context": SimpleNamespace(
                        get_exec=lambda: SimpleNamespace(moe=moe)
                    )
                },
            ),
            patch.object(
                layout,
                "_require_fixed_moe_topology",
                wraps=layout._require_fixed_moe_topology,
            ) as require_topology,
        ):
            plan = layout.GpuDeltaLayout(root, inventory)
        require_topology.assert_called_once_with(moe)
        self.assertEqual(len(plan.excluded), 2)
        for binding in plan.bindings:
            relative = binding.name.removeprefix(prefix)
            if relative in expected_views:
                target = expected_views[relative]
                self.assertEqual(binding.shape, tuple(target.shape))
                self.assertEqual(binding.slices, [[0, n] for n in target.shape])
                self.assertEqual(binding.storage[0].data_ptr(), target.data_ptr())
                self.assertEqual(binding.storage[0].stride(), target.stride())
        for image, source in zip(plan.derived[-2:], (key, value.transpose(1, 2))):
            self.assertEqual(image.source.data_ptr(), source.data_ptr())
            self.assertEqual(image.source.stride(), source.stride())
        values = torch.tensor(
            [0.0, -0.0, float("inf"), float("nan"), 1.25, -3.5, 0.125, -16],
            dtype=torch.bfloat16,
        ).repeat(16)
        with torch.no_grad():
            for binding in plan.bindings:
                if ".k_norm." not in binding.name:
                    continue
                self.assertEqual(binding.encoding, "raw_bytes")
                target = getattr(norm, binding.name.rsplit(".", 1)[1])
                # An all-zero target is a replacement, not an omitted XOR.
                source = (
                    torch.zeros_like(values)
                    if binding.name.endswith("bias")
                    else values
                )
                target.fill_(7)
                pointer = target.data_ptr()
                expected = torch.empty_like(target)
                expected.copy_(source)  # The ordinary default_weight_loader.
                torch._foreach_copy_([binding.storage[0]], [source.to(target.dtype)])
                torch.testing.assert_close(
                    _bytes(target), _bytes(expected), rtol=0, atol=0
                )
                self.assertEqual(target.data_ptr(), pointer)
        plan.check_identity()
        original_key = attn.w_kc
        attn.w_kc = attn.w_kc.clone()
        with self.assertRaisesRegex(RuntimeError, "consumer storage changed"):
            plan.check_identity()
        attn.w_kc = original_key
        experts._cutedsl_scales = list(experts._cutedsl_scales)
        with self.assertRaisesRegex(RuntimeError, "consumer storage changed"):
            plan.check_identity()

    def test_movable_or_reordered_experts_rejected_before_plan(self):
        defaults = dict(
            enable_eplb=False,
            elastic_ep_backend=None,
            init_expert_location="trivial",
            ep_num_redundant_experts=0,
            moe_a2a_backend="none",
            moe_runner_backend="flashinfer_cutedsl",
        )
        layout._require_fixed_moe_topology(SimpleNamespace(**defaults))
        for key, value in (
            ("enable_eplb", True),
            ("elastic_ep_backend", "nixl"),
            ("init_expert_location", "custom.json"),
            ("ep_num_redundant_experts", 2),
            ("moe_a2a_backend", "deepep"),
            ("moe_runner_backend", "triton"),
        ):
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "fixed trivial EP"),
            ):
                layout._require_fixed_moe_topology(
                    SimpleNamespace(**(defaults | {key: value}))
                )

    def test_invalid_layout_geometry_rejected(self):
        prefix = "model.layers.0.mlp.experts"
        name = prefix + ".0.gate_proj.input_scale"
        for override in (
            {"moe_tp_size": 2},
            {"use_presharded_weights": True},
            {"quant_method": type("UnadmittedMoE", (), {"runner": None})()},
        ):
            layer = SimpleNamespace(
                **(
                    dict(
                        moe_tp_size=1,
                        use_presharded_weights=False,
                        quant_method=_nvfp4_method(),
                        moe_runner_config=SimpleNamespace(is_gated=True),
                    )
                    | override
                )
            )
            plan = byte_layout.ParameterBindings.__new__(byte_layout.ParameterBindings)
            plan.modules, plan.moe_layers, plan.excluded = {prefix: layer}, {}, {}
            plan.moe_layouts = {}
            with (
                self.subTest(override=override),
                self.assertRaises(ValueError),
            ):
                plan.bind(name, {"dtype": "F32", "shape": []}, name)
            self.assertFalse(plan.moe_layers)


if __name__ == "__main__":
    unittest.main()
