"""Byte-layout tests require PyTorch only; no FlashInfer import or GPU JIT."""

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# Keep this algebra test runnable in CPU development environments without
# importing serving-time CUDA dependencies through sglang's package root.
_path = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/weight_sync/gpu_delta_layout.py"
)
_spec = importlib.util.spec_from_file_location("gpu_delta_layout_under_test", _path)
layout = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = layout
_spec.loader.exec_module(layout)


class TestFlashInferDeltaLayout(unittest.TestCase):
    def test_projection_masks_commute_with_full_value_layout(self):
        for backend, group, up_first in (("cutedsl", 64, True), ("megamoe", 16, False)):
            for kind in ("weight", "scale"):
                cols = 19 if kind == "scale" else 128
                gate = torch.randint(256, (128, cols), dtype=torch.uint8)
                up = torch.randint(256, gate.shape, dtype=torch.uint8)
                new_gate = torch.randint(256, gate.shape, dtype=torch.uint8)
                new_up = torch.randint(256, gate.shape, dtype=torch.uint8)

                def full(gate, up):
                    fused = layout.interleave_gate_up_bytes(
                        gate, up, group_rows=group, up_first=up_first
                    )
                    return (
                        layout.swizzle_scale_bytes(fused) if kind == "scale" else fused
                    )

                current = full(gate, up)
                pointer = current.data_ptr()
                for projection, before, after in (
                    ("gate", gate, new_gate),
                    ("up", up, new_up),
                ):
                    mask = layout.flashinfer_delta_layout(
                        before ^ after,
                        dtype="nvfp4",
                        backend=backend,
                        kind=kind,
                        projection=projection,
                    )
                    current.bitwise_xor_(mask)
                self.assertEqual(pointer, current.data_ptr())
                torch.testing.assert_close(
                    current, full(new_gate, new_up), rtol=0, atol=0
                )

    def test_scale_swizzle_matches_physical_offsets_and_zero_padding(self):
        for rows, cols in ((17, 3), (128, 64), (256, 19)):
            source = torch.randint(256, (2, rows, cols), dtype=torch.uint8)
            swizzled = layout.swizzle_scale_bytes(source)
            row = torch.arange(rows)[:, None]
            col = torch.arange(cols)[None, :]
            # Physical order: row tile, column tile, row within 32,
            # row group within 128, column within 4. Padding stays zero.
            offsets = (
                ((row // 128 * ((cols + 3) // 4) + col // 4) * 32 + row % 32) * 4
                + row % 128 // 32
            ) * 4 + col % 4
            expected = torch.zeros_like(swizzled).reshape(2, -1)
            expected[:, offsets.flatten()] = source.reshape(2, -1)
            torch.testing.assert_close(
                swizzled, expected.reshape_as(swizzled), rtol=0, atol=0
            )

    def test_gate_up_order_is_backend_specific(self):
        gate = torch.ones((128, 8), dtype=torch.uint8)
        up = torch.full_like(gate, 2)
        cute = layout.interleave_gate_up_bytes(gate, up, group_rows=64, up_first=True)
        mega = layout.interleave_gate_up_bytes(gate, up, group_rows=16, up_first=False)
        self.assertTrue(torch.all(cute[:64] == 2))
        self.assertTrue(torch.all(cute[64:128] == 1))
        self.assertTrue(torch.all(mega[:16] == 1))
        self.assertTrue(torch.all(mega[16:32] == 2))

    def test_noncontiguous_gate_destination_preserves_up(self):
        destination = torch.randint(256, (256, 32), dtype=torch.uint8)
        old = destination.clone()
        gate = destination.reshape(2, 2, 64, 32)[:, 1]
        binding = layout._direct_binding(
            "gate", {"dtype": "U8", "shape": [128, 32]}, gate
        )
        mask = torch.randint(256, (128 * 32,), dtype=torch.uint8)
        pointer = destination.data_ptr()
        before = layout._bytes(gate).clone()
        binding.xor(mask)
        torch.testing.assert_close(layout._bytes(gate), before ^ mask)
        torch.testing.assert_close(
            destination.reshape(2, 2, 64, 32)[:, 0], old.reshape(2, 2, 64, 32)[:, 0]
        )
        self.assertEqual(pointer, destination.data_ptr())

    def test_full_tensor_then_tp_column_selection(self):
        full = torch.randint(256, (20, 32), dtype=torch.uint8)
        target = full[:, 8:16].clone()
        binding = layout._direct_binding(
            "weight",
            {"dtype": "U8", "shape": list(full.shape)},
            target,
            [[0, 20], [8, 16]],
        )
        mask = torch.randint(256, full.shape, dtype=torch.uint8)
        binding.xor(binding.selected_bytes(mask.reshape(-1)))
        torch.testing.assert_close(target, (full ^ mask)[:, 8:16])

    def test_bf16_masks_are_never_floating_point_values(self):
        # Exercise all byte values, including BF16 NaN bit patterns.
        mask = torch.arange(256, dtype=torch.uint8).repeat(2).reshape(16, 32)
        out = layout.flashinfer_delta_layout(
            mask, dtype="bf16", backend="cutedsl", kind="weight"
        )
        self.assertEqual(out.data_ptr(), mask.data_ptr())
        with self.assertRaises(ValueError):
            layout.flashinfer_delta_layout(
                mask.view(torch.bfloat16),
                dtype="bf16",
                backend="cutedsl",
                kind="weight",
            )

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
        root.config = SimpleNamespace(num_hidden_layers=1)
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
        root._gpu_delta_load_generation = 1
        root._gpu_delta_canonical_inventory = {
            f"model.layers.0.mlp.shared_experts.{key}_proj.weight": {
                "dtype": "BF16",
                "shape": list(value.shape),
            }
            for key, value in before.items()
        }
        plan = layout.GpuDeltaLayout(root)
        for binding in plan.bindings:
            key = binding.name.split(".")[-2].removesuffix("_proj")
            mask = layout._bytes(before[key]) ^ layout._bytes(after[key])
            binding.xor(binding.selected_bytes(mask))
        plan.check_identity()
        torch.testing.assert_close(
            layout._bytes(shared.gate_up_proj.weight),
            layout._bytes(
                torch.cat([after["gate"][selection], after["up"][selection]])
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            layout._bytes(shared.down_proj.weight),
            layout._bytes(after["down"][:, selection]),
            rtol=0,
            atol=0,
        )

    def test_scalar_metadata_direct_targets_preserve_exact_bits(self):
        for old, new in ((1.0, 0.25), (-0.0, 3.5)):
            current = torch.tensor(old, dtype=torch.float32)
            expected = torch.tensor(new, dtype=torch.float32)
            binding = layout._direct_binding(
                "weight_scale_2", {"dtype": "F32", "shape": []}, current
            )
            pointer = current.data_ptr()
            self.assertEqual(binding.encoding, "raw_bytes")
            self.assertIsNone(binding.xor)
            torch._foreach_copy_([binding.storage[0]], [expected])
            torch.testing.assert_close(
                current.reshape(1).view(torch.uint8),
                expected.reshape(1).view(torch.uint8),
            )
            self.assertEqual(current.data_ptr(), pointer)

    def test_derived_refresh_preserves_rank_and_decoder_error_gate(self):
        for shape in ((), (2,), (2, 2)):
            for error in (0, 1):
                with self.subTest(shape=shape, decoder_error=error):
                    source = torch.full(shape, 0.25)
                    target = torch.full(shape, -0.0)
                    before, pointer = target.clone(), target.data_ptr()
                    other_source = torch.full((3,), 0.5)
                    other_target = torch.full((3,), -1.0)
                    other_before = other_target.clone()
                    other_pointer = other_target.data_ptr()
                    prepared = layout.PreparedDelta.__new__(layout.PreparedDelta)
                    prepared.backend = SimpleNamespace(
                        layout=SimpleNamespace(check_identity=lambda: None)
                    )
                    prepared.device = torch.device("cpu")
                    prepared.stream = SimpleNamespace(
                        wait_event=lambda _: None, wait_stream=lambda _: None
                    )
                    prepared.ready = object()
                    prepared.applied = prepared.timing_enabled = False
                    prepared.raw_copies, prepared.units, prepared.raw_units = {}, [], []
                    prepared.derived = [
                        layout.DerivedImage(
                            "consumer", target, lambda source=source: source
                        ),
                        layout.DerivedImage(
                            "other", other_target, lambda value=other_source: value
                        ),
                    ]
                    prepared.error = torch.tensor([error], dtype=torch.int32)
                    prepared.timings, prepared.h2d_bytes = {}, 0
                    prepared.manifest = {"target_version": 1}
                    # Exercise apply's actual tensor logic with CPU tensors;
                    # only the CUDA scheduling boundary is stubbed here.
                    with (
                        patch.object(torch.cuda, "device", return_value=nullcontext()),
                        patch.object(torch.cuda, "stream", return_value=nullcontext()),
                        patch.object(torch.cuda, "default_stream", return_value=None),
                        patch.object(
                            torch.cuda,
                            "Event",
                            return_value=SimpleNamespace(
                                record=lambda _: None, synchronize=lambda: None
                            ),
                        ),
                    ):
                        if error:
                            with self.assertRaisesRegex(RuntimeError, "poisoned"):
                                prepared.apply()
                        else:
                            self.assertTrue(prepared.apply()["applied"])
                    torch.testing.assert_close(
                        layout._bytes(target),
                        layout._bytes(before if error else source),
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(target.shape, shape)
                    self.assertEqual(target.data_ptr(), pointer)
                    torch.testing.assert_close(
                        other_target, other_before if error else other_source
                    )
                    self.assertEqual(other_target.data_ptr(), other_pointer)

    def test_w4a16_calibration_is_static_but_weight_scales_remain_mutable(self):
        prefix = "model.layers.0.mlp.experts"
        layer = SimpleNamespace(
            moe_tp_size=1,
            quant_method=SimpleNamespace(_is_cutedsl_v2_standard=True),
            moe_runner_config=SimpleNamespace(is_gated=True),
            _map_global_expert_id_to_local_expert_id=lambda expert: (
                expert if expert < 2 else -1
            ),
            w13_input_scale=torch.full((4, 2), 7.0),
            w2_input_scale=torch.full((4,), 11.0),
            w13_weight_scale_2=torch.ones(2, 2),
            w2_weight_scale_2=torch.ones(2),
        )
        plan = layout.GpuDeltaLayout.__new__(layout.GpuDeltaLayout)
        plan.model = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=1),
            mutate_weight_preload=lambda name: name,
        )
        plan._modules, plan._moe_layers, plan.excluded = {prefix: layer}, {}, {}
        meta = {"dtype": "F32", "shape": []}
        for projection in ("gate", "up", "down"):
            for expert in (0, 3):
                name = f"{prefix}.{expert}.{projection}_proj.input_scale"
                self.assertIsNone(plan._bind(name, meta))
                self.assertEqual(
                    plan.excluded[name], "static W4A16 activation calibration"
                )
            name = f"{prefix}.0.{projection}_proj.weight_scale_2"
            binding = plan._bind(name, meta)
            after = torch.tensor(0.5)
            self.assertEqual(binding.encoding, "raw_bytes")
            torch._foreach_copy_([binding.storage[0]], [after])
            torch.testing.assert_close(
                layout._bytes(binding.storage[0]), layout._bytes(after)
            )
        self.assertTrue(torch.all(layer.w13_input_scale == 7))
        self.assertTrue(torch.all(layer.w2_input_scale == 11))

    def test_publication_cannot_reintroduce_static_calibration(self):
        name = "model.layers.0.mlp.experts.0.gate_proj.input_scale"
        metadata = dict(
            stream_id="test", base_version=0, target_version=1, plan_digest="plan"
        )
        manifest = dict(
            protocol_version=4,
            codec="snappy-zstd",
            frame_bytes=1 << 20,
            tensors=[{"name": name}],
            **metadata,
        )
        content = json.dumps(manifest).encode()
        backend = SimpleNamespace(
            codec="snappy-zstd",
            device=torch.device("cpu"),
            layout=SimpleNamespace(
                inventory={name: {"dtype": "F32", "shape": []}},
                excluded={name: "static W4A16 activation calibration"},
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_bytes(content)
            with (
                patch.object(torch.cuda, "Stream", return_value=object()),
                self.assertRaisesRegex(ValueError, "unadmitted tensor"),
            ):
                layout.PreparedDelta(
                    backend,
                    path,
                    layout.hashlib.sha256(content).hexdigest(),
                    metadata,
                )

    def test_snappy_streaming_unwraps_only_local_tensors_and_reuses_scratch(self):
        import zstandard as zstd

        targets = [torch.zeros(2, size, dtype=torch.uint8) for size in (4, 6)]
        local = [
            layout._direct_binding(str(i), {"dtype": "U8", "shape": list(t.shape)}, t)
            for i, t in enumerate(targets)
        ]
        foreign = layout._direct_binding(
            "foreign",
            {"dtype": "U8", "shape": [3, 4]},
            torch.zeros(3, 4, dtype=torch.uint8),
        )
        definitions = [b.describe() for b in (*local, foreign)]
        blobs, entries, records = {}, [], []
        for binding in (*local, foreign):
            name, size = binding.name, binding.storage[0].numel()
            # A valid Snappy literal block expands these tiny matrix frames.
            # Only the CUDA decoder is mocked below; CPU Zstd and file checks run.
            inner = bytes([size, (size - 1) << 2]) + bytes(range(size))
            blob = zstd.ZstdCompressor().compress(inner)
            if name == "foreign":
                blob = b"not-a-zstd-frame"  # Foreign EP data is never unwrapped.
            blobs[name] = blob
            file = name + ".bin"
            entries.append(
                binding.describe()
                | {
                    "byte_order": "little",
                    "nbytes": size,
                    "outer": dict(
                        file=file,
                        encoded_offset=0,
                        encoded_bytes=len(blob),
                        decoded_bytes=len(inner),
                        frames=[
                            dict(
                                encoded_offset=0,
                                encoded_bytes=len(blob),
                                decoded_offset=0,
                                decoded_bytes=len(inner),
                            )
                        ],
                    ),
                    "frames": [
                        dict(
                            encoded_offset=0,
                            encoded_bytes=len(inner),
                            decoded_offset=0,
                            decoded_bytes=size,
                        )
                    ],
                }
            )
            records.append(
                dict(
                    name=file,
                    nbytes=len(blob),
                    sha256=layout.hashlib.sha256(blob).hexdigest(),
                )
            )
        metadata = dict(
            stream_id="test",
            base_version=0,
            target_version=1,
            plan_digest=layout._digest(definitions),
        )
        manifest = dict(
            protocol_version=4,
            codec="snappy-zstd",
            frame_bytes=1 << 20,
            tensors=entries,
            files=records,
            **metadata,
        )
        backend = SimpleNamespace(
            codec="snappy-zstd",
            device=torch.device("cpu"),
            layout=SimpleNamespace(
                bindings=[local[0], local[0], local[1]],
                inventory={
                    b.name: {"dtype": b.dtype, "shape": list(b.shape)}
                    for b in (*local, foreign)
                },
                excluded={"foreign": "expert owned by another EP rank"},
                derived=[],
            ),
        )

        class CpuLiteralDecoder:
            # Explicit CPU test substitute; this does not qualify nvCOMP/CUDA.
            def __init__(self, device):
                self.device = device

            def allocate_workspace(self, batches):
                return object()

            def prepare(self, frames, encoded, decoded, workspace, stream):
                def enqueue(_):
                    for frame in frames:
                        data = encoded[
                            frame.input_offset : frame.input_offset
                            + frame.encoded_bytes
                        ]
                        assert data[0] == frame.decoded_bytes
                        assert data[1] == (frame.decoded_bytes - 1) << 2
                        decoded[
                            frame.output_offset : frame.output_offset
                            + frame.decoded_bytes
                        ].copy_(data[2:])

                sizes = torch.tensor([f.decoded_bytes for f in frames])
                return SimpleNamespace(
                    enqueue=enqueue,
                    statuses=torch.zeros(len(frames)),
                    actual_sizes=sizes,
                    metadata={2: sizes},
                )

        empty = torch.empty

        def unpinned(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return empty(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for name, blob in blobs.items():
                path.with_name(name + ".bin").write_bytes(blob)
            content = json.dumps(manifest).encode()
            path.write_bytes(content)
            with (
                patch.object(torch, "empty", side_effect=unpinned),
                patch.object(torch.cuda, "Stream", return_value=object()),
                patch.object(torch.cuda, "stream", return_value=nullcontext()),
                patch.object(
                    torch.cuda,
                    "Event",
                    return_value=SimpleNamespace(
                        record=lambda _: None, synchronize=lambda: None
                    ),
                ),
                patch(
                    "sglang.srt.weight_sync.gpu_delta_codec.NvcompDecoder",
                    CpuLiteralDecoder,
                ),
            ):
                prepared = layout.PreparedDelta(
                    backend, path, layout.hashlib.sha256(content).hexdigest(), metadata
                )
                self.assertFalse(prepared.host_files)
                self.assertEqual(prepared.timings["host_outer_zstd_tensors"], 2)
                self.assertEqual(prepared.timings["host_outer_zstd_frames"], 2)
                self.assertEqual(
                    prepared.units[0].pinned.data_ptr(),
                    prepared.units[1].pinned.data_ptr(),
                )
                self.assertEqual(prepared.encoded.numel(), 14)
                self.assertEqual(prepared.decoded.numel(), 12)
                self.assertEqual(prepared.h2d_bytes, 34)
                self.assertTrue(all(torch.count_nonzero(t) == 0 for t in targets))
                pointer = prepared.encoded.data_ptr()
                for unit, target in [
                    (prepared.units[0], targets[0]),
                    (prepared.units[2], targets[1]),
                ]:
                    prepared._apply_tensor(unit)
                    torch.testing.assert_close(
                        target,
                        torch.arange(target.numel(), dtype=torch.uint8).reshape(
                            target.shape
                        ),
                    )
                    self.assertEqual(prepared.encoded.data_ptr(), pointer)
                # Immutable-file corruption cannot reach any model write.
                targets[0].zero_()
                path.with_name("0.bin").write_bytes(
                    blobs["0"][:-1] + bytes([blobs["0"][-1] ^ 1])
                )
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    layout.PreparedDelta(
                        backend,
                        path,
                        layout.hashlib.sha256(content).hexdigest(),
                        metadata,
                    )
                self.assertTrue(torch.all(targets[0] == 0))

    def test_backend_freezes_configured_codec_in_advertised_plan(self):
        fake_plan = SimpleNamespace(
            check_identity=lambda: None,
            rank_plan_digest="digest",
            bindings=[],
            excluded={},
        )
        fake_model = SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda", 0))])
        )
        runtime = SimpleNamespace(get_exec=lambda: SimpleNamespace(moe=object()))
        with (
            patch.dict(sys.modules, {"sglang.srt.runtime_context": runtime}),
            patch.object(layout, "_require_fixed_moe_topology"),
            patch.object(layout, "GpuDeltaLayout", return_value=fake_plan),
            patch.dict("os.environ", {"WEIGHT_DELTA_CODEC": "snappy-zstd"}),
        ):
            backend = layout.GpuDeltaBackend(SimpleNamespace(model=fake_model), {})
            with patch.dict("os.environ", {"WEIGHT_DELTA_CODEC": "invalid"}):
                self.assertEqual(backend.describe()["codec"], "snappy-zstd")
                with self.assertRaisesRegex(ValueError, "WEIGHT_DELTA_CODEC"):
                    layout.GpuDeltaBackend(SimpleNamespace(model=fake_model), {})

    def test_indexer_norm_replacement_matches_fp32_loader_and_preserves_pointer(self):
        root = torch.nn.Module()
        root.config = SimpleNamespace(num_hidden_layers=1)
        root.mutate_weight_preload = lambda name: name
        root.stacked_params_mapping = []
        root.model = torch.nn.Module()
        root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = root.model.layers[0]
        layer.self_attn = torch.nn.Module()
        layer.self_attn.indexer = torch.nn.Module()
        norm = layer.self_attn.indexer.k_norm = torch.nn.LayerNorm(128)
        root._gpu_delta_load_generation = 1
        root._gpu_delta_canonical_inventory = {
            f"model.layers.0.self_attn.indexer.k_norm.{key}": {
                "dtype": "BF16",
                "shape": [128],
            }
            for key in ("weight", "bias")
        }
        plan = layout.GpuDeltaLayout(root)
        values = torch.tensor(
            [0.0, -0.0, float("inf"), float("nan"), 1.25, -3.5, 0.125, -16],
            dtype=torch.bfloat16,
        ).repeat(16)
        with torch.no_grad():
            for binding in plan.bindings:
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
                    layout._bytes(target), layout._bytes(expected), rtol=0, atol=0
                )
                self.assertEqual(target.data_ptr(), pointer)
        plan.check_identity()

    def test_raw_vectors_scalars_prepare_once_and_copy_in_place(self):
        vector = torch.full((4,), 7, dtype=torch.bfloat16)
        scalar = torch.tensor(-0.0)
        indexer = torch.full((2,), 9, dtype=torch.float32)
        unchanged = torch.ones(3)
        odd = torch.zeros(3, dtype=torch.uint8)
        bindings = [
            layout._direct_binding("0", {"dtype": "U8", "shape": [3]}, odd),
            layout._direct_binding(
                "a", {"dtype": "BF16", "shape": [8]}, vector, [[2, 6]]
            ),
            layout._direct_binding("b", {"dtype": "F32", "shape": []}, scalar),
            layout._indexer_norm_binding("c", {"dtype": "BF16", "shape": [2]}, indexer),
            layout._direct_binding("d", {"dtype": "F32", "shape": [3]}, unchanged),
        ]
        values = [
            torch.arange(3, dtype=torch.uint8),
            torch.arange(8).bfloat16(),
            torch.tensor(0.25),
            torch.tensor([0.5, -2]).bfloat16(),
        ]
        blob, entries = bytearray(), []
        for binding, value in zip(bindings, values + [None]):
            size = layout.math.prod(binding.shape) * layout._itemsize(binding.dtype)
            entry = binding.describe() | {
                "byte_order": "little",
                "nbytes": size,
                "changed_bytes": size if value is not None else 0,
                "frames": [],
            }
            if value is not None:
                data = bytes(layout._bytes(value).numpy())
                entry["raw"] = {
                    "file": "owner.bin",
                    "encoded_offset": len(blob),
                    "encoded_bytes": size,
                }
                blob.extend(data)
            entries.append(entry)
        metadata = dict(
            stream_id="raw",
            base_version=0,
            target_version=1,
            plan_digest=layout._digest([b.describe() for b in bindings]),
        )
        manifest = dict(
            protocol_version=4,
            codec="snappy-zstd",
            frame_bytes=1 << 20,
            tensors=entries,
            files=[
                {
                    "name": "owner.bin",
                    "nbytes": len(blob),
                    "sha256": layout.hashlib.sha256(blob).hexdigest(),
                }
            ],
            **metadata,
        )
        backend = SimpleNamespace(
            codec="snappy-zstd",
            device=torch.device("cpu"),
            layout=SimpleNamespace(
                bindings=bindings,
                excluded={},
                derived=[],
                inventory={
                    b.name: {"dtype": b.dtype, "shape": list(b.shape)} for b in bindings
                },
            ),
        )
        empty = torch.empty

        def unpinned(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return empty(*args, **kwargs)

        pointers = [b.storage[0].data_ptr() for b in bindings]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.with_name("owner.bin").write_bytes(blob)
            content = json.dumps(manifest).encode()
            path.write_bytes(content)
            with (
                patch.object(torch, "empty", side_effect=unpinned),
                patch.object(torch.cuda, "Stream", return_value=object()),
                patch.object(torch.cuda, "stream", return_value=nullcontext()),
                patch.object(
                    torch.cuda,
                    "Event",
                    return_value=SimpleNamespace(
                        record=lambda _: None, synchronize=lambda: None
                    ),
                ),
            ):
                prepared = layout.PreparedDelta(
                    backend, path, layout.hashlib.sha256(content).hexdigest(), metadata
                )
            self.assertFalse(prepared.units)
            self.assertIsNone(prepared.decoder)
            self.assertEqual(prepared.timings["raw_tensors"], 4)
            self.assertEqual(prepared.timings["raw_bytes"], len(blob))
            self.assertEqual(prepared.h2d_bytes, 36)  # Includes dtype alignment gaps.
            self.assertTrue(torch.all(vector == 7))
            self.assertTrue(torch.all(indexer == 9))
            for targets, sources in prepared.raw_copies.values():
                torch._foreach_copy_(targets, sources)
            torch.testing.assert_close(odd, values[0])
            torch.testing.assert_close(vector, values[1][2:6])
            torch.testing.assert_close(scalar, values[2])
            torch.testing.assert_close(indexer, values[3].float())
            torch.testing.assert_close(unchanged, torch.ones(3))
            self.assertEqual(pointers, [b.storage[0].data_ptr() for b in bindings])

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
        with self.assertRaises(ValueError):
            layout.flashinfer_delta_layout(
                torch.zeros(16, 4, dtype=torch.uint8),
                dtype="nvfp4",
                backend="cutedsl",
                kind="scale",
                projection="gate",
            )


if __name__ == "__main__":
    unittest.main()
