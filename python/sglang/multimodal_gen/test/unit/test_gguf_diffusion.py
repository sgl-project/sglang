"""CPU unit tests for the diffusion GGUF load path.

These cover the two things that must hold before any GPU run is meaningful:
the header-derived tensor layout (which is what lets the generic weight loader
work unchanged) and the quant-method selection per layer.
"""

import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from gguf import GGMLQuantizationType as WeightType

from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.gguf import (
    GGUFConfig,
    GGUFLinearMethod,
)
from sglang.multimodal_gen.runtime.loader.gguf_weights import (
    GGUFTensorMeta,
    gguf_weights_iterator,
    names_gguf_checkpoint,
    read_gguf_tensor_meta,
)
from sglang.srt.layers.quantization.gguf import UNQUANTIZED_TYPES
from sglang.srt.utils.hf_transformers import check_gguf_file

_F32 = WeightType.F32
_BF16 = WeightType.BF16
_Q4_K = WeightType.Q4_K

_Q4_K_BLOCK, _Q4_K_TYPE_SIZE = 256, 144


def _kv_string(key: str, value: str, bo: str = "<") -> bytes:
    out = struct.pack(f"{bo}Q", len(key)) + key.encode()
    out += struct.pack(f"{bo}I", 8)  # value type: string
    out += struct.pack(f"{bo}Q", len(value)) + value.encode()
    return out


def _write_gguf(
    path: Path,
    tensors: list[tuple[str, list[int], int, bytes]],
    byte_order: str = "<",
) -> None:
    """Write a minimal GGUF v3 file containing ``tensors``.

    Each entry is ``(name, ne_dims, ggml_type, payload)`` where ``ne_dims`` is in
    GGUF order (fastest-varying first). ``byte_order`` is a struct prefix; pass
    ``">"`` on a little-endian host to produce a file gguf-py reports as
    swapped.
    """
    bo = byte_order
    header = b"GGUF" + struct.pack(f"{bo}I", 3)
    header += struct.pack(f"{bo}QQ", len(tensors), 1)
    header += _kv_string("general.architecture", "test", bo)

    # Tensor info blocks, then padded data.
    infos = b""
    offset = 0
    alignment = 32
    payloads = []
    for name, dims, ggml_type, payload in tensors:
        infos += struct.pack(f"{bo}Q", len(name)) + name.encode()
        infos += struct.pack(f"{bo}I", len(dims))
        infos += b"".join(struct.pack(f"{bo}Q", d) for d in dims)
        infos += struct.pack(f"{bo}I", ggml_type)
        infos += struct.pack(f"{bo}Q", offset)
        payloads.append(payload)
        padded = (len(payload) + alignment - 1) // alignment * alignment
        offset += padded

    body = header + infos
    pad = (alignment - len(body) % alignment) % alignment
    body += b"\0" * pad
    for payload in payloads:
        padded = (len(payload) + alignment - 1) // alignment * alignment
        body += payload + b"\0" * (padded - len(payload))
    path.write_bytes(body)


class TestGGUFTensorMeta(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_quantized_layout_is_packed_rows(self):
        """A quantized weight is registered as (out_features, row_bytes) uint8."""
        out_features, in_features = 4, 512
        row_bytes = in_features // _Q4_K_BLOCK * _Q4_K_TYPE_SIZE
        payload = bytes(out_features * row_bytes)
        path = self.tmp / "q.gguf"
        # GGUF stores ne fastest-varying first: [in, out].
        _write_gguf(path, [("w.weight", [in_features, out_features], _Q4_K, payload)])

        meta = read_gguf_tensor_meta(str(path))["w.weight"]
        self.assertEqual(meta.ggml_type, _Q4_K)
        self.assertTrue(meta.is_quantized)
        # logical_shape is torch order, i.e. reversed from the file.
        self.assertEqual(meta.logical_shape, (out_features, in_features))
        self.assertEqual(meta.stored_shape, (out_features, row_bytes))
        self.assertEqual(meta.stored_dtype, torch.uint8)
        # The layer registers `qweight`, so that is what the iterator must yield.
        self.assertEqual(meta.param_name, "w.qweight")

    def test_unquantized_layout_matches_logical_shape(self):
        out_features, in_features = 3, 8
        payload = np.zeros((out_features, in_features), dtype=np.float32).tobytes()
        path = self.tmp / "f32.gguf"
        _write_gguf(path, [("w.weight", [in_features, out_features], _F32, payload)])

        meta = read_gguf_tensor_meta(str(path))["w.weight"]
        self.assertFalse(meta.is_quantized)
        self.assertEqual(meta.logical_shape, (out_features, in_features))
        self.assertEqual(meta.stored_shape, (out_features, in_features))
        self.assertEqual(meta.stored_dtype, torch.float32)
        self.assertEqual(meta.param_name, "w.weight")

    def test_pruned_adaln_curve_shape_is_available_before_model_init(self):
        grid, width = 1025, 8
        path = self.tmp / "pruned.gguf"
        _write_gguf(
            path,
            [
                (
                    "adaln_t_table",
                    [width, grid],
                    _F32,
                    bytes(grid * width * 4),
                )
            ],
        )

        metadata = read_gguf_tensor_meta(str(path))["adaln_t_table"]

        self.assertEqual(metadata.logical_shape, (grid, width))
        self.assertFalse(metadata.is_quantized)

    def test_non_block_aligned_inner_dim_is_rejected(self):
        """A row that is not a whole number of blocks must not load.

        gguf-py validates this first, so the message comes from there rather
        than from read_gguf_tensor_meta's own guard; either way the invariant
        row_bytes depends on is enforced.
        """
        # 100 is not a multiple of the Q4_K block size (256).
        path = self.tmp / "bad.gguf"
        _write_gguf(path, [("w.weight", [100, 2], _Q4_K, bytes(64))])
        with self.assertRaisesRegex(ValueError, "not a multiple of.*block size"):
            read_gguf_tensor_meta(str(path))

    def test_super_block_element_count_is_enforced(self):
        """IQ4_NL has 32-element blocks but a 256-element super-block kernel with
        no bounds check, so a tensor that is not a whole number of super blocks
        would read or write past the buffer."""
        _IQ4_NL, block, type_size = 20, 32, 18
        # in=1440 is a multiple of 32 (so the row check passes) and out=4 makes
        # numel=5760, which is not a multiple of 256.
        out_features, in_features = 4, 1440
        row_bytes = in_features // block * type_size
        path = self.tmp / "iq4nl.gguf"
        _write_gguf(
            path,
            [
                (
                    "w.weight",
                    [in_features, out_features],
                    _IQ4_NL,
                    bytes(row_bytes * out_features),
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "super\\s+blocks"):
            read_gguf_tensor_meta(str(path))

    def test_standard_quant_type_does_not_require_super_block_alignment(self):
        """Q4_0 has native MMVQ/MMQ kernels, so 256 must not be required."""
        _Q4_0, block, type_size = 2, 32, 18
        out_features, in_features = 1, 32  # numel 32: not a super block
        row_bytes = in_features // block * type_size
        path = self.tmp / "q40.gguf"
        _write_gguf(
            path,
            [
                (
                    "w.weight",
                    [in_features, out_features],
                    _Q4_0,
                    bytes(row_bytes * out_features),
                )
            ],
        )
        meta = read_gguf_tensor_meta(str(path))["w.weight"]
        self.assertEqual(meta.stored_shape, (out_features, row_bytes))

    def test_bf16_is_reinterpreted_not_cast(self):
        """gguf-py returns BF16 as raw bytes; a cast would corrupt the values."""
        values = torch.tensor([1.0, -2.5, 3.75], dtype=torch.bfloat16)
        payload = values.view(torch.uint8).numpy().tobytes()
        path = self.tmp / "bf16.gguf"
        _write_gguf(path, [("norm.weight", [3], _BF16, payload)])

        meta = read_gguf_tensor_meta(str(path))
        loaded = dict(gguf_weights_iterator(str(path), meta))["norm.weight"]
        self.assertEqual(loaded.dtype, torch.bfloat16)
        torch.testing.assert_close(loaded, values)

    def test_iterator_yields_stored_shapes(self):
        out_features, in_features = 2, 256
        row_bytes = in_features // _Q4_K_BLOCK * _Q4_K_TYPE_SIZE
        payload = bytes(range(row_bytes)) * out_features
        path = self.tmp / "iter.gguf"
        _write_gguf(path, [("w.weight", [in_features, out_features], _Q4_K, payload)])

        meta = read_gguf_tensor_meta(str(path))
        loaded = dict(gguf_weights_iterator(str(path), meta))["w.qweight"]
        self.assertEqual(loaded.dtype, torch.uint8)
        self.assertEqual(tuple(loaded.shape), (out_features, row_bytes))

    def test_swapped_endian_file_is_rejected(self):
        """A quantized block embeds its scales, so a swapped file cannot just be
        byte-swapped whole; refuse it instead of dequantizing garbage."""
        if sys.byteorder != "little":
            self.skipTest("test builds a big-endian file to be the swapped one")
        path = self.tmp / "be.gguf"
        _write_gguf(path, [("w.weight", [4], _F32, bytes(16))], byte_order=">")

        with self.assertRaisesRegex(ValueError, "opposite byte order"):
            read_gguf_tensor_meta(str(path))

    def test_truncated_file_names_the_likely_cause(self):
        """An interrupted download is common for a multi-GiB file; the bare
        reshape error gguf-py raises reads like a layout bug instead."""
        row_bytes = 256 // _Q4_K_BLOCK * _Q4_K_TYPE_SIZE
        full = self.tmp / "full.gguf"
        _write_gguf(full, [("w.weight", [256, 4], _Q4_K, bytes(row_bytes * 4))])
        # Keep the header, drop most of the tensor data.
        cut = self.tmp / "cut.gguf"
        cut.write_bytes(full.read_bytes()[: -row_bytes * 3])

        with self.assertRaisesRegex(ValueError, "incomplete or\\s+corrupt download"):
            read_gguf_tensor_meta(str(cut))

    def test_is_gguf_file_detects_by_magic(self):
        path = self.tmp / "no-suffix.bin"
        _write_gguf(path, [("w.weight", [4], _F32, bytes(16))])
        self.assertTrue(check_gguf_file(str(path)))
        other = self.tmp / "other.bin"
        other.write_bytes(b"NOTGGUF")
        self.assertFalse(check_gguf_file(str(other)))
        self.assertFalse(check_gguf_file(str(self.tmp / "missing.gguf")))

    def test_quantized_non_linear_tensor_is_rejected(self):
        path = self.tmp / "bad-norm.gguf"
        _write_gguf(path, [("norm.weight", [256], _Q4_K, bytes(144))])

        with self.assertRaisesRegex(ValueError, "only for 2D linear"):
            read_gguf_tensor_meta(str(path))


class TestGGUFQuantMethodSelection(unittest.TestCase):
    def _config(self, **metas):
        return GGUFConfig(gguf_file="/dev/null", tensor_meta=dict(metas))

    def _meta(self, ggml_type, out_features, in_features, stored_shape=None):
        return GGUFTensorMeta(
            ggml_type=ggml_type,
            logical_shape=(out_features, in_features),
            stored_shape=stored_shape or (out_features, in_features),
            stored_dtype=(
                torch.float32 if ggml_type in UNQUANTIZED_TYPES else torch.uint8
            ),
            param_name=("w.weight" if ggml_type in UNQUANTIZED_TYPES else "w.qweight"),
        )

    def test_quantized_layer_gets_gguf_method(self):
        config = self._config(**{"w.weight": self._meta(_Q4_K, 4, 512, (4, 288))})
        layer = ReplicatedLinear(512, 4, bias=False, quant_config=config, prefix="w")
        self.assertIsInstance(layer.quant_method, GGUFLinearMethod)
        self.assertEqual(layer.qweight.dtype, torch.uint8)
        self.assertEqual(tuple(layer.qweight.shape), (4, 288))
        self.assertEqual(layer.quant_method.weight_type, _Q4_K)

    def test_unquantized_layer_falls_back(self):
        """H3 keeps its FP32 projections unquantized inside the same file."""
        config = self._config(**{"w.weight": self._meta(_F32, 4, 8)})
        layer = ReplicatedLinear(8, 4, bias=False, quant_config=config, prefix="w")
        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)

    def test_missing_tensor_fails_fast(self):
        config = self._config()
        with self.assertRaisesRegex(ValueError, "no weight in the GGUF checkpoint"):
            ReplicatedLinear(8, 4, bias=False, quant_config=config, prefix="absent")

    def test_shape_mismatch_fails_fast(self):
        config = self._config(**{"w.weight": self._meta(_Q4_K, 8, 512, (8, 288))})
        with self.assertRaisesRegex(ValueError, "logical shape"):
            ReplicatedLinear(512, 4, bias=False, quant_config=config, prefix="w")

    @patch(
        "sglang.multimodal_gen.runtime.layers.quantization.gguf.dequantize_gguf_weight"
    )
    def test_apply_reuses_srt_dequantization(self, dequantize):
        config = self._config(**{"w.weight": self._meta(_Q4_K, 4, 512, (4, 288))})
        layer = ReplicatedLinear(512, 4, bias=False, quant_config=config, prefix="w")
        dequantize.return_value = torch.ones(4, 512)

        output, _ = layer(torch.ones(2, 4, 512))

        dequantize.assert_called_once_with(layer.qweight, _Q4_K, torch.float32)
        self.assertEqual(tuple(output.shape), (2, 4, 4))
        torch.testing.assert_close(output, torch.full_like(output, 512.0))


class TestGGUFTensorParallelLoading(unittest.TestCase):
    def setUp(self):
        self.group = SimpleNamespace(world_size=2, rank_in_group=1)
        self.meta = GGUFTensorMeta(
            ggml_type=int(_Q4_K),
            logical_shape=(8, 512),
            stored_shape=(8, 288),
            stored_dtype=torch.uint8,
            param_name="w.qweight",
        )
        self.config = GGUFConfig("/dev/null", {"w.weight": self.meta})
        values = torch.arange(8 * 288, dtype=torch.int64).remainder(251)
        self.loaded = values.to(torch.uint8).reshape(8, 288)

    def test_column_parallel_slices_output_rows(self):
        layer = ColumnParallelLinear(
            512,
            8,
            bias=False,
            quant_config=self.config,
            prefix="w",
            tp_group=self.group,
        )

        layer.weight_loader(layer.qweight, self.loaded)

        torch.testing.assert_close(layer.qweight, self.loaded[4:])

    def test_row_parallel_slices_packed_input_blocks(self):
        layer = RowParallelLinear(
            512,
            8,
            bias=False,
            quant_config=self.config,
            prefix="w",
            tp_group=self.group,
        )

        layer.weight_loader(layer.qweight, self.loaded)

        torch.testing.assert_close(layer.qweight, self.loaded[:, 144:])

    def test_merged_column_parallel_slices_each_output_group(self):
        layer = MergedColumnParallelLinear(
            512,
            [4, 4],
            bias=False,
            quant_config=self.config,
            prefix="w",
            tp_group=self.group,
        )

        layer.weight_loader(layer.qweight, self.loaded)

        expected = torch.cat((self.loaded[2:4], self.loaded[6:8]))
        torch.testing.assert_close(layer.qweight, expected)

    def test_row_parallel_rejects_unaligned_partition(self):
        metadata = GGUFTensorMeta(
            ggml_type=int(_Q4_K),
            logical_shape=(8, 256),
            stored_shape=(8, 144),
            stored_dtype=torch.uint8,
            param_name="w.qweight",
        )
        config = GGUFConfig("/dev/null", {"w.weight": metadata})

        with self.assertRaisesRegex(ValueError, "not aligned"):
            RowParallelLinear(
                256,
                8,
                bias=False,
                quant_config=config,
                prefix="w",
                tp_group=self.group,
            )


class TestGGUFIncompatibleOptions(unittest.TestCase):
    """Combinations that cannot work must fail at startup, not mid-run."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.gguf = self.tmp / "t.gguf"
        _write_gguf(self.gguf, [("w.weight", [8, 2], _F32, bytes(64))])

    def _resolve(self, **overrides):
        """Resolve with CUDA mocked in, so each test exercises its own guard.

        These are CPU-only tests; without the mock the CUDA guard fires first
        and every case would report the wrong reason.
        """
        from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
            _resolve_gguf_quant_load_spec,
        )

        server_args = Mock()
        server_args.tp_size = 1
        server_args.use_fsdp_inference = False
        server_args.lora_path = None
        server_args.minimax_h3_adaln_online = False
        server_args.minimax_h3_adaln_cache_path = None
        server_args.quantization = None
        server_args.nunchaku_config = None
        for key, value in overrides.items():
            setattr(server_args, key, value)
        with patch(
            "sglang.multimodal_gen.runtime.loader.transformer_load_utils.current_platform"
        ) as platform:
            platform.is_cuda.return_value = True
            return _resolve_gguf_quant_load_spec(
                gguf_file=str(self.gguf),
                server_args=server_args,
                model_cls=Mock(packed_modules_mapping=None),
            )

    def test_accepts_tp_without_fsdp(self):
        spec = self._resolve(tp_size=2)
        self.assertEqual(spec.gguf_file, str(self.gguf))
        self.assertEqual(spec.safetensors_list, [])
        # Each tensor keeps its checkpoint dtype rather than a single cast.
        self.assertIsNone(spec.param_dtype)

    def test_rejects_non_cuda_platform(self):
        """Fail before reading a multi-GiB checkpoint, not at the first linear."""
        from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
            _resolve_gguf_quant_load_spec,
        )

        server_args = Mock()
        server_args.tp_size = 1
        server_args.use_fsdp_inference = False
        server_args.lora_path = None
        server_args.minimax_h3_adaln_online = False
        server_args.minimax_h3_adaln_cache_path = None
        server_args.quantization = None
        server_args.nunchaku_config = None
        with patch(
            "sglang.multimodal_gen.runtime.loader.transformer_load_utils.current_platform"
        ) as platform:
            platform.is_cuda.return_value = False
            platform.device_type = "rocm"
            with self.assertRaisesRegex(ValueError, "require CUDA"):
                _resolve_gguf_quant_load_spec(
                    gguf_file=str(self.gguf),
                    server_args=server_args,
                    model_cls=Mock(packed_modules_mapping=None),
                )

    def test_rejects_fsdp_when_this_component_is_fsdp_managed(self):
        server_args_kwargs = {"use_fsdp_inference": True}
        with self.assertRaisesRegex(ValueError, "FSDP"):
            self._resolve(**server_args_kwargs)

    def test_allows_global_fsdp_when_this_component_is_offloaded(self):
        """FSDP is per component: an offloaded transformer is never sharded.

        Rejecting on the global flag would block FSDP on other resident
        components for no reason.
        """
        from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
            _resolve_gguf_quant_load_spec,
        )

        server_args = Mock()
        server_args.tp_size = 1
        server_args.use_fsdp_inference = True
        # ...but not for this component.
        server_args.should_use_fsdp_for_component.return_value = False
        server_args.lora_path = None
        server_args.minimax_h3_adaln_online = False
        server_args.minimax_h3_adaln_cache_path = None
        server_args.quantization = None
        server_args.nunchaku_config = None
        with patch(
            "sglang.multimodal_gen.runtime.loader.transformer_load_utils.current_platform"
        ) as platform:
            platform.is_cuda.return_value = True
            spec = _resolve_gguf_quant_load_spec(
                gguf_file=str(self.gguf),
                server_args=server_args,
                model_cls=Mock(packed_modules_mapping=None),
                component_name="transformer",
            )
        self.assertEqual(spec.gguf_file, str(self.gguf))
        server_args.should_use_fsdp_for_component.assert_called_once_with("transformer")

    def test_rejects_quantization_gguf_flag(self):
        """The file selects GGUF; the flag would be a second, silent selector."""
        with self.assertRaisesRegex(ValueError, "not\\s+`--quantization gguf`"):
            self._resolve(quantization="gguf")

    def test_rejects_conflicting_quantization_flag(self):
        """--quantization fp8 with a GGUF file must not be silently dropped."""
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self._resolve(quantization="fp8")

    def test_rejects_svdquant(self):
        """Nunchaku shares --transformer-weights-path; a silent drop is worse."""
        with self.assertRaisesRegex(ValueError, "svdquant"):
            self._resolve(nunchaku_config=object())

    def test_rejects_lora(self):
        with self.assertRaisesRegex(ValueError, "LoRA"):
            self._resolve(lora_path="some/adapter")

    def test_rejects_h3_adaln_online(self):
        with self.assertRaisesRegex(ValueError, "adaln-online"):
            self._resolve(minimax_h3_adaln_online=True)

    def test_rejects_h3_adaln_cache(self):
        with self.assertRaisesRegex(ValueError, "adaln-cache-path"):
            self._resolve(minimax_h3_adaln_cache_path="/tmp/cache.safetensors")


class TestGGUFKeyFilter(unittest.TestCase):
    def test_key_filter_matches_checkpoint_names(self):
        """The filter sees checkpoint names, like the safetensors iterator."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "f.gguf"
        row_bytes = 256 // _Q4_K_BLOCK * _Q4_K_TYPE_SIZE
        _write_gguf(
            path,
            [
                ("keep.weight", [256, 1], _Q4_K, bytes(row_bytes)),
                ("drop.weight", [256, 1], _Q4_K, bytes(row_bytes)),
            ],
        )
        meta = read_gguf_tensor_meta(str(path))
        loaded = dict(
            gguf_weights_iterator(
                str(path), meta, key_filter=lambda name: name.startswith("keep")
            )
        )
        # Filtered on the checkpoint name, yielded under the param name.
        self.assertEqual(sorted(loaded), ["keep.qweight"])


class TestGGUFPreDownloadValidation(unittest.TestCase):
    """A Hub reference must be rejected before it is fetched."""

    def test_hub_reference_is_recognized_without_io(self):
        self.assertTrue(names_gguf_checkpoint("owner/repo:Q4_K_M"))
        self.assertTrue(names_gguf_checkpoint("owner/repo/sub/model.gguf"))
        self.assertFalse(names_gguf_checkpoint("owner/repo"))
        self.assertFalse(names_gguf_checkpoint("owner/repo/model.safetensors"))
        self.assertFalse(names_gguf_checkpoint(""))
        # A local safetensors override must keep taking the safetensors path.
        self.assertFalse(names_gguf_checkpoint("/models/transformer.safetensors"))

    def test_missing_local_path_is_not_sent_to_the_hub(self):
        """A typo'd local path must report itself, not become a repo lookup.

        Recognition must not depend on directory depth: /a/x.gguf and
        /a/b/c/x.gguf are both local paths.
        """
        from sglang.multimodal_gen.runtime.loader import transformer_load_utils

        server_args = Mock(
            transformer_weights_path="/models/missing.gguf",
            revision=None,
            tp_size=1,
            use_fsdp_inference=False,
            lora_path=None,
            minimax_h3_adaln_online=False,
            minimax_h3_adaln_cache_path=None,
            quantization=None,
            nunchaku_config=None,
        )
        with (
            patch.object(
                transformer_load_utils, "resolve_hf_gguf_reference"
            ) as resolve,
            patch.object(transformer_load_utils, "current_platform") as platform,
            self.assertRaisesRegex(ValueError, "/models/missing.gguf"),
        ):
            platform.is_cuda.return_value = True
            transformer_load_utils.resolve_transformer_gguf_to_load(server_args)
        resolve.assert_not_called()

    def test_home_relative_path_is_expanded(self):
        """A `~` can arrive unexpanded from a config file or quoted argument."""
        import os

        from sglang.multimodal_gen.runtime.loader import transformer_load_utils

        # Put a real GGUF where an unexpanded "~" would miss it.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = Path(tmp.name) / "home"
        (home / "models").mkdir(parents=True)
        real = home / "models" / "h3.gguf"
        _write_gguf(real, [("w.weight", [4], _F32, bytes(16))])

        server_args = Mock()
        server_args.transformer_weights_path = "~/models/h3.gguf"
        server_args.revision = None
        server_args.tp_size = 1
        server_args.use_fsdp_inference = False
        server_args.lora_path = None
        server_args.minimax_h3_adaln_online = False
        server_args.minimax_h3_adaln_cache_path = None
        server_args.quantization = None
        server_args.nunchaku_config = None

        with (
            patch.dict(os.environ, {"HOME": str(home)}),
            patch.object(transformer_load_utils, "current_platform") as platform,
        ):
            platform.is_cuda.return_value = True
            resolved = transformer_load_utils.resolve_transformer_gguf_to_load(
                server_args
            )
        self.assertEqual(resolved, str(real))

    def test_revision_is_forwarded_to_the_hub_resolver(self):
        """--revision must pin the GGUF download, not be silently dropped."""
        from sglang.multimodal_gen.runtime.loader import transformer_load_utils

        server_args = Mock()
        server_args.transformer_weights_path = "owner/repo:Q4_K_M"
        server_args.revision = "abc123"
        server_args.tp_size = 1
        server_args.use_fsdp_inference = False
        server_args.lora_path = None
        server_args.minimax_h3_adaln_online = False
        server_args.minimax_h3_adaln_cache_path = None
        server_args.quantization = None
        server_args.nunchaku_config = None

        with (
            patch.object(
                transformer_load_utils, "resolve_hf_gguf_reference", return_value=None
            ) as resolve,
            patch.object(transformer_load_utils, "current_platform") as platform,
        ):
            platform.is_cuda.return_value = True
            # Resolution returns None, so the override itself is checked and
            # rejected as a non-file; the call arguments are what matters here.
            with self.assertRaises(ValueError):
                transformer_load_utils.resolve_transformer_gguf_to_load(server_args)
            resolve.assert_called_once_with("owner/repo:Q4_K_M", revision="abc123")

    def test_unsupported_config_rejected_before_download(self):
        from sglang.multimodal_gen.runtime.loader import transformer_load_utils

        server_args = Mock()
        # A Hub reference: resolving it would download the whole checkpoint.
        server_args.transformer_weights_path = "owner/repo:Q4_K_M"
        server_args.tp_size = 1
        server_args.use_fsdp_inference = True
        server_args.lora_path = None
        server_args.minimax_h3_adaln_online = False
        server_args.minimax_h3_adaln_cache_path = None
        server_args.quantization = None
        server_args.nunchaku_config = None

        with (
            patch.object(
                transformer_load_utils, "resolve_hf_gguf_reference"
            ) as resolve,
            patch.object(transformer_load_utils, "current_platform") as platform,
        ):
            platform.is_cuda.return_value = True
            with self.assertRaisesRegex(ValueError, "FSDP"):
                transformer_load_utils.resolve_transformer_gguf_to_load(server_args)
            resolve.assert_not_called()


class TestGGUFRejectsLoraConversion(unittest.TestCase):
    """The dynamic set_lora path must refuse before replacing any layer."""

    def _pipeline_with(self, layer):
        from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import (
            LoRAPipeline,
        )

        transformer = torch.nn.Module()
        transformer.add_module("blocks", torch.nn.Module())
        transformer.blocks.add_module("attn", torch.nn.Module())
        transformer.blocks.attn.add_module("qkv_proj", layer)

        # LoRAPipeline is abstract; the guard only needs `modules` and
        # `is_target_layer`, so a minimal concrete subclass keeps the test on
        # the real method rather than a copy of it.
        class _Pipeline(LoRAPipeline):
            def create_pipeline_stages(self, *args, **kwargs):
                raise NotImplementedError

        pipeline = _Pipeline.__new__(_Pipeline)
        pipeline.modules = {"transformer": transformer}
        pipeline.lora_initialized = False
        pipeline.is_target_layer = lambda name: name.endswith("qkv_proj")
        return pipeline

    def _gguf_like_layer(self):
        layer = torch.nn.Module()
        # uint8 cannot require grad, which is why GGUFLinearMethod passes
        # requires_grad=False when it registers the real parameter.
        layer.register_parameter(
            "qweight",
            torch.nn.Parameter(
                torch.zeros(4, 8, dtype=torch.uint8), requires_grad=False
            ),
        )
        return layer

    def _plain_layer(self):
        layer = torch.nn.Module()
        layer.register_parameter("weight", torch.nn.Parameter(torch.zeros(4, 8)))
        return layer

    def test_packed_weights_are_rejected(self):
        pipeline = self._pipeline_with(self._gguf_like_layer())
        with self.assertRaisesRegex(ValueError, "LoRA is not supported"):
            pipeline._reject_lora_on_packed_weights()
        # Nothing was converted, so a later retry on an unquantized model works.
        self.assertFalse(pipeline.lora_initialized)

    def test_plain_weights_are_accepted(self):
        pipeline = self._pipeline_with(self._plain_layer())
        pipeline._reject_lora_on_packed_weights()

    def test_rejects_while_still_offloaded(self):
        """The check must not need the weights materialized.

        Layerwise offload swaps `.data` for a 1-element placeholder but keeps the
        Parameter and its name, so the guard can run before offload is disabled --
        which is the point: disabling it would materialize the whole DiT first.
        """
        layer = self._gguf_like_layer()
        layer.qweight.data = torch.empty((1,), dtype=torch.uint8)
        pipeline = self._pipeline_with(layer)
        with self.assertRaisesRegex(ValueError, "LoRA is not supported"):
            pipeline._reject_lora_on_packed_weights()

    def test_set_lora_rejects_before_disabling_offload(self):
        """Ordering matters: disabling offload first would OOM a memory-limited
        deployment instead of returning the unsupported-LoRA error."""
        from unittest.mock import MagicMock

        pipeline = self._pipeline_with(self._gguf_like_layer())
        pipeline._resolve_lora_merge_mode = lambda *a, **k: "auto"
        pipeline._normalize_lora_params = lambda *a, **k: (
            ["n"],
            ["p"],
            [1.0],
            ["all"],
            [None],
        )
        entered = MagicMock(
            side_effect=AssertionError("offload was disabled before the check")
        )
        pipeline._temporarily_disable_offload = entered

        with self.assertRaisesRegex(ValueError, "LoRA is not supported"):
            pipeline.set_lora("n", lora_path="p")
        entered.assert_not_called()


if __name__ == "__main__":
    unittest.main()
