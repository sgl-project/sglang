"""Exercise the presharded file format with a stand-in for packed-NZ transfers."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_loader.loader import PreshardedModelLoader
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPreshardedPackedNZ(unittest.TestCase):
    def test_hash_dump_and_restore_keep_bits_views_and_extra_aliases(self):
        source = torch.tensor(
            [[-(2**31), -1], [0, 42], [2**31 - 1, 256]], dtype=torch.int32
        ).T
        weight = torch.empty_like(source)
        module = torch.nn.Module()
        module.register_buffer("saved_extra", torch.empty_like(source))
        extra_alias = module.saved_extra

        # Model the unsupported direct int32 transfers. CPU tensors supply real
        # dtype views, strided copies, safetensors serialization and hashing.
        original_to, original_copy = torch.Tensor.to, torch.Tensor.copy_

        def to(tensor, *args, **kwargs):
            if tensor.dtype == torch.int32 and kwargs.get("device") == "cpu":
                raise AssertionError("direct packed int32 D2H")
            return original_to(tensor, *args, **kwargs)

        def copy(tensor, other, *args, **kwargs):
            if tensor.dtype == torch.int32:
                raise AssertionError("direct packed int32 restore")
            return original_copy(tensor, other, *args, **kwargs)

        loader = object.__new__(PreshardedModelLoader)
        loader._verify_on_load = True
        loader._hash_num_threads = 1
        ordinary_hash = loader._hash_tensor(source)
        file = {
            "writer_rank": 0,
            "filename": "rank.safetensors",
            "tensors": [{"rank_to_names": {"0": ["weight"]}, "stored_key": "value"}],
        }
        items = [
            {"name": "weight", "stored_key": "value"},
            {"name": "saved_extra", "stored_key": "value", "is_extra": True},
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                PreshardedModelLoader,
                "_is_npu_packed_nz",
                side_effect=lambda t: t.dtype == torch.int32,
            ),
            patch.object(torch.Tensor, "to", to),
            patch.object(torch.Tensor, "copy_", copy),
        ):
            self.assertEqual(loader._hash_tensor(source), ordinary_hash)
            loader._dump_files_for_rank({"weight": source}, {"files": [file]}, 0, tmp)
            cached = loader._read_presharded_file(
                str(Path(tmp) / file["filename"]), ["value"]
            )
            hashes = []
            loader._apply_presharded_file(
                items=items,
                cached=cached,
                model=module,
                state_dict={"weight": weight},
                loaded_param_keys=set(),
                target_device=torch.device("cpu"),
                verify_hashes=hashes,
            )
        self.assertEqual(
            hashes, [("weight", ordinary_hash), ("saved_extra", ordinary_hash)]
        )
        self.assertTrue(torch.equal(weight, source))
        self.assertIs(module.saved_extra, extra_alias)
        self.assertTrue(torch.equal(extra_alias, source))

    def test_format_guard_queries_only_npu_int32(self):
        get_format = Mock(side_effect=[29, 2])
        with patch.dict(
            sys.modules, {"torch_npu": SimpleNamespace(get_npu_format=get_format)}
        ):
            classify = PreshardedModelLoader._is_npu_packed_nz
            self.assertFalse(classify(torch.zeros(1, dtype=torch.int32)))
            npu = SimpleNamespace(type="npu")
            self.assertFalse(classify(SimpleNamespace(device=npu, dtype=torch.int8)))
            self.assertFalse(get_format.called)
            packed = SimpleNamespace(device=npu, dtype=torch.int32)
            self.assertTrue(classify(packed))
            self.assertFalse(classify(packed))
            self.assertEqual(get_format.call_count, 2)

    def test_byte_views_preserve_shapes_and_narrowed_storage(self):
        with patch.object(
            PreshardedModelLoader, "_is_npu_packed_nz", return_value=True
        ):
            for source in (
                torch.tensor(-1, dtype=torch.int32),
                torch.empty((2, 0), dtype=torch.int32),
            ):
                with self.subTest(shape=source.shape):
                    value = PreshardedModelLoader._to_cpu_bits(source, copy=True)
                    self.assertTrue(torch.equal(value, source))
                    self.assertEqual(value.shape, source.shape)
            target = torch.zeros((3, 4), dtype=torch.int32)
            source = torch.tensor([[1, 2], [-3, -4], [5, 6]], dtype=torch.int32)
            PreshardedModelLoader._copy_packed_nz(target[:, ::2], source)
            self.assertTrue(torch.equal(target[:, ::2], source))
            self.assertEqual(torch.count_nonzero(target[:, 1::2]).item(), 0)

    def test_incompatible_packed_extra_is_rejected_before_writing(self):
        loader = object.__new__(PreshardedModelLoader)
        loader._verify_on_load = False
        module = torch.nn.Module()
        original = torch.tensor([42], dtype=torch.int32)
        module.register_buffer("extra", original)
        with patch.object(
            PreshardedModelLoader, "_is_npu_packed_nz", return_value=True
        ):
            for invalid in (
                torch.tensor([1, 2], dtype=torch.int32),
                torch.tensor([1.5]),
            ):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    loader._apply_presharded_file(
                        items=[{"name": "extra", "stored_key": "v", "is_extra": True}],
                        cached={"v": invalid},
                        model=module,
                        state_dict={},
                        loaded_param_keys=set(),
                        target_device=torch.device("cpu"),
                        verify_hashes=[],
                    )
                self.assertIs(module.extra, original)
                self.assertEqual(original.item(), 42)


if __name__ == "__main__":
    unittest.main()
