import asyncio
import importlib.util
import sys
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np
import pybase64
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()
if importlib.util.find_spec("uvloop") is None:
    _uvloop_stub = types.ModuleType("uvloop")
    _uvloop_stub.EventLoopPolicy = asyncio.DefaultEventLoopPolicy
    sys.modules["uvloop"] = _uvloop_stub

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.state_capturer.routed_experts import (
    encode_topk_b64,
    extract_routed_experts_from_meta_info,
    serialize_routed_experts_array,
    validate_routed_experts_dtype,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRoutedExpertsSerialization(CustomTestCase):
    def test_encode_topk_b64_matches_legacy_bytes_for_tensors_and_arrays(self):
        base = torch.arange(2 * 6 * 4, dtype=torch.int32).reshape(6, 4, 2)
        non_contig = base.transpose(0, 1)  # (4, 6, 2) non-contiguous view
        for tensor in (base, non_contig):
            expected = pybase64.b64encode(tensor.numpy().tobytes()).decode("utf-8")
            self.assertEqual(encode_topk_b64(tensor), expected)
            self.assertEqual(encode_topk_b64(tensor.numpy()), expected)
        self.assertIsNone(encode_topk_b64(None))
        self.assertEqual(encode_topk_b64("already-b64"), "already-b64")

    def test_serialize_and_extract_roundtrip_across_wire_dtypes(self):
        rows_u8 = torch.randint(0, 256, (17, 4, 8), dtype=torch.int32)
        for dtype_name, expected_np_dtype, ratio in (
            (None, np.int32, 1),
            ("int32", np.int32, 1),
            ("int16", np.int16, 2),
            ("uint8", np.uint8, 4),
        ):
            with self.subTest(dtype_name=dtype_name):
                arr = serialize_routed_experts_array(rows_u8, dtype_name=dtype_name)
                self.assertIsInstance(arr, np.ndarray)
                self.assertTrue(arr.flags.c_contiguous)
                self.assertEqual(arr.dtype, expected_np_dtype)
                self.assertEqual(arr.nbytes * ratio, rows_u8.numel() * 4)

                meta_info = {"routed_experts": encode_topk_b64(arr)}
                if dtype_name is not None:
                    meta_info["routed_experts_dtype"] = dtype_name
                decoded = extract_routed_experts_from_meta_info(
                    {"meta_info": meta_info}
                )
                self.assertEqual(decoded.dtype, np.int32)
                np.testing.assert_array_equal(
                    decoded.reshape(rows_u8.shape), rows_u8.numpy()
                )

    def test_serialize_rejects_out_of_range_or_unknown_dtypes(self):
        with self.assertRaises(ValueError):
            serialize_routed_experts_array(
                torch.tensor([[[256]]], dtype=torch.int32), dtype_name="uint8"
            )
        with self.assertRaises(ValueError):
            serialize_routed_experts_array(
                torch.tensor([[[-1]]], dtype=torch.int32), dtype_name="uint8"
            )
        with self.assertRaises(ValueError):
            serialize_routed_experts_array(
                torch.tensor([[[40000]]], dtype=torch.int32), dtype_name="int16"
            )
        with self.assertRaises(ValueError):
            serialize_routed_experts_array(
                torch.tensor([[[1]]], dtype=torch.int32), dtype_name="float16"
            )
        with self.assertRaises(ValueError):
            extract_routed_experts_from_meta_info(
                {
                    "meta_info": {
                        "routed_experts": "AAAA",
                        "routed_experts_dtype": "int64",
                    }
                }
            )

    def test_validate_routed_experts_dtype_checks_logical_expert_count(self):
        from types import SimpleNamespace

        with patch(
            "sglang.srt.state_capturer.routed_experts.get_global_expert_location_metadata",
            return_value=SimpleNamespace(num_logical_experts=256),
        ):
            self.assertIsNone(validate_routed_experts_dtype("uint8"))
            self.assertIsNone(validate_routed_experts_dtype("int16"))
            self.assertIsNone(validate_routed_experts_dtype(None))
            self.assertIsNotNone(validate_routed_experts_dtype("float16"))

        with patch(
            "sglang.srt.state_capturer.routed_experts.get_global_expert_location_metadata",
            return_value=SimpleNamespace(num_logical_experts=288),
        ):
            self.assertIsNotNone(validate_routed_experts_dtype("uint8"))
            self.assertIsNone(validate_routed_experts_dtype("int16"))

    def test_generate_req_input_validates_routed_experts_dtype(self):
        single = GenerateReqInput(
            input_ids=[1, 2, 3],
            return_routed_experts=True,
            routed_experts_dtype="uint8",
        )
        single.normalize_batch_and_arguments()
        self.assertEqual(single.routed_experts_dtype, "uint8")

        batched = GenerateReqInput(
            input_ids=[[1, 2, 3], [4, 5]],
            return_routed_experts=True,
            routed_experts_dtype="int16",
        )
        batched.normalize_batch_and_arguments()
        self.assertEqual(batched[0].routed_experts_dtype, "int16")
        self.assertEqual(batched[1].routed_experts_dtype, "int16")

        with self.assertRaises(ValueError):
            GenerateReqInput(
                input_ids=[1, 2, 3],
                return_routed_experts=False,
                routed_experts_dtype="uint8",
            ).normalize_batch_and_arguments()

        with self.assertRaises(ValueError):
            GenerateReqInput(
                input_ids=[1, 2, 3],
                return_routed_experts=True,
                routed_experts_dtype="int8",
            ).normalize_batch_and_arguments()

    def test_detokenizer_b64_encode_per_request_accepts_numpy_and_tensor(self):
        arr = np.arange(16, dtype=np.uint8).reshape(2, 2, 4)
        tensor = torch.arange(16, dtype=torch.int32).reshape(2, 2, 4)
        out = DetokenizerManager._b64_encode_per_request([arr, None, tensor])
        self.assertEqual(
            out,
            [encode_topk_b64(arr), None, encode_topk_b64(tensor)],
        )
        self.assertIsNone(DetokenizerManager._b64_encode_per_request(None))

    def test_tokenizer_manager_offloads_large_topk_batches_off_event_loop(self):
        small = [np.arange(16, dtype=np.uint8).reshape(2, 2, 4)]
        large = [np.zeros((256, 16, 8), dtype=np.int32)]  # 128 KiB >= 64 KiB

        async def _run():
            main_tid = threading.get_ident()
            worker_tids = []
            orig_to_thread = asyncio.to_thread

            async def _spy_to_thread(fn, *args, **kwargs):
                def _wrapped():
                    worker_tids.append(threading.get_ident())
                    return fn(*args, **kwargs)

                return await orig_to_thread(_wrapped)

            with patch("asyncio.to_thread", side_effect=_spy_to_thread):
                (
                    small_routed,
                    small_idx,
                    small_dtypes,
                ) = await TokenizerManager._encode_batch_topk_outputs(small, None)
                self.assertEqual(worker_tids, [])
                self.assertEqual(small_routed, [encode_topk_b64(small[0])])
                self.assertIsNone(small_idx)
                self.assertEqual(small_dtypes, ["uint8"])

                (
                    large_routed,
                    large_idx,
                    large_dtypes,
                ) = await TokenizerManager._encode_batch_topk_outputs(large, None)
                self.assertEqual(len(worker_tids), 1)
                self.assertNotEqual(worker_tids[0], main_tid)
                self.assertEqual(large_routed, [encode_topk_b64(large[0])])
                self.assertIsNone(large_idx)
                self.assertEqual(large_dtypes, ["int32"])

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
