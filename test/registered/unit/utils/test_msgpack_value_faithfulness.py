"""Preserve values that need the existing pickle fallback in msgpack IPC."""

import unittest
from array import array
from unittest.mock import patch

import msgspec
import numpy as np
import torch

from sglang.srt.managers import io_struct
from sglang.srt.managers.io_struct import (
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput
from sglang.srt.multimodal.modality import Modality
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.msgpack_utils import (
    enc_hook,
    ext_hook,
    pickle_unfaithful_values,
    unpickle_values,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _round_trip_values(values):
    wrapped = pickle_unfaithful_values(values)
    forwarded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(wrapped, enc_hook=enc_hook), ext_hook=ext_hook
    )
    # A forwarding process re-encodes the packet without unwrapping its values.
    received = msgspec.msgpack.decode(
        msgspec.msgpack.encode(forwarded, enc_hook=enc_hook), ext_hook=ext_hook
    )
    return unpickle_values(received)


class TestMsgpackValueFaithfulness(unittest.TestCase):
    def test_integers_outside_msgpack_range_round_trip(self):
        for value in (-(1 << 63) - 1, 1 << 64, -(1 << 200), (1 << 256) - 1):
            for container in ("scalar", "flat_list", "nested_dict"):
                with self.subTest(value=value, container=container):
                    item = {
                        "scalar": value,
                        "flat_list": [0, value, 1],
                        "nested_dict": {"metadata": [0, {"id": value}]},
                    }[container]
                    self.assertEqual(_round_trip_values([item]), [item])

    def test_representable_integers_keep_native_wire_format(self):
        values = [-(1 << 63), -1, 0, (1 << 63) - 1, (1 << 64) - 1, True, False]
        wrapped = pickle_unfaithful_values(values)
        self.assertIs(wrapped, values)
        self.assertEqual(
            msgspec.msgpack.encode(wrapped), msgspec.msgpack.encode(values)
        )
        self.assertEqual(_round_trip_values(values), values)
        nested = [{"values": values}, values]
        self.assertIs(pickle_unfaithful_values(nested), nested)

    def test_structured_arrays_keep_fields_titles_and_layout(self):
        dtypes = (
            np.dtype([("token", "<i8"), ("score", "<f4")]),
            np.dtype(
                [("box", "<f4", (4,)), ("stats", [("mean", "<f4"), ("count", "<i2")])]
            ),
            np.dtype(
                [(("class title", "label"), ">i2"), ("weight", "<f8")], align=True
            ),
            np.dtype(
                {
                    "names": ["left", "right"],
                    "formats": ["u1", "<f8"],
                    "offsets": [0, 16],
                    "itemsize": 32,
                }
            ),
        )
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                source = np.zeros(3, dtype=dtype)
                source[source.dtype.names[0]] = 2
                decoded = _round_trip_values([source])[0]
                self.assertEqual(decoded.dtype, source.dtype)
                self.assertEqual(decoded.dtype.descr, source.dtype.descr)
                self.assertEqual(decoded.shape, source.shape)
                np.testing.assert_array_equal(decoded, source)
                self.assertFalse(np.shares_memory(decoded, source))

    def test_object_arrays_keep_python_values(self):
        source = np.empty((2, 2), dtype=object)
        source[0, 0] = {"label": "東京", "id": 1 << 100}
        source[0, 1] = (1, 2)
        source[1, 0] = None
        source[1, 1] = ["a", "b"]
        decoded = _round_trip_values([source])[0]
        self.assertEqual(decoded.dtype, source.dtype)
        self.assertEqual(decoded.shape, source.shape)
        self.assertEqual(decoded.tolist(), source.tolist())
        self.assertIsInstance(decoded[0, 1], tuple)
        source[0, 0]["label"] = "changed"
        self.assertEqual(decoded[0, 0]["label"], "東京")

    def test_dtype_metadata_and_zero_itemsize_arrays(self):
        for source in (
            np.array([1, 2], dtype=np.dtype("f4", metadata={"unit": "score"})),
            np.empty((2, 3), dtype="V0"),
        ):
            with self.subTest(dtype=source.dtype, itemsize=source.dtype.itemsize):
                decoded = _round_trip_values([source])[0]
                self.assertEqual(decoded.dtype, source.dtype)
                self.assertEqual(decoded.dtype.metadata, source.dtype.metadata)
                self.assertEqual(decoded.shape, source.shape)
                np.testing.assert_array_equal(decoded, source)

    def test_plain_arrays_still_use_existing_buffer_extension(self):
        for source in (
            np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2],
            np.array(2, dtype=np.int64),
            np.empty((2, 0), dtype=np.float16),
            np.array(["a", "東京"], dtype="U2"),
            np.array(["2025-01-02"], dtype="datetime64[D]"),
        ):
            with self.subTest(dtype=source.dtype, shape=source.shape):
                values = [source]
                self.assertIs(pickle_unfaithful_values(values), values)
                wrapped = enc_hook(source)
                self.assertEqual(wrapped.code, 3)
                decoded = _round_trip_values(values)[0]
                self.assertEqual(decoded.dtype, source.dtype)
                np.testing.assert_array_equal(decoded, source)

    def test_shared_fallback_array_keeps_one_object(self):
        source = np.array([(1, 0.25)], dtype=[("token", "<i4"), ("score", "<f4")])
        decoded = _round_trip_values([source, {"array": source}, source])
        self.assertIs(decoded[0], decoded[1]["array"])
        self.assertIs(decoded[0], decoded[2])
        np.testing.assert_array_equal(decoded[0], source)

    def test_real_multimodal_request_survives_forwarding(self):
        records = np.array(
            [(7, 0.25), (9, 0.75)], dtype=[("token", "<i4"), ("score", "<f4")]
        )
        source_feature = torch.tensor([[1.0, 2.0]])
        request = TokenizedGenerateReqInput(
            input_text="",
            input_ids=array("q", [1, 2]),
            input_embeds=None,
            mm_inputs=MultimodalProcessorOutput(
                mm_items=[
                    MultimodalDataItem(
                        modality=Modality.IMAGE,
                        feature=source_feature,
                        model_specific_data={
                            "records": records,
                            "large_id": 1 << 200,
                            "nested": {"records": records},
                        },
                    )
                ]
            ),
            token_type_ids=[0, 0],
            sampling_params=SamplingParams(),
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
        )
        with patch.object(io_struct, "_USE_PICKLE_IPC", False):
            request.wrap_pickle_fields()
            forwarded = msgpack_decode(msgpack_encode(request))
            decoded = msgpack_decode(msgpack_encode(forwarded))
            decoded.unwrap_pickle_fields()
        item = decoded.mm_inputs.mm_items[0]
        self.assertIsInstance(decoded, TokenizedGenerateReqInput)
        self.assertEqual(item.model_specific_data["large_id"], 1 << 200)
        self.assertEqual(item.model_specific_data["records"].dtype, records.dtype)
        np.testing.assert_array_equal(item.model_specific_data["records"], records)
        self.assertIs(
            item.model_specific_data["records"],
            item.model_specific_data["nested"]["records"],
        )
        torch.testing.assert_close(item.feature, source_feature, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
