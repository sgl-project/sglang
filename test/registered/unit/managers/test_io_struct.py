import copy
import unittest
import weakref
from array import array

import msgspec
import numpy as np
import torch

from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    _restore_torch_tensor,
    enc_hook,
    ext_hook,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalProcessorOutput,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.cuda_ipc_transport_utils import CudaIpcTensorTransportProxy
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
)

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class TestTokenizedReqInputMsgpack(unittest.TestCase):
    def _make_mm_inputs(self, device="cpu"):
        return MultimodalProcessorOutput(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[(0, 1)],
                    format=MultimodalInputFormat.NORMAL,
                    feature=torch.tensor(
                        [[1.0, 2.0]], dtype=torch.float32, device=device
                    ),
                    model_specific_data={
                        "image_grid_thw": torch.tensor(
                            [[1, 1, 2]], dtype=torch.int64, device=device
                        ),
                        "patch_counts": np.array([2], dtype=np.int32),
                        "names": ["image0"],
                        "count": np.int64(2),
                        "enabled": np.bool_(True),
                        "size": (336, 336),
                    },
                )
            ],
            input_ids=[1, 2],
            padded_input_ids=[10, 10],
            im_token_id=10,
            mrope_positions=torch.tensor([[0, 1]], dtype=torch.int64, device=device),
            token_type_ids=torch.tensor([0, 0], dtype=torch.int64, device=device),
        )

    def _round_trip(self, req):
        req.wrap_pickle_fields()
        decoded = msgpack_decode(msgpack_encode(req))
        decoded.unwrap_pickle_fields()
        return decoded

    def _round_trip_mm_inputs(self, mm_inputs):
        decoded = self._round_trip(
            TokenizedGenerateReqInput(
                input_text="",
                input_ids=array("q", [1, 2]),
                input_embeds=None,
                mm_inputs=mm_inputs,
                token_type_ids=[0, 0],
                sampling_params=SamplingParams(),
                return_logprob=False,
                logprob_start_len=0,
                top_logprobs_num=0,
                token_ids_logprob=None,
                stream=False,
            )
        )
        return decoded.mm_inputs

    def test_generate_mm_inputs_round_trip_without_pickle_wrapper(self):
        decoded = self._round_trip(
            TokenizedGenerateReqInput(
                input_text="",
                input_ids=array("q", [1, 2]),
                input_embeds=None,
                mm_inputs=self._make_mm_inputs(),
                token_type_ids=[0, 0],
                sampling_params=SamplingParams(),
                return_logprob=False,
                logprob_start_len=0,
                top_logprobs_num=0,
                token_ids_logprob=None,
                stream=False,
            )
        )

        self.assertIsInstance(decoded.mm_inputs, MultimodalProcessorOutput)
        item = decoded.mm_inputs.mm_items[0]
        self.assertIsInstance(item, MultimodalDataItem)
        self.assertEqual(item.modality, Modality.IMAGE)
        self.assertEqual(item.offsets, [(0, 1)])
        self.assertTrue(
            torch.equal(item.feature, torch.tensor([[1.0, 2.0]], device="cpu"))
        )
        self.assertTrue(
            torch.equal(
                item.model_specific_data["image_grid_thw"],
                torch.tensor([[1, 1, 2]], dtype=torch.int64, device="cpu"),
            )
        )
        np.testing.assert_array_equal(
            item.model_specific_data["patch_counts"],
            np.array([2], dtype=np.int32),
        )
        self.assertEqual(item.model_specific_data["count"], 2)
        self.assertIs(item.model_specific_data["enabled"], True)
        self.assertEqual(item.model_specific_data["size"], [336, 336])
        self.assertTrue(
            torch.equal(
                decoded.mm_inputs.mrope_positions,
                torch.tensor([[0, 1]], dtype=torch.int64, device="cpu"),
            )
        )

    def test_dynamic_model_specific_attribute_round_trip(self):
        mm_inputs = self._make_mm_inputs()
        mm_inputs.mm_items[0].audio_feature_lens = torch.tensor([2])

        decoded = self._round_trip_mm_inputs(mm_inputs)

        self.assertTrue(
            torch.equal(decoded.mm_items[0].audio_feature_lens, torch.tensor([2]))
        )
        self.assertIn("audio_feature_lens", decoded.mm_items[0].model_specific_data)

    def test_multimodal_hash_is_normalized_to_uint64(self):
        mm_inputs = self._make_mm_inputs()
        mm_inputs.mm_items[0].hash = (1 << 256) - 1
        constructed = MultimodalDataItem(modality=Modality.IMAGE, hash=(1 << 128) - 1)

        decoded = self._round_trip_mm_inputs(mm_inputs)

        self.assertEqual(decoded.mm_items[0].hash, (1 << 64) - 1)
        self.assertEqual(constructed.hash, (1 << 64) - 1)

    def test_multimodal_processor_output_supports_weakrefs(self):
        mm_inputs = self._make_mm_inputs()

        ref = weakref.ref(mm_inputs)

        self.assertIs(ref(), mm_inputs)

    def test_unknown_ext_payload_is_preserved_without_decoding(self):
        ext = msgspec.msgpack.Ext(99, b"not msgpack")

        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(ext), ext_hook=ext_hook)

        self.assertEqual(decoded, ext)

    def test_malformed_known_buffer_ext_is_rejected(self):
        with self.assertRaisesRegex(msgspec.DecodeError, "missing metadata"):
            ext_hook(3, memoryview(b"bad"))

    def test_embedding_mm_inputs_round_trip_without_pickle_wrapper(self):
        decoded = self._round_trip(
            TokenizedEmbeddingReqInput(
                input_text="",
                input_ids=array("q", [1, 2]),
                mm_inputs=self._make_mm_inputs(),
                token_type_ids=[0, 0],
                sampling_params=SamplingParams(),
            )
        )

        self.assertIsInstance(decoded.mm_inputs, MultimodalProcessorOutput)
        self.assertTrue(
            torch.equal(
                decoded.mm_inputs.mm_items[0].feature,
                torch.tensor([[1.0, 2.0]], device="cpu"),
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_generate_mm_inputs_round_trip_preserves_cuda_tensor_device(self):
        decoded = self._round_trip(
            TokenizedGenerateReqInput(
                input_text="",
                input_ids=array("q", [1, 2]),
                input_embeds=None,
                mm_inputs=self._make_mm_inputs(device="cuda:0"),
                token_type_ids=[0, 0],
                sampling_params=SamplingParams(),
                return_logprob=False,
                logprob_start_len=0,
                top_logprobs_num=0,
                token_ids_logprob=None,
                stream=False,
            )
        )

        item = decoded.mm_inputs.mm_items[0]
        self.assertEqual(item.feature.device.type, "cuda")
        self.assertEqual(item.model_specific_data["image_grid_thw"].device.type, "cuda")
        self.assertEqual(decoded.mm_inputs.mrope_positions.device.type, "cuda")

    def test_cuda_ipc_proxy_state_round_trip_preserves_tuple_types(self):
        proxy = CudaIpcTensorTransportProxy.__new__(CudaIpcTensorTransportProxy)
        proxy.proxy_state = {
            "ipc_extra": {
                "shape": torch.Size([2, 3]),
                "stride": (3, 1),
                "dtype": torch.float16,
                "nested": [(1, 2), torch.Size([4])],
            },
            "tensor_data": None,
        }
        proxy.reconstruct_tensor = None
        proxy.sync_data_meta = {
            "handle": "dummy",
            "shape": torch.Size([1]),
            "dtype": np.dtype("float32"),
        }
        proxy.sync_buffer = None

        mm_inputs = self._make_mm_inputs()
        mm_inputs.mm_items[0].model_specific_data["ipc_proxy"] = proxy
        decoded = self._round_trip(
            TokenizedGenerateReqInput(
                input_text="",
                input_ids=array("q", [1, 2]),
                input_embeds=None,
                mm_inputs=mm_inputs,
                token_type_ids=[0, 0],
                sampling_params=SamplingParams(),
                return_logprob=False,
                logprob_start_len=0,
                top_logprobs_num=0,
                token_ids_logprob=None,
                stream=False,
            )
        )

        decoded_proxy = decoded.mm_inputs.mm_items[0].model_specific_data["ipc_proxy"]
        ipc_extra = decoded_proxy.proxy_state["ipc_extra"]
        self.assertIsInstance(ipc_extra["shape"], torch.Size)
        self.assertEqual(ipc_extra["shape"], torch.Size([2, 3]))
        self.assertIsInstance(ipc_extra["stride"], tuple)
        self.assertEqual(ipc_extra["stride"], (3, 1))
        self.assertIsInstance(ipc_extra["nested"][0], tuple)
        self.assertIsInstance(ipc_extra["nested"][1], torch.Size)
        self.assertIsInstance(decoded_proxy.sync_data_meta["shape"], torch.Size)
        self.assertIsInstance(decoded_proxy.sync_data_meta["dtype"], np.dtype)
        self.assertFalse(decoded_proxy._consumer_acknowledged)

    def test_cuda_ipc_proxy_tensor_fallback_round_trip(self):
        proxy = CudaIpcTensorTransportProxy.__new__(CudaIpcTensorTransportProxy)
        proxy.proxy_state = {
            "ipc_extra": None,
            "tensor_data": torch.tensor([1.0, 2.0], device="cpu"),
        }
        proxy.reconstruct_tensor = None
        proxy.sync_data_meta = {
            "handle": "dummy",
            "shape": (1,),
            "dtype": np.dtype("uint8"),
        }
        proxy.sync_buffer = None

        mm_inputs = self._make_mm_inputs()
        mm_inputs.mm_items[0].model_specific_data["ipc_proxy"] = proxy
        decoded = self._round_trip(
            TokenizedGenerateReqInput(
                input_text="",
                input_ids=array("q", [1, 2]),
                input_embeds=None,
                mm_inputs=mm_inputs,
                token_type_ids=[0, 0],
                sampling_params=SamplingParams(),
                return_logprob=False,
                logprob_start_len=0,
                top_logprobs_num=0,
                token_ids_logprob=None,
                stream=False,
            )
        )

        decoded_proxy = decoded.mm_inputs.mm_items[0].model_specific_data["ipc_proxy"]
        self.assertTrue(
            torch.equal(
                decoded_proxy.proxy_state["tensor_data"],
                torch.tensor([1.0, 2.0], device="cpu"),
            )
        )

    def test_evs_model_specific_data_round_trip(self):
        mm_inputs = self._make_mm_inputs()
        item = mm_inputs.mm_items[0]
        item.modality = Modality.VIDEO
        item.model_specific_data.update(
            {
                "thw_grids": [(2, 3, 4)],
                "pre_chunked_input_ids": [1, 2, 3],
            }
        )
        decoded = self._round_trip(
            TokenizedGenerateReqInput(
                input_text="",
                input_ids=array("q", [1, 2]),
                input_embeds=None,
                mm_inputs=mm_inputs,
                token_type_ids=[0, 0],
                sampling_params=SamplingParams(),
                return_logprob=False,
                logprob_start_len=0,
                top_logprobs_num=0,
                token_ids_logprob=None,
                stream=False,
            )
        )

        decoded_item = decoded.mm_inputs.mm_items[0]
        self.assertEqual(decoded_item.thw_grids, [[2, 3, 4]])
        self.assertEqual(decoded_item.pre_chunked_input_ids, [1, 2, 3])

    def test_torch_tensor_ext_wire_format(self):
        ext = enc_hook(torch.tensor([1, 2], dtype=torch.int16, device="cpu"))
        self.assertIsInstance(ext, msgspec.msgpack.Ext)
        self.assertEqual(ext.code, 2)
        self.assertEqual(
            bytes(ext.data).hex(),
            "0000000d939102a5696e743136a363707501000200",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_empty_cpu_tensor_restore_ignores_default_device(self):
        previous_device = torch.get_default_device()
        try:
            torch.set_default_device("cuda")
            tensor = _restore_torch_tensor((0,), "float32", b"", "cpu")
            self.assertEqual(tensor.device.type, "cpu")
        finally:
            torch.set_default_device(previous_device)


class TestGenerateReqInputNormalization(CustomTestCase):
    """Test the normalization of GenerateReqInput for batch processing and different input formats."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

    def setUp(self):
        # Common setup for all tests
        self.base_req = GenerateReqInput(
            text=["Hello", "World"],
            sampling_params=[{}, {}],
            rid=["id1", "id2"],
        )

    def test_single_image_to_list_of_lists(self):
        """Test that a single image is converted to a list of single-image lists."""
        req = copy.deepcopy(self.base_req)
        req.image_data = "single_image.jpg"  # A single image (non-list)

        req.normalize_batch_and_arguments()

        # Should be converted to [[image], [image]]
        self.assertEqual(len(req.image_data), 2)
        self.assertEqual(len(req.image_data[0]), 1)
        self.assertEqual(len(req.image_data[1]), 1)
        self.assertEqual(req.image_data[0][0], "single_image.jpg")
        self.assertEqual(req.image_data[1][0], "single_image.jpg")

        # Check modalities
        self.assertEqual(req.modalities, ["image", "image"])

    def test_list_of_images_to_list_of_lists(self):
        """Test that a list of images is converted to a list of single-image lists."""
        req = copy.deepcopy(self.base_req)
        req.image_data = ["image1.jpg", "image2.jpg"]  # List of images

        req.normalize_batch_and_arguments()

        # Should be converted to [[image1], [image2]]
        self.assertEqual(len(req.image_data), 2)
        self.assertEqual(len(req.image_data[0]), 1)
        self.assertEqual(len(req.image_data[1]), 1)
        self.assertEqual(req.image_data[0][0], "image1.jpg")
        self.assertEqual(req.image_data[1][0], "image2.jpg")

        # Check modalities
        self.assertEqual(req.modalities, ["image", "image"])

    def test_list_of_lists_with_different_modalities(self):
        """Test handling of list of lists of images with different modalities."""
        req = copy.deepcopy(self.base_req)
        req.image_data = [
            ["image1.jpg"],  # Single image (image modality)
            ["image2.jpg", "image3.jpg"],  # Multiple images (multi-images modality)
        ]

        req.normalize_batch_and_arguments()

        # Structure should remain the same
        self.assertEqual(len(req.image_data), 2)
        self.assertEqual(len(req.image_data[0]), 1)
        self.assertEqual(len(req.image_data[1]), 2)

        # Check modalities
        self.assertEqual(req.modalities, ["image", "multi-images"])

    def test_list_of_lists_with_none_values(self):
        """Test handling of list of lists with None values."""
        req = copy.deepcopy(self.base_req)
        req.image_data = [
            [None],  # None value
            ["image.jpg"],  # Single image
        ]

        req.normalize_batch_and_arguments()

        # Structure should remain the same
        self.assertEqual(len(req.image_data), 2)
        self.assertEqual(len(req.image_data[0]), 1)
        self.assertEqual(len(req.image_data[1]), 1)

        # Check modalities
        self.assertEqual(req.modalities, [None, "image"])

    def test_expanding_parallel_sample_correlation(self):
        """Test that when expanding with parallel samples, prompts, images and modalities are properly correlated."""
        req = copy.deepcopy(self.base_req)
        req.text = ["Prompt 1", "Prompt 2"]
        req.image_data = [
            ["image1.jpg"],
            ["image2.jpg", "image3.jpg"],
        ]
        req.sampling_params = {"n": 3}  # All prompts get 3 samples

        # Define expected values before normalization
        expected_text = req.text * 3
        expected_images = req.image_data * 3
        expected_modalities = ["image", "multi-images"] * 3

        req.normalize_batch_and_arguments()

        # Should be expanded to 6 items (2 original * 3 parallel)
        self.assertEqual(len(req.image_data), 6)

        # Check that images are properly expanded
        self.assertEqual(req.image_data, expected_images)

        # Check modalities
        self.assertEqual(req.modalities, expected_modalities)

        # Ensure that text items are properly duplicated too
        self.assertEqual(req.text, expected_text)

    def test_specific_parallel_n_per_sample(self):
        """Test parallel expansion when different samples have different n values."""
        req = copy.deepcopy(self.base_req)
        req.text = ["Prompt 1", "Prompt 2"]
        req.image_data = [
            ["image1.jpg"],
            ["image2.jpg", "image3.jpg"],
        ]
        req.sampling_params = [
            {"n": 2},
            {"n": 2},
        ]  # First prompt gets 2 samples, second prompt gets 2 samples

        expected_images = req.image_data * 2
        expected_modalities = ["image", "multi-images"] * 2
        expected_text = req.text * 2

        req.normalize_batch_and_arguments()

        # Should be expanded to 4 items (2 original * 2 parallel)
        self.assertEqual(len(req.image_data), 4)

        # Check that the first 2 are copies for the first prompt
        self.assertEqual(req.image_data, expected_images)

        # Check modalities
        self.assertEqual(req.modalities, expected_modalities)

        # Check text expansion
        self.assertEqual(req.text, expected_text)

    def test_mixed_none_and_images_with_parallel_samples(self):
        """Test that when some batch items have images and others None, parallel expansion works correctly."""
        req = copy.deepcopy(self.base_req)
        req.text = ["Prompt 1", "Prompt 2", "Prompt 3"]
        req.rid = ["id1", "id2", "id3"]
        req.image_data = [
            ["image1.jpg"],
            None,
            ["image3_1.jpg", "image3_2.jpg"],
        ]
        req.sampling_params = {"n": 2}  # All prompts get 2 samples

        expected_images = req.image_data * 2
        expected_modalities = ["image", None, "multi-images"] * 2
        expected_text = req.text * 2

        req.normalize_batch_and_arguments()

        # Should be expanded to 6 items (3 original * 2 parallel)
        self.assertEqual(len(req.image_data), 6)

        # Check image data
        self.assertEqual(req.image_data, expected_images)

        # Check modalities
        self.assertEqual(req.modalities, expected_modalities)

        # Check text expansion
        self.assertEqual(req.text, expected_text)

    def test_correlation_with_sampling_params(self):
        """Test that sampling parameters are correctly correlated with prompts during expansion."""
        req = copy.deepcopy(self.base_req)
        req.text = ["Prompt 1", "Prompt 2"]
        req.image_data = [
            ["image1.jpg"],
            ["image2.jpg"],
        ]
        req.sampling_params = [
            {"temperature": 0.7, "n": 2},
            {"temperature": 0.9, "n": 2},
        ]

        req.normalize_batch_and_arguments()

        # Check sampling params expansion
        self.assertEqual(len(req.sampling_params), 4)
        self.assertEqual(req.sampling_params[0]["temperature"], 0.7)
        self.assertEqual(req.sampling_params[1]["temperature"], 0.9)
        self.assertEqual(req.sampling_params[2]["temperature"], 0.7)
        self.assertEqual(req.sampling_params[3]["temperature"], 0.9)

        # Should be expanded to 4 items (2 original * 2 parallel)
        self.assertEqual(len(req.image_data), 4)

        # Check correlation with images
        self.assertEqual(req.image_data[0], ["image1.jpg"])
        self.assertEqual(req.image_data[1], ["image2.jpg"])
        self.assertEqual(req.image_data[2], ["image1.jpg"])
        self.assertEqual(req.image_data[3], ["image2.jpg"])

    def test_single_example_with_image(self):
        """Test handling of single example with image."""
        req = GenerateReqInput(
            text="Hello",
            image_data="single_image.jpg",
        )

        req.normalize_batch_and_arguments()

        # For single examples, image_data doesn't get processed into lists
        self.assertEqual(req.image_data, "single_image.jpg")
        self.assertIsNone(req.modalities)  # Modalities isn't set for single examples

    def test_single_to_batch_with_parallel_sampling(self):
        """Test single example converted to batch with parallel sampling."""
        req = GenerateReqInput(
            text="Hello",
            image_data="single_image.jpg",
            sampling_params={"n": 3},  # parallel_sample_num = 3
        )

        # Define expected values before normalization
        expected_text = ["Hello"] * 3

        req.normalize_batch_and_arguments()

        # Should be converted to batch with text=["Hello"]
        self.assertEqual(req.text, expected_text)

        # Image should be automatically wrapped to list of lists with length 1*3=3
        self.assertEqual(len(req.image_data), 3)
        self.assertEqual(req.image_data[0][0], "single_image.jpg")
        self.assertEqual(req.image_data[1][0], "single_image.jpg")
        self.assertEqual(req.image_data[2][0], "single_image.jpg")

        # Modalities should be set for all 3 examples
        self.assertEqual(req.modalities, ["image", "image", "image"])

    def test_audio_data_handling(self):
        """Test handling of audio_data."""
        req = copy.deepcopy(self.base_req)
        req.audio_data = "audio.mp3"  # Single audio

        req.normalize_batch_and_arguments()

        # Should be converted to ["audio.mp3", "audio.mp3"]
        self.assertEqual(len(req.audio_data), 2)
        self.assertEqual(req.audio_data[0], "audio.mp3")
        self.assertEqual(req.audio_data[1], "audio.mp3")

        # Test with list
        req = copy.deepcopy(self.base_req)
        req.audio_data = ["audio1.mp3", "audio2.mp3"]

        req.normalize_batch_and_arguments()

        # Should remain the same
        self.assertEqual(len(req.audio_data), 2)
        self.assertEqual(req.audio_data[0], "audio1.mp3")
        self.assertEqual(req.audio_data[1], "audio2.mp3")

    def test_input_ids_normalization(self):
        """Test normalization of input_ids instead of text."""
        # Test single input_ids
        req = GenerateReqInput(input_ids=[1, 2, 3])
        req.normalize_batch_and_arguments()
        self.assertTrue(req.is_single)
        self.assertEqual(req.batch_size, 1)

        # Test batch input_ids
        req = GenerateReqInput(input_ids=[[1, 2, 3], [4, 5, 6]])
        req.normalize_batch_and_arguments()
        self.assertFalse(req.is_single)
        self.assertEqual(req.batch_size, 2)

        # Test with parallel sampling
        req = GenerateReqInput(
            input_ids=[[1, 2, 3], [4, 5, 6]], sampling_params={"n": 2}
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(len(req.input_ids), 4)  # 2 original * 2 parallel

    def test_input_embeds_normalization(self):
        """Test normalization of input_embeds."""
        # Test single input_embeds
        req = GenerateReqInput(input_embeds=[[0.1, 0.2], [0.3, 0.4]])
        req.normalize_batch_and_arguments()
        self.assertTrue(req.is_single)
        self.assertEqual(req.batch_size, 1)

        # Test batch input_embeds
        req = GenerateReqInput(input_embeds=[[[0.1, 0.2]], [[0.3, 0.4]]])
        req.normalize_batch_and_arguments()
        self.assertFalse(req.is_single)
        self.assertEqual(req.batch_size, 2)

    def test_input_embeds_with_parallel_sampling(self):
        """Test input_embeds normalization with parallel sampling (n > 1)."""
        # Test single input_embeds with parallel sampling
        req = GenerateReqInput(
            input_embeds=[[0.1, 0.2]],  # single embedding vector
            sampling_params={"n": 2},
        )
        req.normalize_batch_and_arguments()

        # Should be converted from single to batch and then expanded
        self.assertFalse(req.is_single)
        self.assertEqual(len(req.input_embeds), 2)
        # Both should be the same input_embeds
        self.assertEqual(req.input_embeds[0], [[0.1, 0.2]])
        self.assertEqual(req.input_embeds[1], [[0.1, 0.2]])

        # Test batch input_embeds with parallel sampling
        req = GenerateReqInput(
            input_embeds=[[[0.1, 0.2]], [[0.3, 0.4]]], sampling_params={"n": 3}
        )
        req.normalize_batch_and_arguments()

        # Should be expanded
        self.assertFalse(req.is_single)
        self.assertEqual(len(req.input_embeds), 6)

        # Check that the expansion is correct
        expected_embeds = [[[0.1, 0.2]], [[0.3, 0.4]]] * 3
        self.assertEqual(req.input_embeds, expected_embeds)

        # Test with different n values per sample (should raise error)
        req = GenerateReqInput(
            input_embeds=[[[0.1, 0.2]], [[0.3, 0.4]]],
            sampling_params=[{"n": 2}, {"n": 3}],
        )
        with self.assertRaises(ValueError):
            req.normalize_batch_and_arguments()

    def test_lora_path_normalization(self):
        """Test normalization of lora_path."""
        # Test single lora_path with batch input
        req = GenerateReqInput(text=["Hello", "World"], lora_path="path/to/lora")

        # Define expected lora_paths before normalization
        expected_lora_paths = ["path/to/lora", "path/to/lora"]

        req.normalize_batch_and_arguments()
        self.assertEqual(req.lora_path, expected_lora_paths)

        # Test list of lora_paths
        req = GenerateReqInput(text=["Hello", "World"], lora_path=["path1", "path2"])

        # Define expected lora_paths before normalization
        expected_lora_paths = ["path1", "path2"]

        req.normalize_batch_and_arguments()
        self.assertEqual(req.lora_path, expected_lora_paths)

        # Test with parallel sampling
        req = GenerateReqInput(
            text=["Hello", "World"],
            lora_path=["path1", "path2"],
            sampling_params={"n": 2},
        )

        # Define expected lora_paths before normalization
        expected_lora_paths = ["path1", "path2"] * 2

        req.normalize_batch_and_arguments()
        self.assertEqual(req.lora_path, expected_lora_paths)

    def test_extra_key_normalization(self):
        """Test normalization of extra_key."""
        # Per-request list
        req = GenerateReqInput(
            text=["Hello", "World"],
            extra_key=["tenant-A", "tenant-B"],
            sampling_params=[{}, {}],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.extra_key, ["tenant-A", "tenant-B"])
        self.assertEqual(req[0].extra_key, "tenant-A")
        self.assertEqual(req[1].extra_key, "tenant-B")

        # Scalar broadcast
        req = GenerateReqInput(
            text=["Hello", "World"],
            extra_key="shared",
            sampling_params=[{}, {}],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.extra_key, ["shared", "shared"])

        # None stays None
        req = GenerateReqInput(text=["Hello", "World"], sampling_params=[{}, {}])
        req.normalize_batch_and_arguments()
        self.assertIsNone(req.extra_key)
        self.assertIsNone(req[0].extra_key)

        # Parallel sampling expansion
        req = GenerateReqInput(
            text=["Hello", "World"],
            extra_key=["tenant-A", "tenant-B"],
            sampling_params={"n": 2},
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.extra_key, ["tenant-A", "tenant-B"] * 2)

        # Wrong-length list
        req = GenerateReqInput(
            text=["Hello", "World"],
            extra_key=["only-one"],
            sampling_params=[{}, {}],
        )
        with self.assertRaisesRegex(ValueError, "batch size"):
            req.normalize_batch_and_arguments()

        # Non-batched scalar unchanged
        req = GenerateReqInput(text="Hello", extra_key="solo")
        req.normalize_batch_and_arguments()
        self.assertEqual(req.extra_key, "solo")

    def test_logprob_parameters_normalization(self):
        """Test normalization of logprob-related parameters."""
        # Test single example
        req = GenerateReqInput(
            text="Hello",
            return_logprob=True,
            logprob_start_len=10,
            top_logprobs_num=5,
            token_ids_logprob=[7, 8, 9],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.return_logprob, True)
        self.assertEqual(req.logprob_start_len, 10)
        self.assertEqual(req.top_logprobs_num, 5)
        self.assertEqual(req.token_ids_logprob, [7, 8, 9])

        # Test batch with scalar values
        req = GenerateReqInput(
            text=["Hello", "World"],
            return_logprob=True,
            logprob_start_len=10,
            top_logprobs_num=5,
            token_ids_logprob=[7, 8, 9],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.return_logprob, [True, True])
        self.assertEqual(req.logprob_start_len, [10, 10])
        self.assertEqual(req.top_logprobs_num, [5, 5])
        self.assertEqual(req.token_ids_logprob, [[7, 8, 9], [7, 8, 9]])

        # Test batch with list values
        req = GenerateReqInput(
            text=["Hello", "World"],
            return_logprob=[True, False],
            logprob_start_len=[10, 5],
            top_logprobs_num=[5, 3],
            token_ids_logprob=[[7, 8, 9], [4, 5, 6]],
            return_hidden_states=[False, False, True],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.return_logprob, [True, False])
        self.assertEqual(req.logprob_start_len, [10, 5])
        self.assertEqual(req.top_logprobs_num, [5, 3])
        self.assertEqual(req.token_ids_logprob, [[7, 8, 9], [4, 5, 6]])
        self.assertEqual(req.return_hidden_states, [False, False, True])

    def test_custom_logit_processor_normalization(self):
        """Test normalization of custom_logit_processor."""
        # Test single processor
        req = GenerateReqInput(
            text=["Hello", "World"], custom_logit_processor="serialized_processor"
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(
            req.custom_logit_processor, ["serialized_processor", "serialized_processor"]
        )

        # Test list of processors
        req = GenerateReqInput(
            text=["Hello", "World"], custom_logit_processor=["processor1", "processor2"]
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.custom_logit_processor, ["processor1", "processor2"])

    def test_session_params_handling(self):
        """Test handling of session_params."""
        # Test with dict
        req = GenerateReqInput(
            text=["Hello", "World"], session_params={"id": "session1", "offset": 10}
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.session_params, {"id": "session1", "offset": 10})

        # Test with list of dicts
        req = GenerateReqInput(
            text=["Hello", "World"],
            session_params=[{"id": "session1"}, {"id": "session2"}],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.session_params, [{"id": "session1"}, {"id": "session2"}])

    def test_session_id_handling(self):
        req = GenerateReqInput(
            text=["Hello", "World"],
            session_id="session1",
            sampling_params={"n": 2},
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.session_id, "session1")
        self.assertIsNone(req.session_params)
        self.assertEqual(req[2].session_id, "session1")

        with self.assertRaisesRegex(ValueError, "cannot both be set"):
            GenerateReqInput(
                text="Hello",
                session_id="explicit",
                session_params={"id": "legacy"},
            ).normalize_batch_and_arguments()

    def test_getitem_method(self):
        """Test the __getitem__ method."""
        req = GenerateReqInput(
            text=["Hello", "World"],
            image_data=[["img1.jpg"], ["img2.jpg"]],
            audio_data=["audio1.mp3", "audio2.mp3"],
            sampling_params=[{"temp": 0.7}, {"temp": 0.8}],
            rid=["id1", "id2"],
            return_logprob=[True, False],
            logprob_start_len=[10, 5],
            top_logprobs_num=[5, 3],
            token_ids_logprob=[[7, 8, 9], [4, 5, 6]],
            stream=True,
            log_metrics=True,
            modalities=["image", "image"],
            lora_path=["path1", "path2"],
            custom_logit_processor=["processor1", "processor2"],
            return_hidden_states=True,
        )
        req.normalize_batch_and_arguments()

        # Get the first item
        item0 = req[0]
        self.assertEqual(item0.text, "Hello")
        self.assertEqual(item0.image_data, ["img1.jpg"])
        self.assertEqual(item0.audio_data, "audio1.mp3")
        self.assertEqual(item0.sampling_params, {"temp": 0.7})
        self.assertEqual(item0.rid, "id1")
        self.assertEqual(item0.return_logprob, True)
        self.assertEqual(item0.logprob_start_len, 10)
        self.assertEqual(item0.top_logprobs_num, 5)
        self.assertEqual(item0.token_ids_logprob, [7, 8, 9])
        self.assertEqual(item0.stream, True)
        self.assertEqual(item0.log_metrics, True)
        self.assertEqual(item0.modalities, "image")
        self.assertEqual(item0.lora_path, "path1")
        self.assertEqual(item0.custom_logit_processor, "processor1")
        self.assertEqual(item0.return_hidden_states, True)

    def test_getitem_preserves_return_prompt_token_ids(self):
        """Batch subrequests must keep the prompt-token-id return flag."""
        req = GenerateReqInput(
            input_ids=[[1, 2, 3], [4, 5, 6]],
            sampling_params=[{}, {}],
            rid=["id1", "id2"],
            return_prompt_token_ids=True,
        )
        req.normalize_batch_and_arguments()

        self.assertTrue(req[0].return_prompt_token_ids)
        self.assertTrue(req[1].return_prompt_token_ids)

    def test_regenerate_rid(self):
        """Test the regenerate_rid method."""
        req = GenerateReqInput(text="Hello")
        req.normalize_batch_and_arguments()

        original_rid = req.rid
        new_rid = req.regenerate_rid()

        self.assertNotEqual(original_rid, new_rid)
        self.assertEqual(req.rid, new_rid)

    def test_error_cases(self):
        """Test various error cases."""
        # Test when neither text, input_ids, nor input_embeds is provided
        with self.assertRaises(ValueError):
            req = GenerateReqInput()
            req.normalize_batch_and_arguments()

        # Test when all of text, input_ids, and input_embeds are provided
        with self.assertRaises(ValueError):
            req = GenerateReqInput(
                text="Hello", input_ids=[1, 2, 3], input_embeds=[[0.1, 0.2]]
            )
            req.normalize_batch_and_arguments()


if __name__ == "__main__":
    unittest.main()
