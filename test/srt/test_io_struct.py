import copy
import unittest

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
)


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
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.return_logprob, [True, False])
        self.assertEqual(req.logprob_start_len, [10, 5])
        self.assertEqual(req.top_logprobs_num, [5, 3])
        self.assertEqual(req.token_ids_logprob, [[7, 8, 9], [4, 5, 6]])

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

    def test_multiple_input_formats(self):
        """Test different combinations of input formats."""
        # Test with text only
        req = GenerateReqInput(text="Hello")
        req.normalize_batch_and_arguments()
        self.assertTrue(req.is_single)

        # Test with input_ids only
        req = GenerateReqInput(input_ids=[1, 2, 3])
        req.normalize_batch_and_arguments()
        self.assertTrue(req.is_single)

        # Test with input_embeds only
        req = GenerateReqInput(input_embeds=[[0.1, 0.2]])
        req.normalize_batch_and_arguments()
        self.assertTrue(req.is_single)


if __name__ == "__main__":
    unittest.main()
