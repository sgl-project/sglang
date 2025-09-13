import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.test.test_utils import CustomTestCase


class MockTokenToKVPool:
    """Mock for token_to_kv_pool with configurable head_num."""

    def __init__(self, head_num=32, page_size=1):
        self.head_num = head_num
        self.page_size = page_size
        self.start_layer = 0

    def get_contiguous_buf_infos(self):
        """Mock method for getting buffer info."""
        return [], [], []


class MockMLATokenToKVPool:
    """Mock for MLA token_to_kv_pool (no head_num attribute)."""

    def __init__(self, page_size=1):
        self.page_size = page_size
        self.start_layer = 0

    def get_contiguous_buf_infos(self):
        """Mock method for getting buffer info."""
        return [], [], []


class MockModelConfig:
    """Mock for model config with num_attention_heads."""

    def __init__(self, num_attention_heads=64):
        self.num_attention_heads = num_attention_heads


class MockScheduler:
    """Mock for scheduler with model config and other attributes."""

    def __init__(self, model_config, dp_rank=0, gpu_id=0):
        self.model_config = model_config
        self.dp_rank = dp_rank
        self.gpu_id = gpu_id
        self.server_args = Mock()
        self.server_args.disaggregation_ib_device = "mlx5_roce0"


class MockMetadataBuffers:
    """Mock for metadata buffers."""

    def get_buf_infos(self):
        return [], [], []


class MockReqToMetadataIdxAllocator:
    """Mock for request to metadata index allocator."""

    pass


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestKvHeadInitialization(CustomTestCase):
    """Test kv_head_num initialization logic in PrefillBootstrapQueue."""

    def setUp(self):
        """Set up common test data."""
        self.tp_rank = 0
        self.tp_size = 1
        self.gpu_id = 0
        self.bootstrap_port = 8000
        self.max_total_num_tokens = 1000
        self.decode_tp_size = 1
        self.decode_dp_size = 1
        self.pp_rank = 0
        self.pp_size = 1
        self.transfer_backend = TransferBackend.FAKE

        # Mock gloo group
        self.gloo_group = Mock()

        # Mock metadata buffers
        self.metadata_buffers = MockMetadataBuffers()

        # Mock request to metadata index allocator
        self.req_to_metadata_buffer_idx_allocator = MockReqToMetadataIdxAllocator()

    def _create_prefill_bootstrap_queue(
        self, token_to_kv_pool, model_config, draft_token_to_kv_pool=None
    ):
        """Helper method to create PrefillBootstrapQueue instance."""
        scheduler = MockScheduler(model_config)

        return PrefillBootstrapQueue(
            token_to_kv_pool=token_to_kv_pool,
            draft_token_to_kv_pool=draft_token_to_kv_pool,
            req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
            metadata_buffers=self.metadata_buffers,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            gpu_id=self.gpu_id,
            bootstrap_port=self.bootstrap_port,
            gloo_group=self.gloo_group,
            max_total_num_tokens=self.max_total_num_tokens,
            decode_tp_size=self.decode_tp_size,
            decode_dp_size=self.decode_dp_size,
            scheduler=scheduler,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            transfer_backend=self.transfer_backend,
        )

    @patch("sglang.srt.disaggregation.prefill.get_kv_class")
    def test_mha_backend_initialization(self, mock_get_kv_class):
        """Test that kv_head_num is correctly set from token_to_kv_pool for MHA backend."""
        # Given: MHA backend (non-MLA) with specific head_num
        expected_head_num = 32
        token_to_kv_pool = MockTokenToKVPool(head_num=expected_head_num)
        model_config = MockModelConfig(num_attention_heads=64)

        # Mock the kv_args class and instance
        mock_kv_args_class = Mock()
        mock_kv_args = Mock()
        mock_kv_args_class.return_value = mock_kv_args
        mock_get_kv_class.return_value = mock_kv_args_class

        # When: Creating PrefillBootstrapQueue (which calls _init_kv_manager)
        prefill_queue = self._create_prefill_bootstrap_queue(
            token_to_kv_pool, model_config
        )

        # Then: kv_head_num should be set from token_to_kv_pool.head_num
        self.assertEqual(mock_kv_args.kv_head_num, expected_head_num)

        # Verify that is_mla_backend is False
        self.assertFalse(prefill_queue.is_mla_backend)

    @patch("sglang.srt.disaggregation.prefill.get_kv_class")
    @patch("sglang.srt.utils.get_attention_tp_size")
    def test_mla_backend_initialization(
        self, mock_get_attention_tp_size, mock_get_kv_class
    ):
        """Test that kv_head_num is correctly calculated for MLA backend."""
        # Given: MLA backend with specific attention heads and TP size
        num_attention_heads = 64
        tp_size = 2
        expected_head_num = num_attention_heads // tp_size  # 32

        token_to_kv_pool = MockMLATokenToKVPool()  # MLA pool has no head_num
        model_config = MockModelConfig(num_attention_heads=num_attention_heads)

        # Mock get_attention_tp_size to return tp_size
        mock_get_attention_tp_size.return_value = tp_size

        # Mock the kv_args class and instance
        mock_kv_args_class = Mock()
        mock_kv_args = Mock()
        mock_kv_args_class.return_value = mock_kv_args
        mock_get_kv_class.return_value = mock_kv_args_class

        # When: Creating PrefillBootstrapQueue (which calls _init_kv_manager)
        prefill_queue = self._create_prefill_bootstrap_queue(
            token_to_kv_pool, model_config
        )

        # Then: kv_head_num should be calculated from model config
        self.assertEqual(mock_kv_args.kv_head_num, expected_head_num)

        # Verify that is_mla_backend is True
        self.assertTrue(prefill_queue.is_mla_backend)

        # Verify that get_attention_tp_size was called
        mock_get_attention_tp_size.assert_called_once()

    @patch("sglang.srt.disaggregation.prefill.get_kv_class")
    def test_mla_backend_with_different_tp_sizes(self, mock_get_kv_class):
        """Test MLA backend initialization with different tensor parallelism sizes."""
        # Given: MLA backend with different TP sizes
        test_cases = [
            (64, 1, 64),  # 64 heads, TP=1 -> 64 heads per rank
            (64, 2, 32),  # 64 heads, TP=2 -> 32 heads per rank
            (64, 4, 16),  # 64 heads, TP=4 -> 16 heads per rank
            (32, 2, 16),  # 32 heads, TP=2 -> 16 heads per rank
        ]

        for num_attention_heads, tp_size, expected_head_num in test_cases:
            with self.subTest(
                num_attention_heads=num_attention_heads,
                tp_size=tp_size,
                expected_head_num=expected_head_num,
            ):
                token_to_kv_pool = MockMLATokenToKVPool()
                model_config = MockModelConfig(num_attention_heads=num_attention_heads)

                # Mock get_attention_tp_size
                with patch(
                    "sglang.srt.utils.get_attention_tp_size", return_value=tp_size
                ):
                    # Mock the kv_args class and instance
                    mock_kv_args_class = Mock()
                    mock_kv_args = Mock()
                    mock_kv_args_class.return_value = mock_kv_args
                    mock_get_kv_class.return_value = mock_kv_args_class

                    # When: Creating PrefillBootstrapQueue
                    prefill_queue = self._create_prefill_bootstrap_queue(
                        token_to_kv_pool, model_config
                    )

                    # Then: kv_head_num should be calculated correctly
                    self.assertEqual(mock_kv_args.kv_head_num, expected_head_num)
                    self.assertTrue(prefill_queue.is_mla_backend)

    @patch("sglang.srt.disaggregation.prefill.get_kv_class")
    def test_mha_backend_with_different_head_nums(self, mock_get_kv_class):
        """Test MHA backend initialization with different head_num values."""
        # Given: MHA backend with different head_num values
        test_cases = [16, 32, 64, 128]

        for head_num in test_cases:
            with self.subTest(head_num=head_num):
                token_to_kv_pool = MockTokenToKVPool(head_num=head_num)
                model_config = MockModelConfig(num_attention_heads=64)

                # Mock the kv_args class and instance
                mock_kv_args_class = Mock()
                mock_kv_args = Mock()
                mock_kv_args_class.return_value = mock_kv_args
                mock_get_kv_class.return_value = mock_kv_args_class

                # When: Creating PrefillBootstrapQueue
                prefill_queue = self._create_prefill_bootstrap_queue(
                    token_to_kv_pool, model_config
                )

                # Then: kv_head_num should be set from token_to_kv_pool
                self.assertEqual(mock_kv_args.kv_head_num, head_num)
                self.assertFalse(prefill_queue.is_mla_backend)

    def test_is_mla_backend_detection(self):
        """Test that is_mla_backend is correctly detected based on token_to_kv_pool type."""
        # Test MHA backend detection
        mha_pool = MockTokenToKVPool()
        model_config = MockModelConfig()

        with patch("sglang.srt.disaggregation.prefill.get_kv_class"):
            prefill_queue = self._create_prefill_bootstrap_queue(mha_pool, model_config)
            self.assertFalse(prefill_queue.is_mla_backend)

        # Test MLA backend detection
        mla_pool = MockMLATokenToKVPool()

        with patch("sglang.srt.disaggregation.prefill.get_kv_class"):
            prefill_queue = self._create_prefill_bootstrap_queue(mla_pool, model_config)
            self.assertTrue(prefill_queue.is_mla_backend)


if __name__ == "__main__":
    unittest.main()
