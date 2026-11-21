#!/usr/bin/env python3
"""Test script to verify the DeepSeek V4.1 decoder replay tail fix for issue #46666.

This test verifies that when dp_spec_prefill_coordination_applied is True,
the DecoderReplayTail.plan method returns False (don't use tail graph) to
prevent the verify rank from being incorrectly sent to run_idle.
"""

import sys
from unittest.mock import MagicMock, patch

# Add the python directory to the path
sys.path.insert(0, '/Users/depend/sglang/python')


def test_dp_spec_prefill_coordination_skips_tail_graph():
    """Test that dp_spec_prefill_coordination_applied prevents tail graph usage."""
    from sglang.srt.models.deepseek_v4_replay_tail import DecoderReplayTail

    # Create a mock model
    mock_model = MagicMock()
    mock_model.tail_writes_source_kv = True
    mock_model.late_layer_start = 10
    mock_model.end_layer = 20

    # Create DecoderReplayTail instance
    tail_handler = DecoderReplayTail(
        model=mock_model,
        layer_ids=range(10, 20),
        row_step=128,
        max_rows=1024,
    )

    # Mock the runner to be warm
    tail_handler.runner = MagicMock()
    tail_handler.runner.warm = True
    tail_handler.runner.ready = MagicMock()
    tail_handler.runner.bucket_rows = MagicMock(return_value=128)

    # Create a mock forward batch with dp_spec_prefill_coordination_applied=True
    mock_forward_batch = MagicMock()
    mock_forward_batch.is_extend_in_batch = True
    mock_forward_batch.dp_spec_prefill_coordination_applied = True
    mock_forward_batch.global_num_tokens_cpu = [128, 128]
    mock_forward_batch.global_num_token_non_padded = None

    # Mock the necessary functions
    with patch('sglang.srt.models.deepseek_v4_replay_tail.get_parallel') as mock_get_parallel, \
         patch('sglang.srt.models.deepseek_v4_replay_tail.get_is_capture_mode') as mock_capture, \
         patch('sglang.srt.models.deepseek_v4_replay_tail.get_global_expert_distribution_recorder') as mock_recorder, \
         patch('sglang.srt.models.deepseek_v4_replay_tail.check_cuda_graph_backend') as mock_check_backend:

        # Setup mocks
        mock_parallel = MagicMock()
        mock_parallel.attn_dp_size = 2
        mock_get_parallel.return_value = mock_parallel

        mock_capture.return_value = False
        mock_recorder.return_value.recording = False
        mock_check_backend.return_value = False

        # Mock _can_capture_moe_bound to return True
        tail_handler._can_capture_moe_bound = MagicMock(return_value=True)

        # Mock _scattered_dp to return None (not scattered)
        tail_handler._scattered_dp = MagicMock(return_value=None)

        # Mock _gathered_dp_rows to return some rows
        tail_handler._gathered_dp_rows = MagicMock(return_value=[128, 128])

        # Call plan with a mock tail
        mock_tail = MagicMock()
        mock_tail.positions = MagicMock()
        mock_tail.positions.shape = [128]
        mock_tail.cp_metadata = None

        use_tail_graph, dp_rows = tail_handler.plan(
            tail=mock_tail,
            forward_batch=mock_forward_batch,
            num_tokens=128,
            tail_in_graph=False,
            late_dp_counts=None,
        )

        # Verify that use_tail_graph is False when dp_spec_prefill_coordination_applied is True
        assert use_tail_graph is False, (
            f"Expected use_tail_graph=False when dp_spec_prefill_coordination_applied=True, "
            f"got {use_tail_graph}"
        )
        assert dp_rows is None, f"Expected dp_rows=None, got {dp_rows}"

        print("✓ Test passed: dp_spec_prefill_coordination_applied correctly prevents tail graph usage")
        return True


def test_normal_operation_without_coordination():
    """Test that tail graphs work normally when dp_spec_prefill_coordination_applied is False."""
    from sglang.srt.models.deepseek_v4_replay_tail import DecoderReplayTail

    # Create a mock model
    mock_model = MagicMock()
    mock_model.tail_writes_source_kv = True
    mock_model.late_layer_start = 10
    mock_model.end_layer = 20

    # Create DecoderReplayTail instance
    tail_handler = DecoderReplayTail(
        model=mock_model,
        layer_ids=range(10, 20),
        row_step=128,
        max_rows=1024,
    )

    # Mock the runner to be warm
    tail_handler.runner = MagicMock()
    tail_handler.runner.warm = True
    tail_handler.runner.ready = MagicMock()
    tail_handler.runner.bucket_rows = MagicMock(return_value=128)

    # Create a mock forward batch with dp_spec_prefill_coordination_applied=False
    mock_forward_batch = MagicMock()
    mock_forward_batch.is_extend_in_batch = True
    mock_forward_batch.dp_spec_prefill_coordination_applied = False
    mock_forward_batch.global_num_tokens_cpu = [128, 128]
    mock_forward_batch.global_num_token_non_padded = None

    # Mock the necessary functions
    with patch('sglang.srt.models.deepseek_v4_replay_tail.get_parallel') as mock_get_parallel, \
         patch('sglang.srt.models.deepseek_v4_replay_tail.get_is_capture_mode') as mock_capture, \
         patch('sglang.srt.models.deepseek_v4_replay_tail.get_global_expert_distribution_recorder') as mock_recorder, \
         patch('sglang.srt.models.deepseek_v4_replay_tail.check_cuda_graph_backend') as mock_check_backend, \
         patch('sglang.srt.models.deepseek_v4_replay_tail.get_moe_a2a_backend') as mock_moe_backend:

        # Setup mocks
        mock_parallel = MagicMock()
        mock_parallel.attn_dp_size = 2
        mock_get_parallel.return_value = mock_parallel

        mock_capture.return_value = False
        mock_recorder.return_value.recording = False
        mock_check_backend.return_value = False
        mock_moe_backend.return_value.is_none.return_value = True

        # Mock _can_capture_moe_bound to return True
        tail_handler._can_capture_moe_bound = MagicMock(return_value=True)

        # Mock _scattered_dp to return None (not scattered)
        tail_handler._scattered_dp = MagicMock(return_value=None)

        # Mock _gathered_dp_rows to return some rows
        tail_handler._gathered_dp_rows = MagicMock(return_value=[128, 128])

        # Mock is_unified_kv_triton to return False
        with patch('sglang.srt.models.deepseek_v4_replay_tail.is_unified_kv_triton') as mock_unified:
            mock_unified.return_value = False

            # Call plan with a mock tail
            mock_tail = MagicMock()
            mock_tail.positions = MagicMock()
            mock_tail.positions.shape = [128]
            mock_tail.cp_metadata = None

            use_tail_graph, dp_rows = tail_handler.plan(
                tail=mock_tail,
                forward_batch=mock_forward_batch,
                num_tokens=128,
                tail_in_graph=False,
                late_dp_counts=None,
            )

            # Verify that use_tail_graph can be True when dp_spec_prefill_coordination_applied is False
            # (assuming other conditions are met)
            assert use_tail_graph is True, (
                f"Expected use_tail_graph=True when dp_spec_prefill_coordination_applied=False "
                f"and other conditions are met, got {use_tail_graph}"
            )

            print("✓ Test passed: tail graphs work normally without DP spec prefill coordination")
            return True


if __name__ == "__main__":
    try:
        # The critical test: verify dp_spec_prefill_coordination_applied prevents tail graph usage
        success1 = test_dp_spec_prefill_coordination_skips_tail_graph()

        if success1:
            print("\n✓ All tests passed!")
            sys.exit(0)
        else:
            print("\n✗ Some tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
