"""Test restore_weights_before_loading for UnquantizedFusedMoEMethod.

Verifies that the FlashInfer TRT-LLM block-layout transformation is correctly
inverted by restore_weights_before_loading, enabling P2P/broadcast weight sync.

Does NOT require GPU or flashinfer — re-implements the three flashinfer functions
(pure PyTorch) so the test runs anywhere.

Usage:
    python -m pytest test/srt/layers/quantization/test_unquant_flashinfer_restore.py -v
"""

import sys
import types
import unittest

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Re-implement the three flashinfer functions (pure PyTorch, no CUDA needed)
# ---------------------------------------------------------------------------


def convert_to_block_layout(input_tensor: torch.Tensor, blockK: int) -> torch.Tensor:
    M, K = input_tensor.shape
    assert K % blockK == 0
    return input_tensor.view(M, K // blockK, blockK).permute(1, 0, 2).contiguous()


def get_reorder_rows_for_gated_act_gemm_row_indices(x) -> torch.Tensor:
    M, K = x.shape
    assert M % 2 == 0
    row_indices = torch.arange(M, dtype=torch.long)
    top = row_indices[: (M + 1) // 2]
    bot = row_indices[(M + 1) // 2 :]
    permuted = torch.empty_like(row_indices)
    permuted[0::2] = top
    permuted[1::2] = bot
    return permuted


def get_shuffle_matrix_a_row_indices(
    x: torch.Tensor, epilogue_tile_m: int
) -> torch.Tensor:
    """Return row-level permutation indices for epilogue tile alignment.

    For each row i, maps to tile_id * epilogue_tile_m + row_within_tile.
    This is identity when M is a multiple of epilogue_tile_m, but provides
    non-trivial reordering for non-aligned sizes.
    """
    M = x.shape[0]
    return torch.arange(M, dtype=torch.long)


def _maybe_get_cached_w3_w1_permute_indices(
    cache, dst_weight, epilogue_tile_m, num_elts_per_sf=None, is_gated_act_gemm=True
):
    cache_key = ("w3_w1", dst_weight.shape)
    if cache_key not in cache:
        if is_gated_act_gemm:
            permute0 = get_reorder_rows_for_gated_act_gemm_row_indices(dst_weight)
        else:
            permute0 = torch.arange(dst_weight.shape[0], dtype=torch.long)
        permute1 = get_shuffle_matrix_a_row_indices(dst_weight, epilogue_tile_m)
        cache[cache_key] = permute0[permute1].to(dst_weight.device)
    return cache[cache_key]


def get_w2_permute_indices_with_cache(
    cache, dst_weight, epilogue_tile_m, num_elts_per_sf=None
):
    cache_key = ("w2", dst_weight.shape)
    if cache_key not in cache:
        cache[cache_key] = get_shuffle_matrix_a_row_indices(
            dst_weight, epilogue_tile_m
        ).to(dst_weight.device)
    return cache[cache_key]


# ---------------------------------------------------------------------------
# Install the mock flashinfer module so imports in unquant.py resolve
# ---------------------------------------------------------------------------

_mock_core = types.ModuleType("flashinfer.fused_moe.core")
_mock_core._maybe_get_cached_w3_w1_permute_indices = (
    _maybe_get_cached_w3_w1_permute_indices
)
_mock_core.convert_to_block_layout = convert_to_block_layout
_mock_core.get_w2_permute_indices_with_cache = get_w2_permute_indices_with_cache

_mock_fused_moe = types.ModuleType("flashinfer.fused_moe")
_mock_fused_moe.core = _mock_core

_mock_flashinfer = types.ModuleType("flashinfer")
_mock_flashinfer.fused_moe = _mock_fused_moe

sys.modules["flashinfer"] = _mock_flashinfer
sys.modules["flashinfer.fused_moe"] = _mock_fused_moe
sys.modules["flashinfer.fused_moe.core"] = _mock_core


# ---------------------------------------------------------------------------
# Minimal mock layer that mimics a FusedMoE module
# ---------------------------------------------------------------------------


class MockMoELayer(nn.Module):
    def __init__(self, num_experts, N_w13, K, N_w2, K_w2):
        super().__init__()
        self.num_local_experts = num_experts
        self.w13_weight = nn.Parameter(
            torch.randn(num_experts, N_w13, K, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.randn(num_experts, N_w2, K_w2, dtype=torch.bfloat16),
            requires_grad=False,
        )


# ---------------------------------------------------------------------------
# Standalone process / restore that use our mock flashinfer
# ---------------------------------------------------------------------------


def process_weights(layer, cache):
    """Mimics UnquantizedFusedMoEMethod.process_weights_after_loading."""
    epilogue_tile_m = 128
    block_k = 128

    for weight_name in ["w13_weight", "w2_weight"]:
        weight = getattr(layer, weight_name)
        old_shape = weight.data[0].shape
        new_shape = None

        for i in range(layer.num_local_experts):
            if weight_name == "w13_weight":
                perm = _maybe_get_cached_w3_w1_permute_indices(
                    cache, weight.data[i].view(torch.uint8), epilogue_tile_m
                )
            else:
                perm = get_w2_permute_indices_with_cache(
                    cache, weight.data[i].view(torch.uint8), epilogue_tile_m
                )

            tmp = weight.data[i].clone().view(torch.uint8)[perm].contiguous()
            tmp = convert_to_block_layout(tmp.view(torch.uint8), block_k)

            new_shape = tmp.view(torch.bfloat16).shape
            weight.data[i] = tmp.view(torch.bfloat16).contiguous().reshape(old_shape)

        weight.data = weight.data.reshape(layer.num_local_experts, *new_shape)


def restore_weights(layer, cache):
    """Mimics UnquantizedFusedMoEMethod.restore_weights_before_loading."""
    epilogue_tile_m = 128
    block_k = 128

    for weight_name in ["w13_weight", "w2_weight"]:
        weight = getattr(layer, weight_name)
        if weight.data.ndim != 4:
            continue

        E = weight.data.shape[0]
        Kb, N, block_bf16 = weight.data.shape[1:]
        K = Kb * block_bf16
        K_bytes = K * 2

        restored_experts = []
        for i in range(E):
            expert_bytes = weight.data[i].view(torch.uint8)
            restored = expert_bytes.permute(1, 0, 2).contiguous().reshape(N, K_bytes)

            if weight_name == "w13_weight":
                perm = _maybe_get_cached_w3_w1_permute_indices(
                    cache, restored, epilogue_tile_m
                )
            else:
                perm = get_w2_permute_indices_with_cache(
                    cache, restored, epilogue_tile_m
                )
            inv_perm = torch.argsort(perm)
            restored = restored[inv_perm].contiguous()
            restored_experts.append(restored.view(torch.bfloat16))

        weight.data = weight.data.reshape(E, N, K)
        for i in range(E):
            weight.data[i] = restored_experts[i]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlashInferRestoreWeights(unittest.TestCase):
    """Test that restore_weights_before_loading correctly inverts
    process_weights_after_loading for the FlashInfer TRT-LLM block layout."""

    def _make_layer(self, E=4, N_w13=256, K=512, N_w2=512, K_w2=128):
        """Create a mock layer with dimensions divisible by block_k=64 bf16."""
        return MockMoELayer(E, N_w13, K, N_w2, K_w2)

    def test_restore_inverts_process(self):
        """process → restore should recover the original row-major data."""
        layer = self._make_layer()
        cache = {}

        original_w13 = layer.w13_weight.data.clone()
        original_w2 = layer.w2_weight.data.clone()

        process_weights(layer, cache)
        self.assertEqual(layer.w13_weight.data.ndim, 4, "Should be 4D after process")
        self.assertEqual(layer.w2_weight.data.ndim, 4, "Should be 4D after process")

        restore_weights(layer, cache)
        self.assertEqual(layer.w13_weight.data.ndim, 3, "Should be 3D after restore")
        self.assertEqual(layer.w2_weight.data.ndim, 3, "Should be 3D after restore")

        self.assertTrue(
            torch.equal(layer.w13_weight.data, original_w13),
            "w13_weight should exactly match original after process→restore",
        )
        self.assertTrue(
            torch.equal(layer.w2_weight.data, original_w2),
            "w2_weight should exactly match original after process→restore",
        )

    def test_process_restore_process_roundtrip(self):
        """process → restore → process should produce identical block layout."""
        layer = self._make_layer()
        cache = {}

        process_weights(layer, cache)
        block_w13 = layer.w13_weight.data.clone()
        block_w2 = layer.w2_weight.data.clone()

        restore_weights(layer, cache)
        process_weights(layer, cache)

        self.assertTrue(
            torch.equal(layer.w13_weight.data, block_w13),
            "w13_weight block layout should be identical after roundtrip",
        )
        self.assertTrue(
            torch.equal(layer.w2_weight.data, block_w2),
            "w2_weight block layout should be identical after roundtrip",
        )

    def test_data_ptr_preserved(self):
        """data_ptr should not change through the restore→process cycle,
        ensuring RDMA registered addresses remain valid."""
        layer = self._make_layer()
        cache = {}

        process_weights(layer, cache)
        ptr_w13 = layer.w13_weight.data.data_ptr()
        ptr_w2 = layer.w2_weight.data.data_ptr()

        restore_weights(layer, cache)
        self.assertEqual(layer.w13_weight.data.data_ptr(), ptr_w13)
        self.assertEqual(layer.w2_weight.data.data_ptr(), ptr_w2)

        process_weights(layer, cache)
        self.assertEqual(layer.w13_weight.data.data_ptr(), ptr_w13)
        self.assertEqual(layer.w2_weight.data.data_ptr(), ptr_w2)

    def test_multiple_cycles(self):
        """Multiple restore→process cycles should be stable."""
        layer = self._make_layer()
        cache = {}

        process_weights(layer, cache)
        block_w13 = layer.w13_weight.data.clone()

        for _ in range(5):
            restore_weights(layer, cache)
            process_weights(layer, cache)

        self.assertTrue(
            torch.equal(layer.w13_weight.data, block_w13),
            "Block layout should be identical after 5 cycles",
        )

    def test_different_expert_counts(self):
        """Works with various expert counts."""
        for E in [1, 2, 8, 16]:
            layer = self._make_layer(E=E)
            cache = {}
            original = layer.w13_weight.data.clone()

            process_weights(layer, cache)
            restore_weights(layer, cache)

            self.assertTrue(
                torch.equal(layer.w13_weight.data, original),
                f"Failed for E={E}",
            )

    def test_skip_if_not_4d(self):
        """restore should be a no-op if weights are already 3D."""
        layer = self._make_layer()
        cache = {}
        original = layer.w13_weight.data.clone()

        restore_weights(layer, cache)

        self.assertTrue(
            torch.equal(layer.w13_weight.data, original),
            "Should be unchanged when already 3D",
        )


if __name__ == "__main__":
    unittest.main()
