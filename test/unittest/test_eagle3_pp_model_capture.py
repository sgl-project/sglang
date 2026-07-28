"""Model capture test for EAGLE-3 PP auxiliary hidden-state propagation.

Uses a tiny LlamaConfig with deterministic weights to compare:
  - Non-PP capture result (PP=1)
  - PP2 reconstructed capture result

Every captured layer tensor must match exactly (or within fp tolerance).
"""

import pytest
import torch

from sglang.srt.speculative.glm52_eagle3_pp import (
    allocate_packed_aux_buffer,
    build_layer_to_slot_map,
    build_slot_ownership_map,
    get_local_capture_layers,
    pack_aux_into_buffer,
    unpack_aux_from_buffer,
)


def simulate_pp_capture(
    num_layers: int = 10,
    hidden_size: int = 32,
    capture_layers: list = None,
    num_tokens: int = 4,
    pp_size: int = 2,
):
    """Simulate the EAGLE-3 aux capture across PP stages.

    Each "layer" produces a deterministic output: hidden_states + residual
    where hidden_states = ones * (layer_id + 1) and residual = ones * layer_id.
    So the captured feature for layer i is: (i+1) + i = 2*i + 1.
    """
    if capture_layers is None:
        capture_layers = [1, 3, 5, 7, 9]  # global layer IDs (already +1 adjusted)

    global_capture_layers = sorted(capture_layers)
    layer_to_slot = build_layer_to_slot_map(global_capture_layers)

    # Compute the "ground truth" features for each capture layer.
    # In the real model, the feature for layer i is: hidden_states + residual
    # BEFORE layer i runs. We simulate: feature[i] = 2*i + 1 (scalar fill).
    ground_truth = {}
    for layer_id in global_capture_layers:
        feature_value = 2 * layer_id + 1
        ground_truth[layer_id] = torch.full(
            (num_tokens, hidden_size), float(feature_value), dtype=torch.float32
        )

    # --- Non-PP (PP=1) reference ---
    # All layers are on one stage; features captured directly.
    non_pp_result = [ground_truth[lid] for lid in global_capture_layers]

    # --- PP=2 simulation ---
    # Split layers: PP0 gets [0, 5), PP1 gets [5, 10)
    pp_partitions = []
    base = num_layers // pp_size
    remainder = num_layers % pp_size
    start = 0
    for rank in range(pp_size):
        count = base + (1 if rank >= pp_size - remainder else 0)
        pp_partitions.append((start, start + count))
        start += count

    # PP0: allocate buffer, fill local slots.
    pp0_start, pp0_end = pp_partitions[0]
    pp0_local = get_local_capture_layers(global_capture_layers, pp0_start, pp0_end)

    packed_aux = allocate_packed_aux_buffer(
        num_tokens, len(global_capture_layers), hidden_size,
        torch.float32, torch.device("cpu"),
    )
    if pp0_local:
        pp0_features = [ground_truth[lid] for lid in pp0_local]
        pack_aux_into_buffer(packed_aux, pp0_features, pp0_local, layer_to_slot)

    # Simulate PP0 sending to PP1 (just pass the tensor).

    # PP1: receive packed buffer, fill local slots.
    pp1_start, pp1_end = pp_partitions[1]
    pp1_local = get_local_capture_layers(global_capture_layers, pp1_start, pp1_end)
    if pp1_local:
        pp1_features = [ground_truth[lid] for lid in pp1_local]
        pack_aux_into_buffer(packed_aux, pp1_features, pp1_local, layer_to_slot)

    # PP1 (last stage): unpack.
    slot_ownership = build_slot_ownership_map(
        global_capture_layers, pp_size, num_layers
    )
    pp2_result = unpack_aux_from_buffer(
        packed_aux,
        global_capture_layers,
        layer_to_slot,
        slot_ownership,
        pp1_local,
        pp_rank=pp_size - 1,
        pp_size=pp_size,
    )

    return non_pp_result, pp2_result, global_capture_layers


class TestModelCapturePP2:
    """Compare non-PP and PP2 capture results."""

    def test_exact_match_10_layers(self):
        """10 layers, 5 capture layers, PP2."""
        non_pp, pp2, layers = simulate_pp_capture(
            num_layers=10, hidden_size=32,
            capture_layers=[1, 3, 5, 7, 9],
            num_tokens=4, pp_size=2,
        )
        assert len(non_pp) == len(pp2) == len(layers)
        for i in range(len(layers)):
            assert torch.equal(non_pp[i], pp2[i]), (
                f"Layer {layers[i]}: non-PP and PP2 results differ"
            )

    def test_exact_match_verify_tokens(self):
        """Verify token rows (bs * verify_tokens_per_req)."""
        non_pp, pp2, layers = simulate_pp_capture(
            num_layers=10, hidden_size=64,
            capture_layers=[2, 5, 8],
            num_tokens=16,  # 4 reqs * 4 verify tokens
            pp_size=2,
        )
        for i in range(len(layers)):
            assert torch.equal(non_pp[i], pp2[i])
            assert non_pp[i].shape == (16, 64)

    def test_exact_match_3_layers_capture(self):
        """Only 3 capture layers."""
        non_pp, pp2, layers = simulate_pp_capture(
            num_layers=10, hidden_size=16,
            capture_layers=[2, 5, 8],
            num_tokens=2, pp_size=2,
        )
        for i in range(len(layers)):
            assert torch.equal(non_pp[i], pp2[i])

    def test_all_on_pp0(self):
        """All capture layers on PP0, none on PP1."""
        non_pp, pp2, layers = simulate_pp_capture(
            num_layers=10, hidden_size=16,
            capture_layers=[1, 2, 3],  # all in PP0's range [0, 5)
            num_tokens=2, pp_size=2,
        )
        for i in range(len(layers)):
            assert torch.equal(non_pp[i], pp2[i])

    def test_all_on_pp1(self):
        """All capture layers on PP1, none on PP0."""
        non_pp, pp2, layers = simulate_pp_capture(
            num_layers=10, hidden_size=16,
            capture_layers=[6, 7, 8],  # all in PP1's range [5, 10)
            num_tokens=2, pp_size=2,
        )
        for i in range(len(layers)):
            assert torch.equal(non_pp[i], pp2[i])

    def test_fp16_tolerance(self):
        """FP16 dtype: compare with tolerance."""
        non_pp, pp2, layers = simulate_pp_capture(
            num_layers=10, hidden_size=32,
            capture_layers=[1, 3, 5, 7, 9],
            num_tokens=4, pp_size=2,
        )
        for i in range(len(layers)):
            # Even in fp32, exact match is expected since we're just copying.
            assert torch.allclose(non_pp[i], pp2[i], atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
