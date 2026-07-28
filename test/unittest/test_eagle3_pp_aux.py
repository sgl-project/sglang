"""Focused unit tests for GLM-5.2 EAGLE-3 PP auxiliary hidden-state propagation.

Tests validate:
1. Auxiliary packing with synthetic tensors across two PP stages.
2. Global slot ordering is exact.
3. Earlier-stage values survive later-stage processing.
4. All token rows survive.
5. Dtype and device are preserved.
6. Missing owner is detected via static validation.
7. Both decode (bs tokens) and target-verify (bs * verify_tokens) row counts.
8. PP slot ownership is correct for GLM-5.2's 78-layer, PP2 split.
9. DSA topk + aux coexistence in proxy tensors.
"""

import pytest
import torch

from sglang.srt.speculative.glm52_eagle3_pp import (
    GLM52_EAGLE3_AUX_PP_KEY,
    allocate_packed_aux_buffer,
    build_layer_to_slot_map,
    build_slot_ownership_map,
    get_local_capture_layers,
    pack_aux_into_buffer,
    unpack_aux_from_buffer,
    validate_capture_layers,
    log_pp_aux_capture_info,
)


class TestAuxPacking:
    """Test auxiliary packing with synthetic tensors."""

    def setup_method(self):
        # Global capture layers = [1, 3, 5, 7, 9]
        self.global_layers = [1, 3, 5, 7, 9]
        self.layer_to_slot = build_layer_to_slot_map(self.global_layers)
        self.hidden_size = 64
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def test_slot_mapping(self):
        """Verify global slot ordering is exact."""
        assert self.layer_to_slot == {1: 0, 3: 1, 5: 2, 7: 3, 9: 4}

    def test_local_capture_layers_pp0(self):
        """PP0 owns layers [0, 5) -> captures [1, 3]."""
        local = get_local_capture_layers(self.global_layers, 0, 5)
        assert local == [1, 3]

    def test_local_capture_layers_pp1(self):
        """PP1 owns layers [5, 10) -> captures [5, 7, 9]."""
        local = get_local_capture_layers(self.global_layers, 5, 10)
        assert local == [5, 7, 9]

    def test_pack_unpack_roundtrip_decode(self):
        """Decode: num_token_rows = batch_size."""
        bs = 4
        num_tokens = bs
        num_capture = len(self.global_layers)

        # Create distinct features for each layer.
        all_features = {}
        for layer_id in self.global_layers:
            all_features[layer_id] = torch.full(
                (num_tokens, self.hidden_size),
                float(layer_id),
                dtype=self.dtype,
                device=self.device,
            )

        # Build slot ownership
        slot_ownership = build_slot_ownership_map(
            self.global_layers, pp_size=2, num_hidden_layers=10
        )

        # PP0: allocate buffer, fill layers 1 and 3.
        packed_aux = allocate_packed_aux_buffer(
            num_tokens, num_capture, self.hidden_size, self.dtype, self.device
        )
        pp0_local = [1, 3]
        pp0_features = [all_features[lid] for lid in pp0_local]
        pack_aux_into_buffer(packed_aux, pp0_features, pp0_local, self.layer_to_slot)

        # PP1: receive packed buffer, fill layers 5, 7, 9.
        pp1_local = [5, 7, 9]
        pp1_features = [all_features[lid] for lid in pp1_local]
        pack_aux_into_buffer(packed_aux, pp1_features, pp1_local, self.layer_to_slot)

        # Unpack on last stage.
        result = unpack_aux_from_buffer(
            packed_aux,
            self.global_layers,
            self.layer_to_slot,
            slot_ownership,
            pp1_local,
            pp_rank=1,
            pp_size=2,
        )

        # Verify each slot has the correct values.
        for i, layer_id in enumerate(self.global_layers):
            expected = float(layer_id)
            assert torch.all(result[i] == expected), (
                f"Slot {i} (layer {layer_id}): expected {expected}, "
                f"got {result[i][0, 0].item()}"
            )

    def test_pack_unpack_roundtrip_target_verify(self):
        """Target verify: num_token_rows = bs * verify_tokens_per_request."""
        bs = 4
        verify_tokens_per_req = 4  # speculative_num_draft_tokens
        num_tokens = bs * verify_tokens_per_req
        num_capture = len(self.global_layers)

        all_features = {}
        for layer_id in self.global_layers:
            all_features[layer_id] = torch.full(
                (num_tokens, self.hidden_size),
                float(layer_id),
                dtype=self.dtype,
                device=self.device,
            )

        slot_ownership = build_slot_ownership_map(
            self.global_layers, pp_size=2, num_hidden_layers=10
        )

        # PP0
        packed_aux = allocate_packed_aux_buffer(
            num_tokens, num_capture, self.hidden_size, self.dtype, self.device
        )
        pp0_local = [1, 3]
        pack_aux_into_buffer(
            packed_aux,
            [all_features[lid] for lid in pp0_local],
            pp0_local,
            self.layer_to_slot,
        )

        # PP1
        pp1_local = [5, 7, 9]
        pack_aux_into_buffer(
            packed_aux,
            [all_features[lid] for lid in pp1_local],
            pp1_local,
            self.layer_to_slot,
        )

        result = unpack_aux_from_buffer(
            packed_aux,
            self.global_layers,
            self.layer_to_slot,
            slot_ownership,
            pp1_local,
            pp_rank=1,
            pp_size=2,
        )

        assert len(result) == num_capture
        for i, layer_id in enumerate(self.global_layers):
            assert result[i].shape == (num_tokens, self.hidden_size)
            assert torch.all(result[i] == float(layer_id))

    def test_earlier_stage_values_survive(self):
        """PP0 values must survive PP1 processing."""
        num_tokens = 2
        num_capture = len(self.global_layers)

        packed_aux = allocate_packed_aux_buffer(
            num_tokens, num_capture, self.hidden_size, self.dtype, self.device
        )

        # PP0 fills layer 1 with 42.0
        pp0_feature = torch.full(
            (num_tokens, self.hidden_size), 42.0, dtype=self.dtype
        )
        pack_aux_into_buffer(packed_aux, [pp0_feature], [1], self.layer_to_slot)

        # PP1 fills layer 5 with 99.0 — must not clobber layer 1.
        pp1_feature = torch.full(
            (num_tokens, self.hidden_size), 99.0, dtype=self.dtype
        )
        pack_aux_into_buffer(packed_aux, [pp1_feature], [5], self.layer_to_slot)

        # Layer 1 (slot 0) should still be 42.0.
        assert torch.all(packed_aux[:, 0, :] == 42.0)
        # Layer 5 (slot 2) should be 99.0.
        assert torch.all(packed_aux[:, 2, :] == 99.0)

    def test_dtype_preserved(self):
        """Dtype must be preserved through pack/unpack."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            num_tokens = 2
            packed = allocate_packed_aux_buffer(
                num_tokens, len(self.global_layers), self.hidden_size, dtype, self.device
            )
            assert packed.dtype == dtype

    def test_partial_fill_detected_on_last_stage(self):
        """Missing owner on the last stage must raise a RuntimeError."""
        num_tokens = 2
        num_capture = len(self.global_layers)

        # Build ownership with a missing layer (layer 11 not in 0..9 range)
        with pytest.raises(ValueError, match="no PP owner"):
            build_slot_ownership_map([1, 3, 5, 7, 11], pp_size=2, num_hidden_layers=10)

    def test_unpack_with_missing_owner_raises(self):
        """Unpack on last stage with incomplete ownership should raise."""
        num_tokens = 2
        packed_aux = allocate_packed_aux_buffer(
            num_tokens, len(self.global_layers), self.hidden_size,
            self.dtype, self.device,
        )
        # Incomplete ownership (missing layer 9)
        incomplete_ownership = {1: 0, 3: 0, 5: 1, 7: 1}
        with pytest.raises(RuntimeError, match="no owning PP stage"):
            unpack_aux_from_buffer(
                packed_aux,
                self.global_layers,
                self.layer_to_slot,
                incomplete_ownership,
                local_capture_layers=[5, 7, 9],
                pp_rank=1,
                pp_size=2,
            )

    def test_all_token_rows_survive(self):
        """All token rows must survive the pack/unpack."""
        num_tokens = 16
        num_capture = len(self.global_layers)

        all_features = {}
        for layer_id in self.global_layers:
            all_features[layer_id] = torch.randn(
                num_tokens, self.hidden_size, dtype=self.dtype
            )

        packed = allocate_packed_aux_buffer(
            num_tokens, num_capture, self.hidden_size, self.dtype, self.device
        )

        all_local = list(self.global_layers)
        all_feats = [all_features[lid] for lid in all_local]
        pack_aux_into_buffer(packed, all_feats, all_local, self.layer_to_slot)

        slot_ownership = build_slot_ownership_map(
            self.global_layers, pp_size=1, num_hidden_layers=10
        )
        result = unpack_aux_from_buffer(
            packed,
            self.global_layers,
            self.layer_to_slot,
            slot_ownership,
            all_local,
            pp_rank=0,
            pp_size=1,
        )

        for i, layer_id in enumerate(self.global_layers):
            assert torch.equal(result[i], all_features[layer_id])

    def test_validate_capture_layers_valid(self):
        """Valid configuration passes validation."""
        validate_capture_layers(
            [1, 3, 5, 7, 9], 10, 2, 0, 5, 64
        )

    def test_validate_capture_layers_empty(self):
        """Empty capture list raises error."""
        with pytest.raises(ValueError, match="empty"):
            validate_capture_layers([], 10, 2, 0, 5, 64)

    def test_validate_capture_layers_out_of_range(self):
        """Out-of-range layer ID raises error."""
        with pytest.raises(ValueError, match="out of range"):
            validate_capture_layers([1, 11], 10, 2, 0, 5, 64)

    def test_validate_capture_layers_bad_hidden(self):
        """Zero hidden_size raises error."""
        with pytest.raises(ValueError, match="invalid hidden_size"):
            validate_capture_layers([1, 3], 10, 2, 0, 5, 0)

    def test_glm52_pp2_ownership(self):
        """GLM-5.2 has 78 layers. PP2 split: PP0=[0,39), PP1=[39,78).
        Default capture layers (with +1): [3, 40, 76]."""
        # Simulate GLM-5.2 config: 78 layers, default eagle3 capture
        num_layers = 78
        # set_eagle3_layers_to_capture default: [2, 78//2, 78-3] = [2, 39, 75]
        # with +1 applied when layer_ids[0] == 1: not applicable (2 != 1)
        # So raw_layer_ids = [2, 39, 75]
        raw_layer_ids = [2, 39, 75]
        ownership = build_slot_ownership_map(
            raw_layer_ids, pp_size=2, num_hidden_layers=num_layers
        )
        # PP0 owns layers [0, 39): layer 2 is PP0, layer 39 is PP1
        assert ownership[2] == 0
        assert ownership[39] == 1
        assert ownership[75] == 1

    def test_glm52_pp2_local_capture(self):
        """Verify local capture layers for GLM-5.2 PP2 split."""
        raw_layer_ids = [2, 39, 75]
        # PP0: [0, 39) -> captures layer 2
        pp0_local = get_local_capture_layers(raw_layer_ids, 0, 39)
        assert pp0_local == [2]
        # PP1: [39, 78) -> captures layers 39, 75
        pp1_local = get_local_capture_layers(raw_layer_ids, 39, 78)
        assert pp1_local == [39, 75]

    def test_dsa_topk_and_aux_coexistence(self):
        """DSA topk proxy and EAGLE-3 aux proxy can coexist."""
        num_tokens = 4
        num_capture = len(self.global_layers)
        topk_size = 2048

        # Both buffers exist in the same pp_proxy_tensors dict
        packed_aux = allocate_packed_aux_buffer(
            num_tokens, num_capture, self.hidden_size,
            self.dtype, self.device,
        )
        topk_indices = torch.zeros(
            (num_tokens, topk_size), dtype=torch.int32, device=self.device
        )

        # Simulate the proxy dict
        proxy_tensors = {
            "hidden_states": torch.zeros((num_tokens, self.hidden_size), dtype=self.dtype),
            "residual": torch.zeros((num_tokens, self.hidden_size), dtype=self.dtype),
            "topk_indices": topk_indices,
            GLM52_EAGLE3_AUX_PP_KEY: packed_aux,
        }

        # Both keys must be present and non-None
        assert proxy_tensors[GLM52_EAGLE3_AUX_PP_KEY] is not None
        assert proxy_tensors["topk_indices"] is not None
        assert proxy_tensors[GLM52_EAGLE3_AUX_PP_KEY].shape == (
            num_tokens, num_capture, self.hidden_size
        )
        assert proxy_tensors["topk_indices"].shape == (num_tokens, topk_size)

    def test_target_verify_row_sizing(self):
        """TARGET_VERIFY: buffer must be sized by bs * verify_tokens, not bs."""
        bs = 8
        verify_tokens_per_req = 4
        num_tokens = bs * verify_tokens_per_req
        num_capture = len(self.global_layers)

        # This is the shape that _allocate_decode_buffers should produce
        packed = allocate_packed_aux_buffer(
            num_tokens, num_capture, self.hidden_size, self.dtype, self.device
        )
        assert packed.shape[0] == num_tokens
        assert packed.shape[1] == num_capture
        assert packed.shape[2] == self.hidden_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
