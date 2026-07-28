"""Phase 7: P1-3 Tiny model-forward equivalence.

Builds a tiny deterministic model using real SGLang model classes and
production forward methods where possible, comparing non-PP vs PP2 results.

Since we cannot import full DeepseekV2Model (due to transformers version
conflicts on Python 3.13), we build a functional equivalent that exercises
the same PP proxy tensor flow, aux capture, and packing/unpacking logic.
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/home/liang/sglang/python")

from sglang.srt.speculative.glm52_eagle3_pp import (
    build_layer_to_slot_map,
    build_slot_ownership_map,
    get_local_capture_layers,
    allocate_packed_aux_buffer,
    pack_aux_into_buffer,
    unpack_aux_from_buffer,
    validate_capture_layers,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors


class TinyTargetLayer(nn.Module):
    """A single target layer with deterministic weights."""
    def __init__(self, layer_id: int, hidden_size: int, seed: int = 42):
        super().__init__()
        self.layer_id = layer_id
        torch.manual_seed(seed + layer_id)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
        h = self.linear(x)
        h = self.norm(h)
        return h, x  # (new hidden, new residual)


class TinyTargetModel(nn.Module):
    """Tiny target model with configurable layers."""
    def __init__(self, num_layers: int, hidden_size: int, vocab_size: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TinyTargetLayer(i, hidden_size) for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.layers_to_capture = []
    
    def forward(self, input_ids, capture_layers=None, pp_proxy_tensors=None,
                start_layer=0, end_layer=None, is_last_stage=True):
        """Forward with PP proxy support."""
        if end_layer is None:
            end_layer = len(self.layers)
        
        # Get hidden states from input or proxy
        if pp_proxy_tensors is not None and "hidden_states" in pp_proxy_tensors.tensors:
            hidden = pp_proxy_tensors.tensors["hidden_states"]
            residual = pp_proxy_tensors.tensors.get("residual")
            # Merge received aux
            received_aux = pp_proxy_tensors.tensors.get("glm52_eagle3_aux_hidden_states")
        else:
            hidden = self.embed(input_ids)
            residual = None
            received_aux = None
        
        aux_hidden_states = []
        capture_set = set(capture_layers or [])
        
        for i in range(start_layer, end_layer):
            if i in capture_set:
                aux_hidden_states.append(hidden.clone())
            hidden, residual = self.layers[i](hidden, residual)
        
        if not is_last_stage:
            # Return proxy tensors for next stage
            proxy = {"hidden_states": hidden, "residual": residual}
            if capture_layers and aux_hidden_states:
                num_capture = len(capture_layers)
                packed_aux = allocate_packed_aux_buffer(
                    hidden.shape[0], num_capture, self.hidden_size, hidden.dtype, hidden.device
                )
                local_capture = get_local_capture_layers(
                    capture_layers, start_layer, end_layer
                )
                layer_to_slot = build_layer_to_slot_map(capture_layers)
                pack_aux_into_buffer(packed_aux, aux_hidden_states, local_capture, layer_to_slot)
                if received_aux is not None:
                    pack_aux_into_buffer(packed_aux, [], [], layer_to_slot)  # Already has remote
                    # Merge: copy received into packed
                    packed_aux.copy_(received_aux[:packed_aux.shape[0]])
                    pack_aux_into_buffer(packed_aux, aux_hidden_states, local_capture, layer_to_slot)
                proxy["glm52_eagle3_aux_hidden_states"] = packed_aux
            return PPProxyTensors(proxy)
        else:
            # Last stage: apply final norm, merge aux
            hidden = self.norm(hidden)
            
            if capture_layers and received_aux is not None:
                layer_to_slot = build_layer_to_slot_map(capture_layers)
                local_capture = get_local_capture_layers(
                    capture_layers, start_layer, end_layer
                )
                # Merge local aux into received
                if aux_hidden_states:
                    pack_aux_into_buffer(received_aux, aux_hidden_states, local_capture, layer_to_slot)
                
                slot_ownership = build_slot_ownership_map(
                    capture_layers, 2, self.num_hidden_layers
                )
                aux_hidden_states = unpack_aux_from_buffer(
                    received_aux, capture_layers, layer_to_slot, slot_ownership,
                    local_capture, pp_rank=1, pp_size=2,
                )
            elif capture_layers and aux_hidden_states:
                pass  # All captures on last stage
            else:
                aux_hidden_states = []
            
            logits = self.lm_head(hidden)
            return hidden, aux_hidden_states, logits


class TinyDraftModel(nn.Module):
    """Tiny separate EAGLE3 draft model."""
    def __init__(self, hidden_size: int, vocab_size: int = 128, num_draft_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_draft_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.hidden_size = hidden_size
    
    def forward(self, hidden_states, aux_hidden_states=None):
        h = hidden_states
        if aux_hidden_states:
            # Concat aux hidden states
            h = torch.cat([h] + aux_hidden_states, dim=-1)
            # Project back
            if not hasattr(self, 'aux_proj'):
                self.aux_proj = nn.Linear(h.shape[-1], self.hidden_size, bias=False)
            h = self.aux_proj(h)
        for layer in self.layers:
            h = torch.relu(layer(h))
        return self.lm_head(h)


def run_non_pp(target_model, input_ids, capture_layers):
    """Reference: single-process non-PP eager."""
    hidden, aux, logits = target_model(
        input_ids, capture_layers=capture_layers,
        start_layer=0, end_layer=target_model.num_hidden_layers,
        is_last_stage=True
    )
    return hidden, aux, logits


def run_pp2(target_model, input_ids, capture_layers, partition):
    """PP2 eager: run in two stages."""
    split = partition[0]
    
    # Stage 0
    pp_proxy = target_model(
        input_ids, capture_layers=capture_layers,
        start_layer=0, end_layer=split,
        is_last_stage=False
    )
    
    # Stage 1
    hidden, aux, logits = target_model(
        input_ids, capture_layers=capture_layers,
        pp_proxy_tensors=pp_proxy,
        start_layer=split, end_layer=target_model.num_hidden_layers,
        is_last_stage=True
    )
    return hidden, aux, logits


def test_equivalence_2_plus_2():
    """Test 2+2 partition equivalence."""
    num_layers = 4
    hidden_size = 32
    vocab_size = 128
    capture_layers = [1, 2]
    
    torch.manual_seed(42)
    target = TinyTargetModel(num_layers, hidden_size, vocab_size)
    target.layers_to_capture = capture_layers
    
    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    
    # Non-PP reference
    ref_hidden, ref_aux, ref_logits = run_non_pp(target, input_ids, capture_layers)
    
    # PP2 with 2+2
    pp_hidden, pp_aux, pp_logits = run_pp2(target, input_ids, capture_layers, [2, 2])
    
    # Compare
    assert torch.allclose(ref_hidden, pp_hidden, atol=1e-5), (
        f"Hidden mismatch: max diff={(ref_hidden - pp_hidden).abs().max().item()}"
    )
    assert torch.allclose(ref_logits, pp_logits, atol=1e-5), (
        f"Logits mismatch: max diff={(ref_logits - pp_logits).abs().max().item()}"
    )
    assert len(ref_aux) == len(pp_aux), f"Aux count mismatch: {len(ref_aux)} vs {len(pp_aux)}"
    for i, (r, p) in enumerate(zip(ref_aux, pp_aux)):
        assert torch.allclose(r, p, atol=1e-5), (
            f"Aux {i} mismatch: max diff={(r - p).abs().max().item()}"
        )
    
    print("  2+2 partition equivalence PASSED")


def test_equivalence_multiple_partitions():
    """Test 1+3, 2+2, 3+1 partitions."""
    num_layers = 4
    hidden_size = 32
    vocab_size = 128
    capture_layers = [1, 2]
    
    for partition in [[1, 3], [2, 2], [3, 1]]:
        torch.manual_seed(42)
        target = TinyTargetModel(num_layers, hidden_size, vocab_size)
        target.layers_to_capture = capture_layers
        
        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        
        ref_hidden, ref_aux, ref_logits = run_non_pp(target, input_ids, capture_layers)
        pp_hidden, pp_aux, pp_logits = run_pp2(target, input_ids, capture_layers, partition)
        
        assert torch.allclose(ref_hidden, pp_hidden, atol=1e-5), (
            f"Partition {partition}: hidden mismatch"
        )
        assert torch.allclose(ref_logits, pp_logits, atol=1e-5), (
            f"Partition {partition}: logits mismatch"
        )
        assert len(ref_aux) == len(pp_aux)
        for i, (r, p) in enumerate(zip(ref_aux, pp_aux)):
            assert torch.allclose(r, p, atol=1e-5), (
                f"Partition {partition}: aux {i} mismatch: {(r-p).abs().max().item()}"
            )
        
        print(f"  Partition {partition}: PASSED")


def test_two_consecutive_rounds():
    """Test two consecutive speculative rounds (state reuse)."""
    num_layers = 4
    hidden_size = 32
    vocab_size = 128
    capture_layers = [1, 2]
    
    torch.manual_seed(42)
    target = TinyTargetModel(num_layers, hidden_size, vocab_size)
    
    # Round 0: prefill
    input_ids_0 = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    hidden_0, aux_0, logits_0 = run_non_pp(target, input_ids_0, capture_layers)
    
    # Round 1: use output as input (simulate decode)
    token_0 = logits_0[0].argmax()
    input_ids_1 = torch.tensor([token_0], dtype=torch.long)
    hidden_1, aux_1, logits_1 = run_non_pp(target, input_ids_1, capture_layers)
    
    # PP2 version
    torch.manual_seed(42)
    target_pp = TinyTargetModel(num_layers, hidden_size, vocab_size)
    
    pp_hidden_0, pp_aux_0, pp_logits_0 = run_pp2(target_pp, input_ids_0, capture_layers, [2, 2])
    token_pp_0 = pp_logits_0[0].argmax()
    input_ids_pp_1 = torch.tensor([token_pp_0], dtype=torch.long)
    pp_hidden_1, pp_aux_1, pp_logits_1 = run_pp2(target_pp, input_ids_pp_1, capture_layers, [2, 2])
    
    assert torch.equal(token_0, token_pp_0), "Token mismatch between non-PP and PP2"
    assert torch.allclose(logits_1, pp_logits_1, atol=1e-5), "Round 1 logits mismatch"
    
    print("  Two consecutive rounds PASSED")


def test_draft_model_separate():
    """Verify draft model is separate from target."""
    hidden_size = 32
    vocab_size = 128
    torch.manual_seed(42)
    target = TinyTargetModel(4, hidden_size, vocab_size)
    torch.manual_seed(99)
    draft = TinyDraftModel(hidden_size, vocab_size)
    
    # Verify they have different weights
    target_embed = target.embed.weight.data
    draft_embed = draft.embed.weight.data
    assert not torch.allclose(target_embed, draft_embed), "Draft and target share embedding!"
    
    # Verify draft produces different logits
    input_ids = torch.tensor([1, 2], dtype=torch.long)
    hidden = target.embed(input_ids)
    target_logits = target.lm_head(hidden)
    draft_logits = draft(hidden)
    assert not torch.allclose(target_logits, draft_logits), "Draft logits same as target!"
    
    # Verify draft has its own lm_head
    assert draft.lm_head is not target.lm_head, "Draft shares lm_head with target!"
    
    print("  Separate draft model PASSED")


if __name__ == "__main__":
    print("=== Phase 7: P1-3 Tiny Model-Forward Equivalence ===")
    test_equivalence_2_plus_2()
    test_equivalence_multiple_partitions()
    test_two_consecutive_rounds()
    test_draft_model_separate()
    print("\n=== All Phase 7 tests PASSED ===")
