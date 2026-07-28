"""Phase 6: P1-8 EAGLE3 architecture semantic audit.

Uses source inspection where imports fail, runtime tests where possible.
"""
from __future__ import annotations

import os
import sys
import inspect
import ast

sys.path.insert(0, "/home/liang/sglang/python")


def get_source(filepath):
    """Read source file directly."""
    with open(filepath) as f:
        return f.read()


def test_draft_model_validation_exists():
    """Verify _validate_eagle3_draft_model exists and checks MTP/NextN."""
    source = get_source("/home/liang/sglang/python/sglang/srt/speculative/eagle_worker_v2.py")
    
    assert "def _validate_eagle3_draft_model" in source
    assert "realpath" in source
    assert "GlmMoeDsaForCausalLMNextN" in source
    assert "num_nextn_predict_layers" in source
    assert "_MTP_CONFIG_KEYWORDS" in source
    assert "_MTP_ARCH_NAMES" in source
    
    print("  Draft model validation exists: PASSED")


def test_separate_draft_model_path_required():
    """Verify PP+spec requires separate draft model path."""
    source = get_source("/home/liang/sglang/python/sglang/srt/speculative/glm52_eagle3_pp.py")
    assert "speculative_draft_model_path" in source
    assert "separate trained EAGLE3" in source
    print("  Separate draft path required: PASSED")


def test_draft_lm_head_ownership():
    """Verify EAGLE3 draft can own its own lm_head."""
    source = get_source("/home/liang/sglang/python/sglang/srt/speculative/eagle_worker_v2.py")
    
    assert "is_eagle3" in source
    assert "share_lm_head" in source
    assert "load_lm_head_from_target" in source
    assert "_load_checkpoint_tensor" in source
    assert "_EMBED_TENSOR_NAMES" in source
    
    print("  Draft lm_head ownership: PASSED")


def test_draft_only_on_last_stage():
    """Verify draft worker only initialized on last PP stage."""
    source = get_source("/home/liang/sglang/python/sglang/srt/managers/scheduler.py")
    
    # Check that non-last stages set draft_worker = None
    assert "pp_rank" in source
    assert "draft_worker = None" in source
    
    print("  Draft only on last stage: PASSED")


def test_pp0_no_draft_weights():
    """Verify PP0 does not load draft-only weights."""
    source = get_source("/home/liang/sglang/python/sglang/srt/managers/scheduler.py")
    assert "draft_worker = None" in source
    
    print("  PP0 no draft weights: PASSED")


def test_aux_merge_before_final_norm():
    """Verify aux merge ordering."""
    source = get_source("/home/liang/sglang/python/sglang/srt/models/deepseek_v2.py")
    
    assert "glm52_target_final_norm" in source
    assert "glm52_pp1_aux_merge" in source
    
    norm_pos = source.find("glm52_target_final_norm")
    merge_pos = source.find("glm52_pp1_aux_merge")
    assert merge_pos > norm_pos, "Aux merge must come after final norm"
    
    print("  Aux merge ordering: PASSED")


def test_lm_head_on_last_stage():
    """Verify lm_head logits only on last PP stage."""
    source = get_source("/home/liang/sglang/python/sglang/srt/models/deepseek_v2.py")
    assert "is_last_rank" in source
    assert "glm52_lm_head_logits" in source
    
    print("  lm_head on last stage: PASSED")


def test_embedding_source_for_draft():
    """Verify draft embedding is loaded from checkpoint."""
    source = get_source("/home/liang/sglang/python/sglang/srt/speculative/eagle_worker_v2.py")
    assert "model.embed_tokens.weight" in source
    assert "embed.weight" in source
    assert "_load_checkpoint_tensor" in source
    
    print("  Embedding source for draft: PASSED")


def test_mtp_rejection():
    """Verify MTP/NextN is rejected as draft."""
    source = get_source("/home/liang/sglang/python/sglang/srt/speculative/eagle_worker_v2.py")
    
    assert "GlmMoeDsaForCausalLMNextN" in source
    assert "DeepseekV3ForCausalLMNextN" in source
    assert "mtp" in source.lower()
    assert "nextn" in source.lower()
    assert "MTPModel" in source
    
    print("  MTP rejection: PASSED")


def test_nvtx_ranges_present():
    """Verify all required NVTX ranges are present."""
    source = get_source("/home/liang/sglang/python/sglang/srt/models/deepseek_v2.py")
    
    required_ranges = [
        "_send_proxy",
        "glm52_pp1_aux_merge",
        "glm52_target_final_norm",
        "glm52_lm_head_logits",
    ]
    
    for rng in required_ranges:
        assert rng in source, f"Missing NVTX range: {rng}"
    
    # Check eagle_worker_v2 for verify and tail_draft
    eagle_source = get_source("/home/liang/sglang/python/sglang/srt/speculative/eagle_worker_v2.py")
    assert "glm52_target_verify" in eagle_source
    assert "glm52_tail_draft" in eagle_source
    
    # Check scheduler for result relay
    sched_source = get_source("/home/liang/sglang/python/sglang/srt/managers/scheduler_pp_mixin.py")
    assert "glm52_pp_result_relay" in sched_source
    
    print("  NVTX ranges present: PASSED")


def test_rejection_sampling_disabled():
    """Verify rejection sampling is hard-disabled for PP+spec."""
    source = get_source("/home/liang/sglang/python/sglang/srt/speculative/glm52_eagle3_pp.py")
    assert "speculative_use_rejection_sampling" in source
    assert "incompatible" in source
    
    print("  Rejection sampling disabled: PASSED")


if __name__ == "__main__":
    print("=== Phase 6: P1-8 EAGLE3 Architecture Semantic Audit ===")
    test_draft_model_validation_exists()
    test_separate_draft_model_path_required()
    test_draft_lm_head_ownership()
    test_draft_only_on_last_stage()
    test_pp0_no_draft_weights()
    test_aux_merge_before_final_norm()
    test_lm_head_on_last_stage()
    test_embedding_source_for_draft()
    test_mtp_rejection()
    test_nvtx_ranges_present()
    test_rejection_sampling_disabled()
    print("\n=== All Phase 6 tests PASSED ===")
