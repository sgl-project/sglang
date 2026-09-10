#!/usr/bin/env python3
"""Patch radix_attention.py: disable _pcg_mha_companion swap for absorbed MLA.

Root cause of accuracy gap: In BCG (breakable CUDA graph) path, unified_attention_with_output
swaps attn_mqa (v_head_dim=512) to attn_mha (v_head_dim=256) when save_kv_cache=False.
But in the absorbed MLA path, save_kv_cache=False is set by fused_qk_rope_cat_and_cache_mla
(NOT by MHA path), so the swap is incorrect — it makes the backend use wrong head/dim metadata.

Fix: Remove the swap entirely. The absorbed MLA path should always use attn_mqa.
This eliminates the need for patch_dsa_rope_dim.py and patch_tilelang_dv.py.
"""
import sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/radix_attention.py"

with open(FILE, "r") as f:
    content = f.read()

old = """    # DeepSeek MLA has two RadixAttention instances per layer (attn_mqa and
    # attn_mha) that share the same layer_id. The attention_layers list only
    # stores attn_mqa. When the MHA path is active (save_kv_cache=False), use
    # the companion attn_mha so the backend sees correct head/dim metadata.
    if _is_hip and not save_kv_cache and hasattr(attention_layer, "_pcg_mha_companion"):
        attention_layer = attention_layer._pcg_mha_companion"""

new = """    # DeepSeek MLA has two RadixAttention instances per layer (attn_mqa and
    # attn_mha) that share the same layer_id. The attention_layers list only
    # stores attn_mqa. When the MHA path is active (save_kv_cache=False), use
    # the companion attn_mha so the backend sees correct head/dim metadata.
    # NOTE: Disabled for absorbed MLA path — save_kv_cache=False in fused rope
    # MLA path does NOT mean MHA is active. Using attn_mha gives wrong v_head_dim
    # (256 instead of 512=kv_lora_rank), causing accuracy degradation.
    # if _is_hip and not save_kv_cache and hasattr(attention_layer, "_pcg_mha_companion"):
    #     attention_layer = attention_layer._pcg_mha_companion
    pass"""

if old in content:
    content = content.replace(old, new, 1)
    with open(FILE, "w") as f:
        f.write(content)
    print("[PATCH] Disabled _pcg_mha_companion swap in radix_attention.py")
else:
    if "_pcg_mha_companion" in content and "# if _is_hip" in content:
        print("[PATCH] Already patched")
        sys.exit(0)
    print("[PATCH] ERROR: Pattern not found")
    sys.exit(1)
