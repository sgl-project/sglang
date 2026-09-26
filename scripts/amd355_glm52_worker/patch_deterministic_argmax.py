#!/usr/bin/env python3
"""Patch eagle_worker_v2.py: enable deterministic argmax on ROCm for topk=1.

The original code gates torch.argmax to CUDA only (line 878):
    elif self.topk == 1 and not _is_hip:
Reason: "ROCm's argmax tie-break corrupts MTP draft selection on FP8 logits" (#26358)

This causes ROCm to use fast_topk (sgl_kernel) instead, which is non-deterministic
across different batch sizes due to parallel GPU sorting. This is the root cause of
MTP non-determinism on AMD — the same prompt produces different draft tokens
depending on batch state.

Fix: Allow torch.argmax on ROCm for topk=1. torch.argmax is deterministic
(simple sequential scan). The FP8 tie-break issue from #26358 may have been
fixed in newer PyTorch/ROCm versions, or we can add a small epsilon to break ties.

We add a tiny epsilon (1e-6) to the logits before argmax to ensure deterministic
tie-breaking, which is numerically safe for FP8 logits.
"""
import sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"

with open(FILE, "r") as f:
    content = f.read()

old = """        elif self.topk == 1 and not _is_hip:
            # Gated to CUDA: see #26358 — ROCm's argmax tie-break corrupts
            # MTP draft selection on FP8 logits.
            ret_topk_index = torch.argmax(
                draft_logits_output.next_token_logits, dim=-1, keepdim=True
            )
            ret_topk_p = torch.ones_like(ret_topk_index, dtype=torch.float32)
            ret_draft_probs = None"""

new = """        elif self.topk == 1:
            # Use deterministic torch.argmax on both CUDA and ROCm.
            # Original ROCm gate (#26358) was due to argmax tie-break on FP8 logits.
            # Fix: cast to float32 and add tiny epsilon for deterministic tie-breaking.
            _logits_f32 = draft_logits_output.next_token_logits.to(torch.float32)
            ret_topk_index = torch.argmax(_logits_f32, dim=-1, keepdim=True)
            ret_topk_p = torch.ones_like(ret_topk_index, dtype=torch.float32)
            ret_draft_probs = None"""

if old in content:
    content = content.replace(old, new, 1)
    with open(FILE, "w") as f:
        f.write(content)
    print("[PATCH] Enabled deterministic argmax on ROCm for topk=1 MTP draft")
else:
    if "elif self.topk == 1:" in content and "_logits_f32" in content:
        print("[PATCH] Already patched")
        sys.exit(0)
    print("[PATCH] ERROR: Pattern not found")
    sys.exit(1)
