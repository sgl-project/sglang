---
doc_type: feature-design
feature: 2026-04-30-ug-native-model-hardening
slug: ug-native-model-hardening
status: approved
created: 2026-04-30
roadmap: ug-official-alignment
roadmap_item: ug-native-model-hardening
related_architecture: [ug-runtime]
---

# UG Native Model Hardening Design

## 1. Decisions And Constraints

This feature starts Phase 6 after VLM, G, interleaved parity, and full entrypoint smoke have passed. The goal is not to change model behavior, but to move remaining shell logic into SRT-owned native state and keep the already-proven parity from regressing.

Mount points:

- `python/sglang/srt/ug/runtime.py`: owns SRT session decode metadata and tokenizer decode results.
- `python/sglang/srt/ug/adapter.py`: exposes safe decoded SRT metadata to UG model adapters without KV details.
- `python/sglang/srt/ug/bagel.py`: consumes SRT-provided native state instead of reconstructing U results in BAGEL adapter code.
- Existing opt-in parity and entrypoint smoke tests remain the regression proof.

Hard constraints:

- Do not import official BAGEL/seed code into runtime.
- Do not expose KV allocator/page/slot to diffusion-side code.
- Do not change sampling/CFG/image parity semantics.
- Do not start batching productization here; that is the next roadmap item.

This feature has no new user-facing requirement. It hardens an internal native runtime boundary after user-visible parity has already been established.

## 2. Flow

First closure:

```text
SRT U decode request
-> req.output_ids
-> official-style greedy SRT decode for post-image U continuation
-> UGSessionRuntime decodes text with the SRT tokenizer
-> UGModelRunnerAdapter passes decoded text in session metadata
-> BAGELInterleaveContextBackend returns post-image text from SRT metadata
```

This removes the current adapter behavior that turns post-image token ids into a space-separated string, and prevents the post-image U decode from silently using sampling. The token ids stay available for debugging and parity artifacts, but user-facing interleaved text must come from the SRT tokenizer decode.

Later closures in this feature can harden more native boundaries, but only after the first closure passes:

- Persist BAGEL segment/rope state directly in SRT-owned session records where adapter reconstruction still exists.
- Narrow native denoise executor metadata so it depends on explicit SRT bindings instead of inferred adapter state.
- Add guard tests that runtime files still do not import official BAGEL/seed modules.

## 3. Verification

Minimal closure:

- CPU unit test proves `decode_next_segment` uses greedy SRT U decode and exposes tokenizer-decoded text in adapter session metadata.
- CPU BAGEL adapter test proves post-image U text is not a token-id string when a tokenizer is present.
- Existing UG diffusion pipeline CPU tests still pass.
- Remote true-weight entrypoint smoke can be rerun to confirm post-image text becomes decoded text while stats stay `prefill_count=1`, `append_image_count=1`, and `srt_u_decode_request_count>0`.

Stop signals:

- If post-image text can only be produced by re-decoding inside BAGEL adapter from official Python state, stop and redesign.
- If SGLD/diffusion code needs KV page/slot details to recover text, stop and redesign.
- If parity or entrypoint smoke regresses, do not continue to wider hardening.
