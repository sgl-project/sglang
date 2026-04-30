---
doc_type: feature-design
feature: 2026-04-30-ug-interleaved-official-parity
slug: ug-interleaved-official-parity
status: approved
created: 2026-04-30
roadmap: ug-official-alignment
roadmap_item: ug-interleaved-official-parity
related_architecture: [ug-runtime]
---

# UG Interleave Official Parity Design

## 1. Decisions And Constraints

This feature proves the first real Interleave result path against official BAGEL:

```text
input image + text
-> U prefill in one SRT-owned session
-> G denoise/decode generated image
-> append generated image to the same session
-> U greedy text decode after image
```

Mount points:

- Extend `test/registered/scheduler/test_bagel_g_official_parity.py` with an `interleave` task. The old `interleaved` spelling is kept only as a legacy input alias.
- Reuse `python/sglang/srt/ug/parity.py` artifact comparison; no new runtime import of official BAGEL.
- Reuse SRT-owned `UGSessionRuntime`; do not add a separate orchestrator.
- Record generated image, post-image text/token ids, and SRT debug counters in the parity artifact.

Hard constraints:

- Official BAGEL code remains only in the opt-in test subprocess.
- SGLang runtime must append the generated image through SRT session state, not restart a new prompt/session.
- The first implementation stays single-card and single-session.
- This feature does not do HTTP/CLI full entrypoint smoke; that is `ug-entrypoint-official-smoke`.

## 2. Flow

The official runner manually uses BAGEL's public interleave primitives:

1. Initialize `gen_context`, `cfg_text_context`, and `cfg_img_context`.
2. Add the input image and edit/generation text.
3. Run the same denoise loop used by T2I/Edit parity.
4. Decode the generated image and record it.
5. Append the generated image with `update_context_image(..., vae=True, vit=True)`.
6. Run short greedy BAGEL text decode and record token ids/text.

The SGLang runner mirrors that sequence through SRT-owned runtime:

1. `UGSessionRuntime.prefill_interleaved(image, text)` creates one session.
2. `decode_next_segment` moves the session into `G_DENOISE`.
3. `prepare_latents`, `predict_velocity`, and `decode_latents_to_image` reuse existing native SRT G path.
4. `append_generated_image` appends the generated image to the same session.
5. A short official-style greedy decode calls `runtime.decode_text` with BAGEL BOS token and logical rope positions.

## 3. Verification

Minimal closure:

- Opt-in remote true-weight smoke with `SGLANG_TEST_BAGEL_G_TASKS=interleave`.
- `report.json` has `passed=true` and `diffs=0`.
- Candidate counters show `prefill_count=1`, `append_image_count=1`, `velocity_count=num_steps-1`, same `session_id`, nonzero `srt_u_decode_request_count`.
- Candidate artifact contains `generated_image`, `post_image_text`, and `token_ids.post_image_text`.

Stop signals:

- If post-image text decode must start a new SRT session, stop and redesign.
- If generated image can be compared but text after image cannot be produced from the same session, stop and keep Phase 5 open.
- If official BAGEL runtime functions are imported under `python/sglang/**`, revert that direction.
