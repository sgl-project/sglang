---
doc_type: feature-design
feature: 2026-04-30-ug-entrypoint-official-smoke
slug: ug-entrypoint-official-smoke
status: approved
created: 2026-04-30
roadmap: ug-official-alignment
roadmap_item: ug-entrypoint-official-smoke
related_architecture: [ug-runtime]
---

# UG Entrypoint Official Smoke Design

## 1. Decisions And Constraints

This feature moves from internal parity harness proof to user-callable experimental entrypoints. The already-proven VLM, T2I, edit, and interleave result paths stay the source of truth; this step only proves that the public experimental entrypoints route into the same UGPipeline/SRT-owned session path.

Mount points:

- Add an opt-in true-weight smoke under `test/registered/scheduler/`.
- Add `DiffGenerator.generate_interleave`, CLI `--ug-interleave-input`, and HTTP `POST /v1/ug/interleave`; keep the older `interleaved` spelling as compatibility aliases.
- Explicitly cover `t2i`, `edit`, `interleave`, `vlm`, and `think_t2i`. `think_t2i` is `mode=t2i` plus a `think` switch, not a separate generation mode.
- Record serialized segment types, runtime stats, and image presence for every entrypoint run.

Hard constraints:

- No official BAGEL/seed import in runtime. Official parity remains in the offline harness only.
- Do not promise OpenAI-compatible schema; `/v1/ug/interleave` is still experimental.
- Single GPU only; no CFG parallel, disagg, or batching productization.
- Entry smoke is not image-quality parity; it must preserve the result path already validated by official parity.

## 2. Flow

The smoke uses explicit mode payloads. The canonical interleave payload is:

```json
{
  "messages": [
    {"type": "image", "image": "/data/BAGEL/test_images/women.jpg"},
    {"type": "text", "text": "Turn the scene into a warm cinematic portrait."}
  ],
  "sampling_params": {
    "height": null,
    "width": null,
    "num_inference_steps": 4,
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 1.5
  }
}
```

It exercises:

1. Python API: `DiffGenerator.generate_interleave_serializable(payload)`.
2. CLI: `python -m sglang.multimodal_gen ... generate --ug-interleave-input payload.json`.
3. HTTP: start diffusion server and call `POST /v1/ug/interleave`.

For interleave, the serialized output must contain an image segment and a post-image text segment. For T2I/Edit it must contain only an image segment. For VLM it must contain only text and `velocity_count=0`. Stats must prove the SRT-owned path: one prefill, G velocity only for image generation modes, append-image only for interleave, and nonzero U decode when text is produced.

## 3. Verification

Minimal closure:

- CPU/static checks pass for the new smoke script.
- Remote GPU2 true-weight Python API smoke passes.
- Remote GPU2 true-weight CLI smoke passes.
- Remote GPU2 true-weight HTTP smoke passes.
- The feature checklist and roadmap are updated with exact commands/output paths.

Stop signals:

- If any entrypoint returns only a one-shot image without post-image text, keep this feature open.
- If stats are missing or show `append_image_count == 0`, stop and fix entrypoint serialization/routing.
- If the server path cannot create the native SRT scheduler for BAGEL, stop and fix scheduler wiring before hardening.
