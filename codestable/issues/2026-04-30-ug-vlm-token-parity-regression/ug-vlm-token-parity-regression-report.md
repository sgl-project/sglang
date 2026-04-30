---
doc_type: issue-report
issue: 2026-04-30-ug-vlm-token-parity-regression
created: 2026-04-30
severity: high
status: fixed
tags: [ug, bagel, vlm, parity]
---

# UG VLM Token Parity Regression Report

## Problem

Phase 3 `ug-vlm-entrypoint-official-smoke` exposed a regression in the Phase 2 VLM result-parity gate. The entrypoint path correctly stays on the U-only path, but the generated text no longer matches official BAGEL.

The current acceptance rule is deliberately narrower than logits parity: the short greedy generated token ids/text must match official BAGEL for the fixed image+text VLM case.

## Reproduction

Remote environment:

- host/container: `radixark-h200-2-omni` / `sgl_flamingo`
- checkpoint: `/data/models/BAGEL-7B-MoT`
- official BAGEL repo: `/data/BAGEL`
- test: `test/registered/scheduler/test_bagel_vlm_official_parity.py`
- case: default `women.jpg`, prompt `Describe this image briefly.`, `max_new_tokens=8`

Observed outputs:

- official: `Audrey Hepburn in a red`
- Phase 3 `forward_vlm`: `Audrey Hepburn is in the`
- direct parity with warm VIT: `Audrey Hepburn wearing a red`

## Expected

SGLang UG should reproduce the official short greedy token/text sequence before Phase 4 G/CFG/T2I/Edit work continues.

## Scope Guard

This issue should not expand G denoise, CFG, server batching, or full interleaved API behavior. It is only about restoring VLM U-path result parity.

## Root Cause

Official BAGEL VLM text generation calls `generate_text(..., do_sample=False)`, so the reference path is greedy. SGLang UG `decode_text` was creating default `SamplingParams(max_new_tokens=1)`, which leaves `temperature=1.0` and `top_k=all`; the SRT sampler therefore sampled instead of taking argmax. The earlier outputs looked close because the distribution was plausible, but the path was not deterministic official parity.

## Fix

Added an explicit greedy option to UG `decode_text` and routed BAGEL VLM official-style decode through it. The SRT request now uses `temperature=0.0`, which normalizes to `top_k=1` in SGLang sampling.

Also fixed the native BAGEL text/sidecar request order so the main text prefill runs before the CFG image sidecar, matching official interleave order.

## Verification

- Local static checks: `py_compile`, `ruff --select F401,F821`, `black --check` on changed UG runtime/BAGEL/parity/unit files.
- Remote unit: `TestUGSessionRuntime.test_decode_text_can_force_greedy_sampling` OK.
- Remote unit: `TestBAGELSRTKVCacheAdapter.test_native_srt_image_edit_prefill_builds_cfg_img_sidecar` OK.
- Remote true-weight direct parity: `test_bagel_vlm_official_parity.py -v` OK with `/tmp/ug-vlm-align-greedy`.
- Remote true-weight `UGPipeline.forward_vlm` smoke: token ids/text exact match official (`Audrey Hepburn in a red`), `prefill_count=1`, `velocity_count=0`, `append_image_count=0`.
