# Diffusion Runtime Architecture

This document defines the architectural boundaries for `sglang.multimodal_gen`.
It is intended to keep refactors incremental, maintain compatibility, and avoid coupling across layers.

## Layer model

SGLang diffusion is organized into four layers:

1. API/SDK adapters
- Role:
  - Expose protocol-facing interfaces (OpenAI-compatible HTTP, CLI, Python entrypoints).
  - Validate and map external requests/responses.
- Must not:
  - Own generation orchestration policy or scheduler transport details.

2. Runtime service and orchestration
- Role:
  - Own generation orchestration, lifecycle, and request dispatch.
  - Provide reusable library-level service interfaces for infra integration.
- Must not:
  - Leak protocol-specific behavior into core orchestration paths.

3. Scheduler transport and execution control
- Role:
  - Abstract sync/async scheduler communication.
  - Coordinate worker execution and request handling.
- Must not:
  - Encode API schema concerns.

4. Pipeline execution and model components
- Role:
  - Execute stage graph (validation, encoding, latent prep, denoising, decoding).
  - Host model-specific pipeline implementations.
- Must not:
  - Depend on protocol adapters.

## Mapping to code

- API/SDK adapters:
  - `python/sglang/multimodal_gen/runtime/entrypoints/openai/*`
  - `python/sglang/multimodal_gen/runtime/entrypoints/http_server.py`
  - `python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py`
- Runtime service/orchestration:
  - `python/sglang/multimodal_gen/runtime/launch_server.py`
  - `python/sglang/multimodal_gen/runtime/managers/scheduler.py`
- Scheduler transport:
  - `python/sglang/multimodal_gen/runtime/scheduler_client.py`
- Pipeline execution/model components:
  - `python/sglang/multimodal_gen/runtime/pipelines_core/*`
  - `python/sglang/multimodal_gen/runtime/pipelines/*`
  - `python/sglang/multimodal_gen/registry.py`

## Refactor contract for abstraction work

When improving abstraction (for example issue #19115), follow these rules:

1. Keep adapters thin
- Protocol handlers should translate input/output only.
- Move reusable generation logic to runtime service layer.

2. Preserve behavior by default
- Any executor policy or transport refactor must keep existing defaults unchanged.
- Behavior changes require explicit tests and documentation.

3. Migrate incrementally
- Use small, single-purpose commits.
- Keep compatibility wrappers while transitioning call sites.

4. Separate transport from business logic
- Callers should depend on a stable scheduler facade, not socket-level details.

5. Keep pipeline contracts explicit
- Stage parallelism and executor policy must be documented and testable.

## Non-goals for abstraction refactors

- Kernel-level performance rewrites in DiT/VAE backends.
- Algorithmic redesign of denoising math or scheduler algorithms.
- Protocol feature expansion unrelated to boundary cleanup.

## PR acceptance checklist

- Layer boundaries are respected.
- Existing external API behavior remains compatible.
- Tests cover migrated boundary behavior.
- Performance-sensitive changes include before/after evidence when applicable.
- Transitional compatibility path is present (or removal is justified after full migration).
