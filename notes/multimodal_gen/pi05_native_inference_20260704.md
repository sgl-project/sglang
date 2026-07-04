# Pi0.5 native inference scaffold - 2026-07-04

- git hash: c7896e8b7d
- added native `Pi05Pipeline` scaffold under `multimodal_gen`, with config, sampling params, registry, preprocess, exact full-prefix cache manager, OpenPI-style policy API, and action output handling
- SRT serving engine/scheduler is intentionally not started; current reuse points are config/registry conventions, radix-style cache concepts, and denoise CUDA graph runner boundary
- real SigLIP/PaliGemma prefix forward, Gemma action expert forward, true prefix/action process groups, and SRT piecewise CUDA graph replay remain the next implementation step
- no local tests, pycompile, serve, or GPU validation were run per local development rule

## 2026-07-04 continuation

- git hash: c7896e8b7d
- remote GPU: `pi05-e2e`, 2x H100; validation ran on `CUDA_VISIBLE_DEVICES=0`
- implemented native Pi05 Torch path with HF Transformers 5.8-compatible PiGemma/PaliGemma modules, streaming safetensors loader, and checkpoint config ingestion
- fixed checkpoint mapping for `vision_tower.vision_model.* -> vision_tower.*`; Pi05 base now loads with no missing/unexpected warnings except the intentionally skipped action expert `lm_head`
- `lerobot/pi05_base` GPU direct E2E: prefix len 968, output shape `[1, 50, 32]`, peak allocated memory 12.817 GiB
- LeRobot reference parity using same fixed state dict: one-step velocity max abs diff `1.17e-6`, final 10-step action max abs diff `1.17e-7`
- denoise CUDA graph now captures one action step by shape bucket and replays with static prefix/action/timestep buffers; eager 10-step `125.4 ms`, first graph request `94.6 ms`, steady replay `50.8 ms`, max diff `0`
- exact global prefixcache validated: exact full-prefix hit reuses the same `PrefixContext`; changed image digest misses; pipeline second request skipped prefix stage (`~0.2 ms`)
- `lerobot/pi05_libero_base` GPU direct E2E passed with image keys `image`, `image2`, `empty_camera_0`, state dim 8, output action dim 7, output shape `[1, 50, 32]`
- Pi05 pipeline stage path passed with `SyncExecutor`: timings include preprocess, cache lookup, prefix, action denoise, postprocess; second request cache hit action denoise was `~51 ms`
- fixed `Pi05SamplingParams` output filename handling so numpy image/state/noise payloads are not JSON-hashed
- disabled diffusion component offload/layerwise auto defaults for `VLA_ACTION`, avoiding misleading Pi05 server args
- not yet validated in this pass: full `sglang serve` HTTP/OpenPI websocket path, multi-GPU prefix/action process-group kernels, and multi-node disaggregation

## 2026-07-04 cookbook placement

- git hash: `c7896e8b7d`
- Pi0.5 cookbook is placed under a new `cookbook/vla` category instead of `cookbook/diffusion`; the implementation uses flow-matching action denoising, but the user-facing workload is robot action policy serving rather than image/video diffusion.

## 2026-07-04 serve and split validation

- git hash: `c7896e8b7d` plus local Pi05/CLI patch set on remote checkout `49e384ce9d304648e9959666ecb8ce8cd98d0deb`
- fixed top-level CLI compatibility for `sglang serve lerobot/pi05_base --model-type diffusion --pipeline pi05`; the wrapper now normalizes a first positional model id into `--model-path` and marks it explicit for multimodal `ServerArgs`
- remote GPU: `pi05-e2e`, 2x H100 80GB; clean validation started from `4 MiB` used on each GPU
- single-GPU server command passed: `sglang serve lerobot/pi05_base --model-type diffusion --pipeline pi05 --host 127.0.0.1 --port 30000 --warmup-mode off --num-gpus 1`
- single-GPU HTTP `/pi05/generate` returned action shape `[50, 32]`; timings: preprocess `27.8 ms`, prefix `238.9 ms`, action denoise `207.3 ms`, postprocess `0.09 ms`
- single-GPU OpenPI websocket `/openpi/policy` returned action shape `[50, 32]`; exact prefixcache hit on the second request; action denoise `50.9 ms`, server timing `61.5 ms`
- single-GPU steady server snapshot after request: GPU0 `21719 MiB` total used, with the scheduler worker using `21186 MiB`
- two-GPU split command passed: `sglang serve lerobot/pi05_base --model-type diffusion --pipeline pi05 --host 127.0.0.1 --port 30001 --warmup-mode off --num-gpus 2 --sp-degree 2 --ulysses-degree 2`
- two-GPU split uses the SP process group as a prefix/action split group: rank0 keeps prefix components, rank1 keeps action components, prefix `PrefixContext` is broadcast once, and actions/timings are broadcast back after denoise
- two-GPU HTTP `/pi05/generate` returned action shape `[50, 32]`; timings: preprocess `29.2 ms`, prefix `202.5 ms`, action denoise `201.4 ms`, postprocess `0.10 ms`
- two-GPU OpenPI websocket `/openpi/policy` returned action shape `[50, 32]`; exact prefixcache hit on the second request; action denoise `51.2 ms`, server timing `75.0 ms`
- single-GPU HTTP action and two-GPU split HTTP action matched exactly for the deterministic zero-noise request: `max_abs_diff = 0.0`
- two-GPU split steady server snapshot after request: GPU0 `19255 MiB` total used (`18722 MiB` prefix worker), GPU1 `9535 MiB` total used (`9002 MiB` action worker)
- validation cleanup now kills leftover GPU compute-app PIDs on the dedicated devbox; confirmed both GPUs returned to `4 MiB`

### Remaining limitations

- this is true multi-process prefix/action process-group execution, but it is not yet action-expert tensor/sequence-parallel sharded compute; action denoise still runs on the action root rank
- role pruning happens after checkpoint load, so peak load memory can still be higher than steady memory; a rank-aware streaming loader is still required for low-VRAM robot GPUs
- current steady memory is suitable for 24GB-class cards in single-GPU mode and split prefix side, but not for 8GB/16GB edge deployment without quantization/offload/rank-aware loading; the split action side is around 9GB on H100 in this test
- rank1 action-role logs are not emitted in the shared server log, though the prefix rank waits for action-root tensor broadcast and the end-to-end result proves the action path executed

## 2026-07-04 lossless memory optimization

- git hash: `c7896e8b7d`; remote checkout hash: `49e384ce9d304648e9959666ecb8ce8cd98d0deb`
- local/remote file hashes after sync:
  - `configs/pipeline_configs/pi05.py`: `3a26dbf471254f6b80fef9786c410df81f215998bf84ad74fd576259f7a4d1ed`
  - `runtime/models/pi05/modeling_pi05.py`: `cce67f533bf61bb0b50c9b0dba2c901fb15e11d41e4c583ec4feb595923c05bf`
  - `runtime/models/pi05/torch_model.py`: `784af1ad2be599bd6d0c6bccca2c916284810ebcad20ded3441d56ca9d119952`
  - `runtime/pipelines_core/stages/model_specific_stages/pi05.py`: `f5eb1bf143cb728ea73beabee2ee8ee07a7395a986d7c715f6370c26bf4d9736`
- changed Pi05 model construction to be role-aware before allocation/load: prefix ranks instantiate only PaliGemma/SigLIP path, action ranks instantiate only action expert/projections, single-GPU keeps `runtime_role=all`
- changed checkpoint load order to load role-selected tensors on CPU and move the constructed role module to GPU after load; skipped tensors are not read with `safe_open.get_tensor`
- fixed role-aware checkpoint component detection: actual LeRobot keys are lowercase and include `paligemma.model.*`; the earlier selective loader skipped prefix weights and produced `602 missing`, now logs `Pi05 weights loaded successfully`
- fixed request-level `enable_pi_cuda_graph`: `Pi05SamplingParams` already exposed it, but the stage/model path now actually bypasses graph replay when it is false
- changed default `prefix_cache_max_entries` from `128` to `1` to prevent unbounded exact-prefix GPU KV growth on changing robot camera frames while preserving the immediate repeated-prefix hit path
- final single-GPU serve validation on `pi05-e2e` H100:
  - command: `PI05_RUN_TAG=single_final bash /tmp/pi05_remote_run_server.sh single 30000`
  - HTTP and OpenPI websocket shapes: `[50, 32]`
  - HTTP timings: preprocess `29.9 ms`, prefix `208.3 ms`, action denoise `192.1 ms`
  - websocket exact prefixcache hit: action denoise `50.9 ms`, server timing `61.5 ms`
  - steady snapshot: GPU0 `19741 MiB` total, worker `19208 MiB`
- final split serve validation on `pi05-e2e` H100:
  - command: `PI05_RUN_TAG=split_final bash /tmp/pi05_remote_run_server.sh split 30001`
  - HTTP and OpenPI websocket shapes: `[50, 32]`
  - HTTP timings: preprocess `28.4 ms`, prefix `206.4 ms`, action denoise `213.1 ms`
  - websocket exact prefixcache hit: action denoise `51.2 ms`, server timing `96.4 ms`
  - steady snapshot: GPU0 `19255 MiB` total (`18722 MiB` prefix worker), GPU1 `9527 MiB` total (`8994 MiB` action worker)
  - single-final vs split-final deterministic HTTP actions: `max_abs_diff = 0.0`
- prefixcache-off validation:
  - command: `PI05_RUN_TAG=split_nocache bash /tmp/pi05_remote_run_server.sh split 30001 --disable-prefix-cache`
  - output shape `[50, 32]`, final-vs-nocache HTTP action `max_abs_diff = 0.0`
  - steady snapshot was effectively unchanged: GPU0 `19291 MiB`, GPU1 `9527 MiB`; conclusion: one PrefixContext is not the main steady-memory cost
- cuda-graph-off validation:
  - command: `PI05_RUN_TAG=split_nograph bash /tmp/pi05_remote_run_server.sh split 30001 --disable-cuda-graph`
  - output shape `[50, 32]`, final-vs-nograph HTTP action `max_abs_diff = 0.0`
  - steady snapshot: GPU0 `19255 MiB`, GPU1 `9397 MiB`, about `130 MiB` lower on action rank
  - websocket action denoise increased from `51.2 ms` to `98.2 ms`; this is a low-VRAM fallback, not the default performance path
- conclusion: lossless changes mainly reduce load-time duplication/IO and long-run cache growth risk; steady memory is still dominated by PaliGemma/SigLIP on prefix rank and action expert/allocator/graph workspaces on action rank. Further reduction below 16GB-class cards likely needs offload, quantization, or real model sharding.

## 2026-07-04 cookbook VRAM tuning update

- git hash: `c7896e8b7d`
- updated `docs_new/cookbook/vla/OpenPI/Pi0.5.mdx` with a dedicated VRAM tuning section
- documented the 16GB target interpretation: it is a reasonable v1 robot workstation bar only if steady serve memory stays around `14-15 GiB` with headroom; current H100 serve snapshots are still above that on single GPU and prefix rank
- added practical knobs:
  - server config JSON using `enable_global_prefix_cache=false`, `prefix_cache_max_entries=0`, and `enable_action_cuda_graph=false`
  - per-request `enable_pi_prefix_cache=false` and `enable_pi_cuda_graph=false`
  - split prefix/action guidance and caveats
- updated validation table to the latest numbers: single total `19.7 GiB`, split prefix/action totals `19.3 GiB` / `9.5 GiB`, CUDA graph off saves about `130 MiB` on action rank but increases cache-hit denoise from about `51 ms` to `98 ms`

## 2026-07-04 private fork sync and 16GB lossless/offload validation

- private sync repo: `mickqian/sglang-diffusion-distill-dev`
- private branch: `codex/pi05-vram-sync`
- latest pushed commit: `935061d46a`
- reason for private fork: avoid copying a large dirty local worktree to the GPU devbox and make remote validation reproducible from one commit
- fixed `PipelineConfig.from_kwargs` so `--pipeline-config-path` is honored; before the fix, only `pipeline_config` / `--config` worked and `pipeline_config_path` could be silently ignored
- added Pi05 memory config logging in `Pi05Pipeline.load_modules`, so server logs now show prefix cache, CUDA graph, and prefix offload choices at startup
- added prefix-rank componentwise empty initialization for low-VRAM split mode:
  - prefix rank first materializes the Pi05 prefix model as empty CPU tensors
  - only the prefix rotary/final norm and non-offloaded submodules are placed on GPU
  - offloaded vision tower/projector, token embedding, and PaliGemma language layers stay on CPU from the start
  - layerwise CPU offload moves one prefix language layer to GPU during prefix compute, then returns it to CPU and empties the CUDA cache
- earlier low-VRAM attempt with only delayed `torch.cuda.empty_cache()` reduced the initial prefix worker from `14928 MiB` to `6622 MiB`, but later loading still rose back to `14928 MiB` and OOMed under the artificial cap
- componentwise empty init fixed that peak: final low-VRAM run logged `Pi05 componentwise empty init enabled for prefix rank`, `Pi05 weights loaded successfully`, and `Pi05 split runtime role on rank: prefix`
- remote validation: `pi05-e2e`, 2x H100; GPU0 artificially constrained by ballast to about `15.5 GiB` free before server start; GPU1 had about `14.1 GiB` free due unrelated orphan allocation
- low-VRAM command:
  - `PYTHONPATH=/sgl-workspace/sglang-pi05-vram/python PI05_REPO_DIR=/sgl-workspace/sglang-pi05-vram PI05_SKIP_GPU_CLEANUP=1 PI05_RUN_TAG=split_16gb_componentwise_msgpack PI05_SERVER_EXTRA_ARGS="--pipeline-config-path /tmp/pi05_low_vram_16gb.json --image-encoder-cpu-offload --text-encoder-cpu-offload" bash /tmp/pi05_remote_run_server.sh split 30012 --disable-prefix-cache --disable-cuda-graph`
- `/tmp/pi05_low_vram_16gb.json` used:
  - `enable_global_prefix_cache=false`
  - `prefix_cache_max_entries=0`
  - `enable_action_cuda_graph=false`
  - `offload_prefix_image_encoder=true`
  - `offload_prefix_token_embedding=true`
  - `offload_prefix_language_layers=true`
  - `offload_prefix_language_layers_empty_cache=true`
  - `empty_cache_after_prefix=true`
- HTTP `/pi05/generate` returned shape `[50, 32]`; timings: preprocess `29.2 ms`, prefix `5641.5 ms`, action denoise `280.9 ms`, postprocess `0.13 ms`
- OpenPI websocket `/openpi/policy` returned shape `[50, 32]`; timings: preprocess `9.1 ms`, prefix `5119.8 ms`, action denoise `95.1 ms`, postprocess `0.11 ms`, server timing `5241.1 ms`
- low-VRAM snapshot after request:
  - prefix GPU total `72842 MiB` because ballast occupied `65214 MiB`; actual server processes were parent `518 MiB` plus prefix worker `7090 MiB`
  - action GPU total `76369 MiB` because of unrelated existing `66976 MiB` usage; actual server processes were parent `518 MiB` plus action worker `8864 MiB`
- conclusion: Pi0.5 now has a numerically lossless 16GB-class compatibility path, but it is not realtime because prefix CPU layerwise offload raises prefix latency to about `5-6 s`. Normal low-latency robot deployment still wants 24GB-class GPUs or further work on quantization / real sharding.
- cleanup: server workers were killed after validation; ballast was also killed. GPU0 returned to `4 MiB`; GPU1 still showed unrelated no-process `66976 MiB` allocation from the devbox environment.

## 2026-07-04 Run:ai Model Streamer note

- Run:ai Model Streamer is relevant for Pi0.5 loader optimization because the checkpoints are safetensors and the loader is currently safetensors-based.
- It can reduce cold-start/load latency by concurrent storage reads and streaming tensor data toward GPU memory. It is most useful for local SSD/object-storage/cloud startup time.
- It does not reduce steady VRAM after the model is resident. The 16GB blocker here was GPU parameter placement / allocator residency / CUDA graph and cache buffers, not SSD read bandwidth.
- Implementation direction: keep the existing role/component-aware loader boundary, then optionally add a Run:ai-backed safetensors reader for target GPU tensors. CPU-offloaded tensors should still load to CPU, otherwise streamer can increase pressure on the exact VRAM budget we are trying to protect.

## 2026-07-04 Run:ai Model Streamer implementation and guarded validation

- private branch: `mickqian/sglang-diffusion-distill-dev:codex/pi05-vram-sync`
- validation checkout before docs-only updates: `98c871171`
- implemented `multimodal_gen.runtime.loader.safetensors_weights_iterator` support for:
  - `key_filter`, so Pi05 can keep skip-before-read behavior on the plain safetensors path
  - direct GPU Run:ai streaming when `to_cpu=False`
  - `clone_streamed_tensors=True` by default, preserving safe generic loader semantics for callers that may retain a yielded tensor beyond the streamer context
- Pi05 uses `clone_streamed_tensors=False` only in the immediate `target.copy_(tensor)` path.
- Pi05 direct GPU streaming is guarded by `_should_stream_weights_to_gpu`:
  - device must be CUDA
  - runtime role must not be `action` or `idle`
  - every load target for the current process must already be GPU-resident
  - distributed world size must be one
- reason for the world-size guard: an attempted two-rank prefix/action split with rank0 using direct GPU streamer stalled after logging `Loading safetensors with Run:ai Model Streamer to cuda:0`; rank1 had loaded action tensors and rank0 held about `17.9 GiB`, but the streamer did not complete. Split/offload needs a real distributed streamer protocol before direct GPU streaming is safe by default.
- final single-GPU direct-stream validation on current code:
  - command: `PI05_RUN_TAG=single_runai_guarded bash /tmp/pi05_remote_run_server.sh single 30012 --disable-prefix-cache`
  - log: `Pi05 weight load streams safetensors directly to cuda`
  - Run:ai load: `13.5 GiB` streamed to `cuda:0` in `1.86 s`, `7.2 GiB/s`
  - HTTP `/pi05/generate`: shape `[50, 32]`, timings preprocess `30.5 ms`, prefix `223.4 ms`, action denoise `207.4 ms`
  - OpenPI websocket `/openpi/policy`: shape `[50, 32]`, timings preprocess `2.8 ms`, prefix `58.1 ms`, action denoise `131.3 ms`
  - GPU snapshot: total GPU0 `20087 MiB`, worker `19554 MiB`
- final guarded split validation:
  - command: `PI05_RUN_TAG=split_runai_guarded bash /tmp/pi05_remote_run_server.sh split 30012 --disable-prefix-cache`
  - no Run:ai direct streaming log appeared; weights loaded through the guarded non-direct loader
  - HTTP `/pi05/generate`: shape `[50, 32]`, timings preprocess `30.9 ms`, prefix `227.5 ms`, action denoise `203.0 ms`
  - OpenPI websocket `/openpi/policy`: shape `[50, 32]`, timings preprocess `27.0 ms`, prefix `54.9 ms`, action denoise `129.4 ms`
  - GPU snapshot: prefix worker `18758 MiB`, action worker `8994 MiB`
- final guarded 16GB-class low-VRAM validation:
  - GPU0 was constrained by ballast to about `15.5 GiB` free before server start
  - command: `PI05_RUN_TAG=split_16gb_runai_guarded PI05_SERVER_EXTRA_ARGS="--pipeline-config-path /tmp/pi05_low_vram_16gb.json --image-encoder-cpu-offload --text-encoder-cpu-offload" bash /tmp/pi05_remote_run_server.sh split 30012 --disable-prefix-cache --disable-cuda-graph`
  - log: `Pi05 componentwise empty init enabled for prefix rank`; no Run:ai direct streaming log
  - HTTP `/pi05/generate`: shape `[50, 32]`, timings preprocess `30.3 ms`, prefix `5050.8 ms`, action denoise `212.8 ms`
  - OpenPI websocket `/openpi/policy`: shape `[50, 32]`, timings preprocess `12.1 ms`, prefix `5864.2 ms`, action denoise `90.7 ms`
  - GPU snapshot after request: ballast `65214 MiB`, prefix worker `7090 MiB`, action worker `8864 MiB`
- cookbook update: `docs_new/cookbook/vla/OpenPI/Pi0.5.mdx` now documents Run:ai enable/disable behavior, single-process direct SSD-to-GPU use, split/offload guardrails, and the fact that Model Streamer is a cold-start optimization rather than a steady-VRAM reduction.
- cleanup: all visible server and ballast GPU processes were killed; GPU0 returned to `4 MiB`. GPU1 still showed unrelated no-process `66976 MiB` allocation from the devbox environment.

## 2026-07-04 robot edge / Jetson-oriented lossless phase offload

- private branch: `mickqian/sglang-diffusion-distill-dev:codex/pi05-vram-sync`
- latest pushed code commit before docs/notes update: `c75164c0a8`
- motivation: the full prefix layerwise CPU-offload path fits a 16GB-class budget, but prefix latency is about `5-6 s`; that is only a compatibility fallback and is too slow for the intended robot edge path.
- implemented stage-granular offload knobs:
  - `offload_prefix_image_encoder_after_embed`: keep SigLIP/projector on CPU between requests, move them to GPU only for image embedding, then release them before prefix language forward
  - `offload_prefix_language_layer_count_after_prefix`: keep only the first N PaliGemma language layers on CPU between prefix passes, moving them to GPU for the prefix forward; the remaining layers stay GPU-resident
  - `offload_action_expert_after_denoise`: keep the action expert/action heads on CPU during prefix compute, move them to GPU for the 10-step denoise loop, then release them after the request
- recommended 16GB-class config after validation:
  - `enable_global_prefix_cache=false`
  - `prefix_cache_max_entries=0`
  - `enable_action_cuda_graph=false`
  - `offload_prefix_image_encoder_after_embed=true`
  - `offload_prefix_token_embedding=true`
  - `offload_prefix_language_layer_count_after_prefix=2`
  - `offload_action_expert_after_denoise=true`
  - `empty_cache_after_prefix=true`
- remote validation: `pi05-e2e`, H100 with GPU0 artificially constrained by ballast to about `15.5 GiB` free before server start.
- failed/less-good configs:
  - token embedding CPU + action after-denoise offload OOMed in prefix language forward; worker was around `14.95 GiB` and the failing allocation was about `20 MiB`
  - token embedding CPU + vision after-embed offload + action after-denoise offload OOMed in the SigLIP image encoder; worker was around `14.94 GiB` and the failing allocation was about `32 MiB`
  - phase offload with all PaliGemma language layers fit, but remained slow: HTTP prefix `5623 ms`, action `1012 ms`; websocket prefix `6244 ms`, action `766 ms`
  - language layer count `1` fit but was slower than count `2`: HTTP prefix `1198 ms`, action `704 ms`; websocket prefix `1175 ms`, action `534 ms`
  - action expert resident with language layer count `2` OOMed while moving the vision encoder to GPU, so strict 16GB currently needs action after-denoise offload
- best 16GB-class result, single-GPU phase offload with language layer count `2`:
  - HTTP `/pi05/generate`: action shape `[50, 32]`, preprocess `31.86 ms`, prefix `995.23 ms`, action denoise `502.56 ms`, postprocess `0.10 ms`
  - OpenPI websocket `/openpi/policy`: action shape `[50, 32]`, preprocess `2.58 ms`, prefix `1202.37 ms`, action denoise `389.56 ms`, postprocess `0.12 ms`, server inference `1612.8 ms`
  - memory snapshot after request: ballast `65214 MiB`, server parent `518 MiB`, worker `12754 MiB`; effective worker footprint about `12.8 GiB` under the artificial 16GB cap
- conclusion: the current Jetson/robot-edge lossless v1 path should be phase offload, not full layerwise offload. It is much faster than the layerwise fallback and fits a 16GB-class cap in validation, but it is still slower than the normal 24GB+ resident path. Further no-quality-loss work should focus on pinned/asynchronous CPU-to-GPU staging, action/prefix sharding, and measuring on real Jetson unified memory rather than relying only on H100 ballast simulation.
- cleanup after this validation: killed the visible GPU0 ballast/server compute apps; GPU0 returned to `4 MiB`. GPU1 still showed the unrelated no-process `66976 MiB` allocation from the devbox environment.

## 2026-07-04 fork PR and local cookbook preview

- PR branch: `mickqian/sglang:codex/pi05-vram-sync`
- draft PR: `https://github.com/mickqian/sglang/pull/12`
- base requested by user: `mickqian/sglang:main`
- caveat: the fork `main` was behind current upstream, so the PR diff against `mick/main` includes upstream sync changes; the clean Pi0.5 diff is still `origin/main...codex/pi05-vram-sync`
- pre-commit after formatter amend: `pre-commit run --from-ref origin/main --to-ref HEAD` passed
- local cookbook preview:
  - Mintlify CLI `4.2.558` requires Node <25; Node `v26.3.0` failed, Node `v24.12.0` worked
  - detached preview uses `screen` session `sglang-pi05-mint`
  - verified `http://127.0.0.1:3000/cookbook/vla/OpenPI/Pi0.5` returned HTTP `200` and rendered the Pi0.5/OpenPI/16GB VRAM tuning page

## 2026-07-04 generic action API rename

- commit before API rename: `1d4b94e78668`
- replaced Pi0.5-specific HTTP endpoints `/pi05/generate` and `/pi05/metadata` with generic action endpoints:
  - `POST /v1/actions/generations`
  - `GET /v1/actions/metadata`
  - `WS /v1/actions/realtime`
- kept the OpenPI-compatible robot adapter at `WS /openpi/policy`
- removed the root websocket alias so the public websocket surface is explicit
- generic HTTP runtime knobs are now `runtime.prefix_cache`, `runtime.cuda_graph`, and `runtime.return_timing`
- Python `Pi05SamplingParams` now uses `enable_prefix_cache` and `enable_cuda_graph`; raw OpenPI observations still accept legacy `enable_pi_prefix_cache` / `enable_pi_cuda_graph` only at the adapter boundary
- Pi0.5 postprocess now includes the actual `parameters.num_inference_steps`, so generic action `usage.denoise_steps` reflects per-request overrides rather than only the checkpoint default
- cookbook and `multimodal_gen` README were updated to document the generic action API; local Mint preview should be rechecked at `http://127.0.0.1:3000/cookbook/vla/OpenPI/Pi0.5`
- cookbook now marks Pi0.5 as a diffusion Vision-Language-Action policy with a `dVLA` sidebar tag and LingBot World-style top tags (`dVLA`, `OpenPI / LeRobot`, `flow matching action`, `robot edge`)
