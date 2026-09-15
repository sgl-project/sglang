# CI performance diagnostics

The NVIDIA H100 one- and two-GPU diffusion jobs set
`SGLANG_DIFFUSION_DIAGNOSTICS_DIR`.
Other jobs and local runs remain opt-in. This records evidence only: it does
not change baselines, tolerances, warmup inputs, retry policy, or exit codes.
The metric and failure contracts below apply independently of this sampler.

Each invocation and retry gets a unique `attempt-N-*` directory:

| Artifact | Contents |
| --- | --- |
| `events.jsonl` | Actual checkout SHA, selected dependency versions, CI run/attempt/partition, observed case/module/worker/stage log boundaries, and subprocess exit status |
| `requests.jsonl` | Every formal request's existing E2E/stage/step/memory metrics, flushed before assertions, including cases with `run_perf_check=False` |
| `processes.jsonl` | Approximately 1 Hz process-tree CPU, I/O, memory, fault and scheduler counters, plus host I/O and memory pressure |
| `resources.jsonl` (GPU opt-in only) | Process counters and attributed GPU utilization, clocks, power/limit, temperature and throttle reasons, including NVML query durations |

GPU sampling requires `SGLANG_DIFFUSION_DIAGNOSTICS_GPU=1` in addition to the
diagnostics directory. It is off by default: NVML process-ownership queries can
contend with inference driver calls and perturb the latency being measured.
Use it for a separate diagnostic replay, not the clean performance comparison.
Process/pressure sampling and request metrics remain enabled without it; the
attempt metadata records whether GPU sampling was enabled.

JSONL is flushed per event so completed evidence survives interrupted pytest
sessions. A missing `attempt_end` means incomplete, not success. The workflow
uploads these files with `always()` separately from the existing final-result
report, so retries do not erase failed samples. Abrupt runner loss can still
prevent artifact upload.

## Interpretation

- `observed_boundary` timestamps are **stdout receipt times**, not synchronized
  CUDA timings. Use case-begin to workers-ready to locate startup, not the first
  tqdm `0/N` refresh, which may occur well into warmup. Do not treat this as a
  precise loading assertion. Module loading is a subset of startup.
- Ordinary request E2E is the existing worker forward metric, not total pytest
  wall time. Realtime E2E measures the complete requested WebSocket generation
  session, from sending initialization to receiving its frames and chunk stats;
  it is not the last chunk's latency. Startup, warmup and later MP4 encoding are
  excluded. Keep these two E2E scopes distinct when comparing results.
- Cumulative CPU/I/O/fault counters must be differenced by `(pid, created)`.
  Processes shorter than a sample interval may be missed. Zero observed disk
  reads do not prove a warm cache or absence of I/O.
- GPU ownership is matched to live descendants, not CUDA ordinal assumptions.
  Before CUDA context creation there may be no attributed GPU. NVML/process PID
  namespace mismatches can also leave GPU samples empty; this is missing data,
  not zero utilization. Unsupported counters are explicit errors, not zero.
- Software power capping is not automatically faulty hardware. Compare its
  bitmask, power limit and clocks across equivalent runs. Resource samples alone
  cannot prove a kernel, synchronization or storage root cause; a focused trace
  may still be necessary. The sampler adds overhead; measure it before making
  performance claims from instrumented runs.

No arbitrary environment, command lines, prompts, file contents or full logs
are saved. Only allowlisted log markers and numeric process/resource data are
collected. Sampling runs in the pytest parent, never adds CUDA synchronization,
does not attach a debugger, and does not change device settings.

## Metric and failure contracts

Every generated testcase must report finite, positive E2E, including cases with
`run_perf_check=False`. Missing request records, absent performance logs and
missing/invalid E2E fail CI. `run_perf_check=False` disables stage/step and
memory checks, not the request's E2E threshold guard. Baseline generation still
requires valid E2E but skips baseline comparisons. Explicit GT generation skips
validation.

A performance failure stops the testcase's remaining repeated requests and
subsequent checks for that attempt. Performance failures use the existing
pytest retry budget (at most six retries), rerunning only failed items.
Exhausting the budget still fails CI; missing metrics and exceeded thresholds
are never treated as passing. Consistency failures remain non-retryable.
Standalone infrastructure failures retain their existing retry policy.
Valid failed measurements are recorded
before threshold validation; realtime chunk and memory guards remain enabled
according to their existing configuration.

## Initial loading references

The initial H100 loading references cover 28 cases with at least three distinct
CI runs whose maximum/minimum startup-time ratio is at most 1.25. Each reference
is the minimum measured `load_time_ms`, rounded to two decimals; repeated requests
sharing one server do not count as separate startups. These are process-start to
all-workers-ready measurements, excluding warmup, not checkpoint-I/O-only times.

Sources are PR Test Base runs [34750203901](https://github.com/sgl-project/sglang/actions/runs/34750203901),
[34751664379](https://github.com/sgl-project/sglang/actions/runs/34751664379),
[34753876622](https://github.com/sgl-project/sglang/actions/runs/34753876622), and
[34755864886](https://github.com/sgl-project/sglang/actions/runs/34755864886).
Only saved valid loading measurements are used; this does not claim those runs
passed all other checks. No E2E reference or tolerance is raised. Cross-run
stability is a conservative selection criterion, not proof of an optimal load
time. The first batches left variable cases uncalibrated; the best-observed
references below now give those cases a loading guard without claiming stable
runtime. B200 and 5090 references are not inferred from H100 or development H200
measurements.

The six RTX 5090 loading references use the same selection rule, from runs
[34753876622](https://github.com/sgl-project/sglang/actions/runs/34753876622),
[34755864886](https://github.com/sgl-project/sglang/actions/runs/34755864886), and
[34764411082](https://github.com/sgl-project/sglang/actions/runs/34764411082).
Their per-case maximum/minimum ratios range from 1.048 to 1.203. The last run's
six recorded requests passed E2E validation and failed the then-missing loading
baseline check; subsequent checks and MiniMax's second request did not run.
The existing MiniMax wall-clock tolerance override is unchanged.

The same three runs also provide initial two-H100 loading references for
`ltx_2.3_one_stage_ti2v`, `ltx_2.3_two_stage_t2v_2gpus`,
`wan2_1_t2v_1.3b_cfg_parallel`, and `zimage_image_t2i_2_gpus`.
Seven partition-0 cases also meet this criterion: `flux2_modelopt_fp8_tp2_t2i`,
`flux_image_t2i_2_gpus`, `ideogram4_fp8_tp2_t2i`, `qwen_image_t2i_2_gpus`,
`wan2_2_i2v_a14b_2gpu`, `wan2_2_t2v_a14b_lora_2gpu`, and
`wan2_2_t2v_a14b_teacache_2gpu`. All eleven cases meet the same three-run
stability criterion. Existing loading references are not raised when a later
run exceeds their limits.

Five more two-H100 references use runs
[34755864886](https://github.com/sgl-project/sglang/actions/runs/34755864886),
[34764411082](https://github.com/sgl-project/sglang/actions/runs/34764411082), and
[34766506722](https://github.com/sgl-project/sglang/actions/runs/34766506722):
`flux_2_image_t2i_2_gpus`, `fsdp-inference`, `mova_360p_tp2`,
`wan2_1_i2v_14b_720P_2gpu`, and `zimage_image_t2i_2_gpus_non_square`.
Their maximum/minimum loading ratios range from 1.093 to 1.205. Each reference
is the minimum observed loading time rounded to two decimal places; the same
existing tolerances apply. These are initial references, not claims that the
full testcases or all later checks passed.

Seven further H100 references use that same recent three-run window:
`flux_image_t2i`, `flux_2_ti2i`, `joyai_image_edit_ti2i`,
`qwen_image_edit_2509_ti2i`, `qwen_image_layered_i2i`,
`minimax_h3_t2va_2gpu_h100`, and `qwen_image_edit_modelopt_fp8_ti2i`.
Their maximum/minimum ratios in this window range from 1.006 to 1.250
(the largest unrounded ratio is 1.249565). Earlier historical runs vary more;
these references do not claim stability across the entire history. Each value
is the minimum in the stated window, rounded to two decimals. Existing loading
references, E2E references, and tolerances are unchanged.

Four additional single-H100 references follow the same minimum-of-three rule.
`flux_2_image_t2i_upscaling_4x` uses runs 34764411082, 34766506722, and
[34770008768](https://github.com/sgl-project/sglang/actions/runs/34770008768).
`flux_2_t2i_customized_vae_path`, `flux_2_ti2i_multi_image_cache_dit`, and
`zimage_image_t2i` use runs 34766506722, 34770008768, and
[34771349145](https://github.com/sgl-project/sglang/actions/runs/34771349145).
Their maximum/minimum ratios range from 1.090 to 1.224. These are initial
loading references only; earlier variable runs remain diagnostic evidence,
and no existing performance reference or tolerance is increased.

`qwen_image_edit_ti2i` and `lingbot_world_realtime_plastic_beach` use runs
34764411082, 34766506722, and 34771349145. Their minimum loading times are
36793.38 ms and 31600.49 ms, respectively, with maximum/minimum ratios of
1.1745 and 1.1256. The realtime case uses the same process-start-to-ready
loading boundary, excluding warmup; its request E2E reference is unchanged.

### Best-observed references for variable cases

The remaining 16 H100 and five B200 cases use the fastest valid startup in
the saved reports from runs 34755864886, 34764411082, 34766506722,
34770008768, and 34771349145, where available. Each value is rounded to two
decimals. H100 cases have three to five distinct startups; B200 cases have
two. These are initial measured references, not claims of stability or proof
that every historical run passed. Requiring all noisy samples to converge
before establishing a guard would leave these cases without a quantified limit.

No existing reference or tolerance is increased. Samples above the resulting
limit still fail, and their infrastructure/code diagnosis remains separate.
In particular, LTX HQ retains the observed 116.531 s startup as its reference,
not the later 187-208 s startups. Downloads in H200 development measurements
are not used to establish either GPU pool's loading references.

| Source run | GPU | Cases supplying the minimum |
| --- | --- | --- |
| 34755864886 | H100 | `fast_hunyuan_video`, `joy_echo_t2v_2gpu`, `wan2_1_i2v_14b_480P_2gpu`, `wan2_1_t2v_14b_2gpu`, `wan2_2_t2v_a14b_2gpu` |
| 34764411082 | H100 | `lingbot_video_moe_t2v`, `ltx_2_3_hq_pipeline`, `qwen_image_t2i_2_gpus_extra_high`, `wan22_modelopt_fp8_t2v` |
| 34766506722 | H100 | `ltx_2_3_two_stage_ti2v_2gpus`, `ltx_2_5_diffusion_decoder_2gpus`, `ltx_2_two_stage_t2v`, `minimax_h3_ref2va_video_audio_2gpu_h100`, `sana_wm_ti2v`, `wan2_1_i2v_14b_lora_2gpu`, `wan2_1_t2v_1_3b_cache_dit_sp_only_2gpu` |
| 34755864886 | B200 | `flux1_modelopt_nvfp4_t2i`, `flux2_modelopt_nvfp4_t2i` |
| 34764411082 | B200 | `ideogram4_nvfp4_t2i`, `qwen_image_2512_modelopt_nvfp4_t2i`, `wan22_modelopt_nvfp4_t2v` |
