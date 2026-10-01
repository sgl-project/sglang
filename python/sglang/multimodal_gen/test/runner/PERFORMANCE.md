# CI performance guards and baselines

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

## B200 runner baselines

`b200.json` keeps the existing Verda references as its defaults. Its
`runner_overrides` map applies metric overrides by the GitHub `RUNNER_NAME`
prefix. DeepInfra runners (`b200-di*`) use separate E2E references for the two
cases below; unknown runners keep the defaults. Loading, stage/step and memory
references, other cases, and the 25% E2E tolerance are unchanged.

| Case | Default E2E (ms) | DeepInfra E2E (ms) |
| --- | ---: | ---: |
| `flux1_modelopt_nvfp4_t2i` | 836.71 | 1334.16 |
| `qwen_image_2512_modelopt_nvfp4_t2i` | 9650.06 | 16126.87 |

The pool mismatch was observed in [B200 CI job 103856482654](https://github.com/sgl-project/sglang/actions/runs/34805428031/job/103856482654).
The DeepInfra references are the medians of three unprofiled, warmed requests
using the CI case configuration at commit
`9b7e11f32b88d4c1bfd9cf44550a30a702dd9df8`:
Flux: 1367.47, 1327.48, 1334.16 ms; Qwen: 16126.87, 16806.36, 15240.80 ms.
The matching Verda measurements were 811.26, 771.97, 783.31 ms and
9169.16, 9045.49, 9071.78 ms, respectively. These calibrate the runner pools;
they do not establish a model-level root cause for the difference.

When refreshing a pool-specific reference, update its `runner_overrides` entry,
not the shared `scenarios` entry. The baseline generation script writes shared
scenarios; use a separate `--out` file when collecting pool-specific candidates.

### Cirrascale historical CI reference

`b200-cirrascale1-0123` has separate E2E references of 1574.32 ms for
`flux1_modelopt_nvfp4_t2i` and 17742.04 ms for
`qwen_image_2512_modelopt_nvfp4_t2i`. The measured Cirrascale 3 runners below also
have separate references. Other Cirrascale runners retain the defaults until
calibrated.

Historical jobs on this runner, all using driver 580.126.20, already recorded
the slower timings before this PR's changes, with unchanged B200 case definitions:

| PR / CI job | Flux1 E2E (ms) | Qwen2512 E2E (ms) |
| --- | ---: | ---: |
| [#39021](https://github.com/sgl-project/sglang/actions/runs/34594936520/job/103248571918) | 1772.69 | 17836.93 |
| [#38782](https://github.com/sgl-project/sglang/actions/runs/34595934993/job/103251774792) | 2617.81 | 17742.04 |
| [#39022](https://github.com/sgl-project/sglang/actions/runs/34670636885/job/103506632867) | 1574.32 | 38392.34 |
| [#39291](https://github.com/sgl-project/sglang/actions/runs/34750152864/job/103707842024) | 3942.31 | 19846.93 |

Use the minimum observed E2E for each case, not the noisy maximum or a median
inflated by slow runs. The existing 25% tolerance yields limits of 1967.90 ms
and 22177.55 ms. Large transient slowdowns must still fail and use the bounded
failed-item retry policy. Historical green jobs did not enforce the new E2E
guard; their recorded timings, not their green status, support these references.
This does not identify the underlying host/GPU contention mechanism. Loading,
other metrics, other cases, and other runner references remain unchanged.

### Cirrascale 3 historical CI references

Match `b200-cirrascale3-0123` and `b200-cirrascale3-4567` separately, without
extending the override to unmeasured runners. Their two NVFP4 case definitions,
sampling configuration and model implementations are unchanged in the historical
comparisons below. These are warmed request timings, not model download or load
times. The independent PRs did not include this PR's E2E guard changes.

| PR / CI job | Runner suffix | Flux1 E2E (ms) | Qwen2512 E2E (ms) |
| --- | --- | ---: | ---: |
| [#40265](https://github.com/sgl-project/sglang/actions/runs/35437052621/job/105881489389) | `3-0123` | 1470.24 | 35377.30 |
| [#39206, earlier head](https://github.com/sgl-project/sglang/actions/runs/35433818034/job/105873136034) | `3-0123` | 1471.59 | 17894.49 |
| [#40293](https://github.com/sgl-project/sglang/actions/runs/35433914839/job/105879952157) | `3-4567` | 1547.77 | 17346.65 |
| [#39983](https://github.com/sgl-project/sglang/actions/runs/35448201646/job/105999591501) | `3-4567` | 1471.19 | 18294.78 |
| [#40374](https://github.com/sgl-project/sglang/actions/runs/35465121078/job/105956031846) | `3-4567` | 1500.43 | 18433.93 |

The earlier #39206 row uses the minimum of its seven attempts, not their noisy
maximum. Apply the same minimum-observed rule per runner across these records:
1470.24 / 17894.49 ms for `3-0123`, and 1471.19 / 17346.65 ms for `3-4567`.
Keep the 25% tolerance and all other metrics unchanged. In particular, the
35-second historical Qwen outlier still fails; it is not a new reference.
These records establish a pre-existing runner-specific mismatch with the Verda
reference, not the underlying cause of contention or a claim that all runs pass.

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
