# Ascend dynamic EPLB: community-port self-test

## Source and validation scope

Community base: `de7055f59645ebe03784b0980e1f222789f468a3`.
The [initial verification manifest](ascend_dynamic_eplb_upstream_20261024/verification.json) binds the original 17 Python files and separately records the CI fixture follow-up. The [optimization manifest](ascend_dynamic_eplb_upstream_20261024/optimization.json) identifies the later optimized source, including exact snapshot and canonical LF SHA-256 values.
Eight production files are changed; the existing community TopK padding and shared-expert masking implementation is retained.

The current community port passed the CPU and bounded NPU graph checks below. Full-model, disaggregated P/D and AIS measurements come from the earlier integration described separately below; they have **not** been rerun on this community base.

## Optimization follow-up

Two unnecessary-work paths were removed after reviewing upstream revision `a7ae5d1f345630e9df51415b4e175dbf103bcea6`:

- Globally unchanged NPU layers are skipped before allocating or converting migration workspaces. The full CPU placement map is compared on every rank, because a rank with unchanged local destinations can still be a remote sender. Elastic recovery and other devices retain their existing path.
- Overflow-only forwards skip the unused batched statistics reduction. The route buffer is cleared only after a forward that could write it. Each NPU graph saves its own batching state; normal replay and replay with input updates restore that state before graph execution. This avoids relying on Python routing hooks during replay.

| Follow-up scenario | Result |
|---|---|
| Ten CPU files, including the NPU backend orchestration fixture | 46 tests PASS; NPU initialization forbidden |
| Generated repeated-layer, invalid, empty and overflowing routes | 60 sequences matched independent bincount oracles within the unit suite |
| Actual NPU backend capture and replay, 43 layers / 272 physical experts / six routes per token | Ten replay cases matched physical and logical counts exactly |
| Actual NPU eager execution after mixed graph sizes | Four cases matched physical counts exactly |
| Large graph and eager overflow-only forwards | 43 original per-layer scatter calls retained; extra batch scatter eliminated |
| Two-rank NPU/HCCL, real W8A8 postprocessing and NZ weights | 18 layer updates PASS; globally unchanged layouts, remote senders, replicated experts and shared slots covered |

The graph check captures descending token sizes 128/16/8/1 using the actual NPU backend and actual decode `run_once` callback with model/attention stubs. Replay alternates large fallback-only and small batched graphs, including zero valid rows. Route values and storage pointers remain unchanged. The input-update replay entry point is additionally checked by the CPU backend fixture.

For an unchanged layer with six tensors, the former workspace path performed 12 whole-tensor copy calls; the new path skips the allocation and copies. This is an operation-count observation. No end-to-end throughput or latency gain is claimed for this optimization.

Two initial migration-test exits are retained in the evidence: first an overwritten CANN Python path prevented `acl` import; then the default NPU HCCL port was already occupied by an existing service. Preserving the CANN path and using OS-assigned test ports allowed the same production source to pass. The test used a 32 MiB communication buffer. These are test-environment/communication-setup failures, not successful migration runs.

![Optimization validation scenarios and results](ascend_dynamic_eplb_upstream_20261024/optimization_verification.png)

This is a browser-rendered screenshot of the verified result summary. Full-model requests, PD integration and AIS performance have not been rerun on the optimized community source. The earlier results below retain their original scope.

## Initial community-port tests

Runtime: Ascend 910C, CANN 9.1.0, PyTorch 2.10.0, torch_npu 2.10.0.post6.

| Scenario | Passed | Result |
|---|---:|---|
| Draft/target recording isolation | 4 | PASS |
| Utilization threshold with reporting disabled | 5 | PASS |
| Exact batched statistics, overflow and capture offsets | 6 | PASS |
| Device group creation, reuse and WORLD replacement | 5 | PASS |
| NPU sender counts, DeepEP modes and idle ranks | 8 | PASS |
| Initial learning period and steady-window invariant | 3 | PASS |
| Community TopK padding, shared experts and device row count | 6 | PASS |
| Processed W8A8 weight/scale/offset registration | 2 | PASS |
| Actual NPU graph replay, physical and logical counts | 8 cases | Exact integer equality |
| Repeated-layer and capacity overflow on NPU | 2 cases | Exact integer equality |

All **39 CPU tests passed without NPU initialization**. During port adaptation, an old test expected nonzero weights in padded rows. The test now checks the community's existing zero-padding behavior; production TopK code was not changed.

The NPU probe ran 43 layers, 272 physical experts, 6 routes/token, and descending BS 16/8/1 capture. It executed the actual `run_once` callback extracted from this port, with attention/model stubs. There was no normal recorder reset between warmups and capture. Capture fallback counts were 0/0/0. Eight replay cases varied valid rows from zero to the full batch and matched independent CPU bincount oracles exactly. Two eager overflow cases retained all counts. Shared storage pointers remained stable and route tensors were unchanged. This is a counter/capture test, not a full-model graph or throughput test.

CI compatibility follow-up: the existing PD role-switch test bypasses `ModelRunner.__init__`; its fixture now sets `is_draft_worker=False`, matching a normal target runner. The full **38-case PD role-switch file** and the existing **4-case draft-isolation suite** passed in an isolated source copy without NPU initialization. The 4 draft cases repeat coverage in the 39-test group above; they are not additional distinct cases. Production code was unchanged by this fixture repair.

Changed-file pre-commit checks passed. Three unchanged system hooks expecting a `python3` executable on Windows were also executed successfully using the installed Python.

Run the selected CPU suites in a compatible SGLang development environment:

```bash
for name in draft_distribution_isolation eplb_threshold npu_batched_statistics npu_eplb_group npu_expert_distribution npu_initial_rebalance npu_topk_padding npu_w8a8_registration; do
  python "test/registered/unit/eplb/test_${name}.py" || exit $?
done
```

![Community-port test scenarios and results](ascend_dynamic_eplb_upstream_20261024/verification.png)

This image is a browser-rendered report screenshot generated from the verified JSON results, rather than a terminal screenshot.

## Earlier integration: full-model and PD switch acceptance

The [pinned integration report](https://github.com/chenxu214/ci_sglang/blob/f040177346d5eca8ea38ce8d569444f4a5176206/test/manual/ep/reports/ascend_dynamic_eplb_delivery_20261024.md) records source identities, raw archive hashes, scenarios and screenshots. Those tests used the earlier integration source, not the community base above.

- DeepSeek-V4-Flash W8A8, Ascend 910C, CANN 9.1.0.
- Final packaged single-node run: 16 successful requests, 0 failed, exact copy oracle 16/16, with a real rebalance on all 16 ranks.
- Fresh OFF/ON PD arms: each used two planned 16-request preparations and 64 formal requests at C16, actual input 689-708 and output 128 tokens. All 192 planned requests succeeded. Formal outputs matched exactly between OFF and ON; the copy oracle passed 64/64 in each arm.
- OFF had zero updates on every P/D rank. ON had P4/D10 updates per rank in the request window, including preparation, and nonzero layout changes. Successful request logs contained no ERROR, Traceback or scheduler exception.
- The deliberately frequent PD migration configuration took 31.867 s OFF / 36.623 s ON for preparation plus formal requests. This establishes switch functionality, not PD net speedup.

The copy oracle is not a full model-accuracy benchmark. GPU hardware, full task-accuracy evaluation, all-layer/all-rank weight export and hardware-utilization measurements are not claimed.

## Earlier integration: fixed-workload single-node performance

These results belong to the frozen `singlecapture_v1` candidate before the community port. The workload was original AIS copy, N320/C128/target 6 req/s, actual input 16388 and output 1024 tokens, cache hit 0%, TP/EP/attention-DP 16 with DP1. DSpark and Compact were disabled. Each fresh process used trivial initial placement, with no offline optimized layout. Source, environment and 16x64 preparation requests matched; only `--enable-eplb` differed.

| Run order | Output tokens/s | Mean TTFT (s) | Mean TPOT (ms) | Mean E2E (s) | Success / failed |
|---|---:|---:|---:|---:|---:|
| First ON | 1397.65 | 19.79 | 55.10 | 76.19 | 320 / 0 |
| Fresh OFF | 1311.01 | 25.62 | 55.80 | 82.67 | 320 / 0 |
| Fresh ON | 1385.85 | 20.45 | 55.20 | 76.90 | 320 / 0 |

The two ON runs improved output throughput by 6.61% and 5.71% versus the fresh OFF. Latest ON mean TTFT/TPOT/E2E changed by -20.19%/-1.08%/-6.98%. All 320 input/output triplets matched exactly; each run passed the copy oracle 320/320. Preparation plus formal time decreased by 3.17%, excluding loading, AIS initialization, orchestration and collection.

This is reproduction on one fixed workload, without a confidence interval or a general/long-term performance claim. It is not a measured speedup of the rebased community source. Earlier positive/negative PD results and failed runs remain linked in the pinned integration report.

![Earlier integration performance screenshot](ascend_dynamic_eplb_upstream_20261024/integration_performance.png)
