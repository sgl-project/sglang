priority of tasks:

---- DONE ----
0. read the .hetermoe and understand all .md
0.1 refine all .md and make yourself comfortable with the instructions

---- TODO ----
[DO NOT MODIFY .HETERMOE UNLESS I EXPLICITLY ASK YOU TO MODIFY!!!]
1. implementing the system (migrate from trtllm's mix {bf16, nvfp4} to sglang's mix{marlin int4, int8})
2. evaluating the implementation kernel-wise with varying fake inputs&weights
    a. usability the implementation should run
    b. efficiency: the mixture of precision should show performance speedup compared to naive bf16
        and the speedup should be reasonable (lies between the two extreme precisions)
    c. accuracy: the implementation should provide negligible accuracy loss (MSE)
        show that the MSE of two gemm kernel lies between two extreme precisions
    d. make necessary plots and tables to demonstrate efficiency and accuracy
3. collect the routing stats for real load imbalance
4. evaluating the implementation layer-wise with collected real routing stats but still fake inputs&weights
    plese make plots and tables to demonstrate efficiency and accuracy

    (collect and present results)
    (wait until further instructions)

5. download/quantize the weights for lower precisions
6. evaluating the implementation E2E:
    a. accuracy metrics of both perplexity and downstream tasks performance (prioritize shorter benchmarks like hellaswag and gsm8k)
    b. efficiency: TTFT, ITL (inter token latency)


failure policy:
    when a test fails:
        1. attempt to fix (max 2 attempts)
        2. if fixed → commit and continue
        3. if NOT fixed after 2 attempts →
            a. log the failure: what failed, what was tried, error output
            b. mark the test as SKIPPED with a TODO comment explaining the issue
            c. commit current state with message indicating the known failure
            d. proceed to the next task
        do NOT block the entire pipeline on a single failing test

    severity tiers:
        BLOCKING:  core forward pass doesn't run at all, shape mismatch, crash → must fix before proceeding
        NON-BLOCKING: accuracy slightly off threshold, efficiency not as expected,
            torch.compile/cudagraph edge case, flaky numeric tolerance → log and skip, continue

    the goal is forward progress. a skipped test with a clear log is better than a stalled project.

versioning and logging:
after each feature and test is implemented, commit and push with informative messages
after the test passed and feature is corrected, commit and push

make sure you log in some path that include the information I should be aware of
    these logs and commits should be pointed to each other when needed
    failures should be logged to /data/heter-moe/logs/failures/ with timestamps

test result logging:
    every test run (pass or fail) must have its output saved to /data/heter-moe/logs/tests/
    filename convention: <test_name>_<step>_<timestamp>.log
    contents: full pytest stdout/stderr, including assertion values and tracebacks
    after saving the log, commit and push the log alongside the code changes
    this way each commit that touches tests also includes the corresponding test output
    example workflow:
        1. run pytest → tee output to /data/heter-moe/logs/tests/test_heter_moe_step2_20260410.log
        2. git add the log + code changes
        3. commit: "heter-moe step 2: kernel unit tests (7/8 pass, see logs/tests/...)"
        4. push

---
## concrete file map per step (added during step 0.1 refinement)

### step 1: implementation — files to create/modify
    NEW  python/sglang/srt/layers/moe/heter_moe.py           — HeterFusedMoE class
    NEW  python/sglang/srt/layers/moe/heter_policy.py         — BaseHeterPolicy, TokenCountPolicy, HeterDispatchPlan
    MOD  python/sglang/srt/server_args.py                      — add --heter-precision-config flag
    MOD  python/sglang/srt/model_loader/loader.py              — thread heter_config to model init
    MOD  python/sglang/srt/layers/moe/ep_moe/layer.py          — get_moe_impl_class returns HeterFusedMoE when heter_config present
    MOD  python/sglang/srt/models/qwen3_moe.py                 — pass heter_config to MoE block constructor

### step 2: kernel-level evaluation
    NEW  test/srt/layers/moe/test_heter_moe.py                — unit tests (fake weights)
    NEW  scripts/heter_moe_benchmark_kernels.py                — kernel profiling script
    OUT  /data/heter-moe/profiles/groupgemm/                    — CSV + plots

### step 3: routing stats collection
    NEW  scripts/heter_moe_collect_routing.py                  — offline routing data collector
    OUT  /data/heter-moe/routing_stats/                         — per-batch JSON files

### step 4: layer-wise evaluation
    NEW  scripts/heter_moe_benchmark_layers.py                 — layer-level profiling with real routing stats
    OUT  /data/heter-moe/profiles/layerwise/                    — CSV + plots

### step 5: weight download
    use huggingface-cli (see weights.md for commands)
    OUT  /data/heter-moe/models/                                — BF16 + INT4 + INT8 checkpoints

### step 6: E2E evaluation
    NEW  scripts/heter_moe_eval_e2e.py                         — perplexity + lm-eval runner
    NEW  test/registered/quant/test_heter_moe_e2e.py           — e2e test cases
    OUT  /data/heter-moe/results/e2e/                           — accuracy + latency results