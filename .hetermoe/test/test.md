ALL the simplest working unit of our implementation should be tested thoroughly. 
try to come up with tests that touches:
    1. usability
    2. accuracy
    3. efficiency
    4. corner case

some tests can be constructed with fake weights(layers) and fake inputs refer to plan.md for designing the tests
other tests requries the use of real model weights and inputs... reduce the frequency of these large tests

you should make yourself guidelines for both unit tests and e2e integration tests in this folder!
these two usually have different metrics for accuracy, for example, unit tests with fake inputs simply needs to compute MSE for accuracy
    however, e2e integration tests require either simple perplexity or to run lm-eval for downstream scoring

efficiency tests should be tested with
    1. torch compile
    2. cudagraph
    3. warm up
    4. a large enough working set to avoid L2 camping [IMPORTANT]

tests are prefered in pytest

---
## failure handling policy

### on test failure: fix-or-skip, never block
    1. attempt fix (max 2 tries)
    2. if still failing → classify severity:
        BLOCKING (crash, shape error, import error)  → must resolve, but can simplify the test
        NON-BLOCKING (numeric tolerance, perf threshold, compile edge case) → skip and log
    3. for skipped tests:
        - mark with @pytest.mark.skip(reason="[HETER-MOE] <description of failure>")
        - add TODO comment with: what failed, error message, what was tried
        - log to /data/heter-moe/logs/failures/<test_name>_<timestamp>.txt
    4. commit with message like: "heter-moe: skip test_X (known issue: <reason>), proceed to step N"
    5. move on to next task

### numeric tolerance guidelines
    unit tests (fake weights):    MSE tolerance = 1e-3 (relaxed, fake data)
    e2e tests (real weights):     perplexity within 5%, downstream score within 2 points
    efficiency tests:             expect speedup but don't hard-fail on regression
                                  log the actual numbers and flag for review
    if a tolerance is too tight → loosen it, log the actual value, continue

---
## sglang test patterns and concrete test plan (added during step 0.1 refinement)

### existing test references in sglang
    test/registered/quant/test_marlin_moe.py      — Marlin MoE test (model-level)
    test/registered/quant/test_int4fp8_moe.py      — INT4/FP8 MoE test
    test infrastructure: test/srt/test_utils.py     — CustomTestCase base class, server fixtures

### proposed test file location
    test/srt/layers/moe/test_heter_moe.py          — unit tests (fake weights, kernel-level)
    test/registered/quant/test_heter_moe_e2e.py    — e2e tests (real weights, model-level)

### unit tests (fake weights, GPU required, fast)
    class TestHeterMoEKernel(pytest.TestCase):

    1. test_marlin_int4_single_expert:
        create random INT4-packed weights for 1 expert, call fused_marlin_moe, check output shape/dtype
    2. test_bf16_single_expert:
        create random BF16 weights for 1 expert, call invoke_fused_moe_kernel, check output shape/dtype
    3. test_mixed_precision_forward:
        create HeterFusedMoE with 128 experts (100 cold INT4 + 28 hot BF16)
        run forward with random input and random topk_ids
        check output shape matches input shape
    4. test_accuracy_mse_between_precisions:
        for same fake weights (quantize BF16 → INT4):
        output_bf16 = forward with all-BF16
        output_int4 = forward with all-INT4
        output_mixed = forward with mixed
        assert MSE(output_mixed, output_bf16) <= MSE(output_int4, output_bf16)
    5. test_dispatch_policy_token_count:
        given known topk_ids, verify TokenCountPolicy assigns hot experts correctly
    6. test_dispatch_policy_all_cold / test_dispatch_policy_all_hot:
        edge cases: all experts cold (ratio=1.0) or all hot (ratio=0.0)
    7. test_expert_assignment_deterministic:
        same input → same assignment (required for cudagraph)
    8. test_config_parsing:
        valid/invalid heter_config JSON → correct parsing or error

### efficiency tests (GPU required, slow)
    class TestHeterMoEEfficiency(pytest.TestCase):

    1. test_mixed_faster_than_bf16:
        for batch_sizes in [1, 8, 64, 256]:
        time_bf16 = benchmark all-BF16 forward
        time_mixed = benchmark mixed forward (80% INT4, 20% BF16)
        assert time_mixed < time_bf16 (at least for small batch)
    2. test_torch_compile_compatibility:
        compiled_forward = torch.compile(heter_moe.forward)
        check no errors, output matches eager
    3. test_cudagraph_compatibility:
        capture heter_moe.forward in cuda graph
        replay, check output matches eager

### e2e tests (real weights, multi-GPU, slow — step 6)
    1. test_perplexity:
        load Qwen3-30B-A3B with heter config
        compute perplexity on wikitext-2
        assert ppl_heter < ppl_bf16 * 1.05 (within 5%)
    2. test_lm_eval_hellaswag:
        run lm-eval with hellaswag
        assert score_heter >= score_bf16 - 2.0 (within 2 points)