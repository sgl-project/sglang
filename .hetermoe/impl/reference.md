reference codebase:      https://github.com/hjzccc/TensorRT-LLM.git
reference branch name:   double-fused-moe

NOTE (step 0.1): the TRT-LLM repo is NOT cloned on this machine.
    all reference understanding comes from the summary below plus the git remote.
    this is sufficient — no need to clone TRT-LLM locally.

there's no need to refine this reference codebase. you only need to understand what is a working version for 
    a mixture of {bf16, nvfp4}
and use the same logic for our current sglang on
    a mixture of {bf16/INT8, INT4}

reference summary:
# BRANCH CONTEXT — huanchen-twoGemmMoe
> Machine-readable context for AI agents working on this branch.
> Base: main | Commits: 2 | Delta: +2184 -1 | Files: 11
## FEATURE
Heterogeneous precision MoE backend ("HETER") for TensorRT-LLM.
Stores N full weight sets (one per precision group, e.g. NVFP4 + BF16) per expert.
Dispatches each group through separate `torch.ops.trtllm.fused_moe()` calls, sums outputs.
Expert-to-group assignment is dynamic via pluggable dispatch policies using router signals.
## FILE MAP
```
STATUS  PATH                                                          ROLE
------  ----                                                          ----
ADD     _torch/modules/fused_moe/fused_moe_heter.py        (817L)    CORE class: HeterCutlassFusedMoE(CutlassFusedMoE)
ADD     _torch/modules/fused_moe/policy/__init__.py          (30L)    Package init, re-exports
ADD     _torch/modules/fused_moe/policy/dispatch_plan.py     (73L)    DispatchPlan dataclass (expert→group mapping)
ADD     _torch/modules/fused_moe/policy/strategies.py       (330L)    BaseDispatchPolicy ABC + 3 concrete policies
ADD     tests/unittest/_torch/modules/moe/test_heter_moe.py (787L)    Unit tests (20 tests)
MOD     _torch/modules/fused_moe/__init__.py                  (+9)    Exports new symbols
MOD     _torch/modules/fused_moe/create_moe.py               (+24)    Factory registration for HETER
MOD     llmapi/llm_args.py                                   (+15)    "HETER" backend literal + heter_config field
MOD     _torch/pyexecutor/model_loader.py                     (+5)    Passes heter_moe_config to extra_attrs
ADD     .agent/FileChange.md                                          Change log (human)
ADD     .agent/TODOs.md                                               Future work tracker
```
All paths relative to `tensorrt_llm/` unless prefixed with `.agent/` or `tests/`.
## CLASS HIERARCHY
```
CutlassFusedMoE (existing, fused_moe_cutlass.py)
  └── HeterCutlassFusedMoE (fused_moe_heter.py)
        owns: List[_GroupDescriptor]       ← parsed from heter_config
        owns: BaseDispatchPolicy           ← pluggable, default=RandomDispatchPolicy(seed=42)
        owns: weight/scale caches          ← rebuilt when dispatch assignment changes
BaseDispatchPolicy (ABC, strategies.py)
  ├── RandomDispatchPolicy          ← deterministic shuffle by ratio, no signals
  ├── ConfidenceThresholdPolicy     ← ranks by mean routing weight, high→BF16
  └── ExpertLoadPolicy              ← ranks by activation frequency, hot→BF16
DispatchPlan (dataclass, dispatch_plan.py)
  └── group_assignments: List[List[int]]   ← group_assignments[i] = sorted expert IDs for group i
```
## DATA FLOW
```
User config:
  MoeConfig(backend="HETER", heter_config={"groups": [...]})
    │
    ▼
  llm_args.py validates → model_loader.py stores in extra_attrs['heter_moe_config']
    │
    ▼
  create_moe.py: get_moe_cls("HETER") → HeterCutlassFusedMoE
    │
    ▼
  __init__: parse heter_config → _group_descs[], init RandomDispatchPolicy
    │
    ▼
  post_load_weights(): super() + _recompute_dispatch() → initial assignment (no signals)
    │
    ▼
  forward_chunk(x, router_logits):
    stash _pending_router_logits = router_logits
    super().forward_chunk()  ← calls run_moe() synchronously
    clear _pending_router_logits
    │
    ▼
  run_moe(x, token_selected_experts, token_final_scales, ...):
    1. _recompute_dispatch(experts, scales, router_logits)
       → policy.assign() → DispatchPlan
       → if assignment changed: _build_group_caches()
    2. Fast path: single group covering all experts → super().run_moe()
    3. Multi-group: for each active group:
       a. Remap expert IDs via remap table (global→local)
       b. Zero scales for non-group experts
       c. torch.ops.trtllm.fused_moe() with group's weight subset
       d. Accumulate output
    4. Return sum of all group outputs
```
## CONFIG SCHEMA
```python
MoeConfig(
    backend="HETER",
    heter_config={
        "groups": [
            {
                "name": "cold",              # str: human label
                "quant_algo": "NVFP4",       # str|None: "NVFP4" or null (BF16)
                "size_ratio": 0.80,          # float: fraction of experts (all must sum to 1.0)
                "checkpoint": "/path/to/...", # str|None: weight checkpoint path
            },
            {
                "name": "hot",
                "quant_algo": None,          # null = BF16/FP16
                "size_ratio": 0.20,
                "checkpoint": "/path/to/...",
            },
        ]
    }
)
```
## KEY CONSTRAINTS
- Supported quant_algos: {None (BF16/FP16), QuantAlgo.NVFP4}
- size_ratio values MUST sum to 1.0 (abs_tol=1e-3)
- Every expert appears in exactly one group (DispatchPlan.validate enforces)
- forward_chunk expects BF16 input (asserts not Fp4QuantizedTensor)
- Policy assign() runs in eager mode (inside moe_custom_op, not traced by Dynamo)
- Under CUDA graph: dispatch assignment is baked at capture time
## INCOMPLETE (phase2 TODOs in code)
| TODO | Location | What's Missing |
|------|----------|----------------|
| Dual weight loading | `_build_group_caches()`, `post_load_weights()` | All groups subset from ONE parent weight set. Need per-group checkpoint loading. |
| Per-group quant flags | `run_moe()` L764-771 | All groups use parent's quant flags. Need to branch on `desc.quant_algo`. |
| NVFP4 input quantization | `run_moe()` | FP4 groups need `torch.ops.trtllm.fp4_quantize` on input before fused_moe. |
| Per-group quant scales | `_build_group_caches()` | Each group needs its own scales (NVFP4 has block scales, BF16 has none). |
| Tests not executed | `test_heter_moe.py` | Requires Docker container with C++ bindings or mocking. |
## ENTRY POINTS FOR COMMON TASKS
| Task | Start Here |
|------|-----------|
| Add new dispatch policy | Subclass `BaseDispatchPolicy` in `policy/strategies.py`, implement `assign()` |
| Change default policy | `HeterCutlassFusedMoE.__init__` L233 |
| Add new quant_algo support | `_SUPPORTED_QUANT_ALGOS` in `fused_moe_heter.py` L74, `_resolve_quant_algo()` |
| Modify per-group dispatch logic | `HeterCutlassFusedMoE.run_moe()` L670+ |
| Modify weight cache building | `_build_group_caches()` L499+ |
| Modify config schema | `MoeConfig` in `llm_args.py`, `_validate_heter_config()` in `fused_moe_heter.py` |
| Run tests | `pytest tests/unittest/_torch/modules/moe/test_heter_moe.py` (needs TRT-LLM C++ bindings) |
## DEPENDENCIES (this branch's code imports from)
```
tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass.CutlassFusedMoE  ← parent class
tensorrt_llm._torch.modules.fused_moe.routing.BaseMoeRoutingMethod
tensorrt_llm._torch.utils.Fp4QuantizedTensor
tensorrt_llm._torch.model_config.ModelConfig
tensorrt_llm.models.modeling_utils.QuantAlgo
tensorrt_llm.logger
torch.ops.trtllm.fused_moe                                                ← C++ kernel (nanobind)
```

---
## sglang equivalence mapping (added during step 0.1 refinement)

| TRT-LLM concept                | SGLang equivalent                                                       | file                                                                      |
|---------------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------|
| CutlassFusedMoE (parent class)  | FusedMoE                                                                | python/sglang/srt/layers/moe/fused_moe_triton/layer.py (L137)            |
| HeterCutlassFusedMoE            | **new: HeterFusedMoE** (to be created)                                  | python/sglang/srt/layers/moe/heter_moe.py (proposed)                     |
| torch.ops.trtllm.fused_moe (BF16) | invoke_fused_moe_kernel (Triton)                                     | python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py               |
| torch.ops.trtllm.fused_moe (NVFP4) | fused_marlin_moe (JIT Marlin)                                       | python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py        |
| —                               | Triton runner w/ use_int8_w8a8=True (INT8 path)                         | python/sglang/srt/layers/quantization/w8a8_int8.py                        |
| MoeConfig.heter_config           | ServerArgs.heter_precision_config → JSON file                           | python/sglang/srt/server_args.py                                          |
| llm_args.py validation           | _initialize_model kwargs in loader                                      | python/sglang/srt/model_loader/loader.py (L258)                          |
| BaseDispatchPolicy               | **new: BaseHeterPolicy** (to be created)                                | python/sglang/srt/layers/moe/heter_policy.py (proposed)                   |
| ExpertLoadPolicy                 | **new: TokenCountPolicy** (to be created)                               | python/sglang/srt/layers/moe/heter_policy.py (proposed)                   |
| DispatchPlan                     | **new: HeterDispatchPlan** (to be created)                              | python/sglang/srt/layers/moe/heter_policy.py (proposed)                   |
| _build_group_caches              | not needed — we load separate weight checkpoints directly per group     | (weight loading handled during model init)                                |
| FusedMoEMethodBase               | FusedMoEMethodBase (already exists)                                     | python/sglang/srt/layers/quantization/base_config.py (L84)               |
| get_moe_cls("HETER")            | get_moe_impl_class returns FusedMoE by default; override for heter      | python/sglang/srt/layers/moe/ep_moe/layer.py (L785)                      |

key architectural difference:
    TRT-LLM subsets ONE parent weight set into per-group caches at runtime.
    SGLang approach: load SEPARATE pre-quantized checkpoints per group at init time.
    this avoids runtime weight subsetting and is simpler + more memory-efficient.