because there will be multiple weight precisions:
w16, w8, w4

we need to support the loading of model weights together into VRAM
    the groupgemm of compute may need to compute on different expert weights

follow reference in reference.md for more details on their weight loading

different from the reference code where no TP or EP is supported, here we need to support both TP and EP:
    in ep the default assignment is to have the different precisions of the experts on the same rank
        however, we need to support different assignments, for example, users can custome the precision assignment like:
            INT4: {rank 0: experts 0-15, ...}
            this leaves potential redundancy of low bit precision weights to speed up by mitigating load imbalance
    this should have low priority since our main focus is to use different precisions for different experts
    *** for now we can first assume everything is done on a single rank...

similar to reference, you should have one flag indicating we want to use heter moe instead of the vanilla, we should have flags on 
    1. what precision schemes are of our interest, for example {"a8w8", "a16w4"}
    2. what kind of criteria: default is token count, and we should be satisfied with that primarily
    3. percentage of each precision
(all similar to reference)

---
## sglang class hierarchy and config flow (added during step 0.1 refinement)

### existing class hierarchy (DO NOT modify these — extend them)
```
FusedMoE(torch.nn.Module)                           # python/sglang/srt/layers/moe/fused_moe_triton/layer.py L137
    __init__(num_experts, hidden_size, intermediate_size, layer_id, top_k, quant_config, ...)
    forward(hidden_states, topk_output) → forward_impl()
    forward_impl():
        dispatch_output = self.dispatcher.dispatch(hidden_states, topk_output)
        combine_input   = self.run_moe_core(dispatch_output)
        final           = self.dispatcher.combine(combine_input)
    run_moe_core(dispatch_output):
        return self.quant_method.apply(layer=self, dispatch_output=dispatch_output)
```

### config flow: CLI flag → model layer
```
ServerArgs.heter_precision_config (str, path to JSON)     # python/sglang/srt/server_args.py
    ↓
ModelRunner.load_model()                                   # python/sglang/srt/model_executor/model_runner.py
    reads JSON, stores in model_config or passes as kwarg
    ↓
_initialize_model(model_config, load_config, quant_config) # python/sglang/srt/model_loader/loader.py L258
    kwargs = {"config": hf_config, "quant_config": quant_config, "heter_config": heter_config}
    ↓
Qwen3MoeForCausalLM.__init__(config, quant_config, ...)   # python/sglang/srt/models/qwen3_moe.py
    threads heter_config down to each MoE block
    ↓
Qwen3MoeSparseMoeBlock.__init__(layer_id, config, quant_config, ...)  # L233
    self.experts = get_moe_impl_class(quant_config)(...)   # L258
    ↓
get_moe_impl_class(quant_config)                           # python/sglang/srt/layers/moe/ep_moe/layer.py L785
    currently returns FusedMoE by default
    → MODIFY to return HeterFusedMoE when heter_config is present
```

### weight parameter naming convention (per FusedMoE)
    w13_weight:       [num_local_experts, 2*intermediate_size_per_partition, hidden_size]
    w2_weight:        [num_local_experts, hidden_size, intermediate_size_per_partition]
    for Marlin INT4: w13_qweight (packed uint8), w13_scales, w13_qzeros (optional)
    for INT8 W8A8:   w13_weight (int8), w13_weight_scale (float32, per-channel)

### proposed new files
    python/sglang/srt/layers/moe/heter_moe.py       — HeterFusedMoE class (extends FusedMoE or wraps two FusedMoE instances)
    python/sglang/srt/layers/moe/heter_policy.py     — dispatch policy (BaseHeterPolicy, TokenCountPolicy, HeterDispatchPlan)

### design decision: composition vs inheritance
    option A (inheritance): HeterFusedMoE extends FusedMoE, overrides run_moe_core()
    option B (composition): HeterFusedMoE owns two quant_method instances (one per group), 
        shares the dispatcher but calls two different apply() methods
    recommendation: option B — cleaner separation, each group's weights/runner are fully independent
