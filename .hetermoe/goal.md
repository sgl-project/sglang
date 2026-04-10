switching from regular model serving to our model serving would require minimal effort
for example by passing in 
--heter-precision-config heter_config.json/yaml

the heter_config should include the configs from model.md, also including the locations of model weights

the entire implementation should be naturally supporting torch.compile as well as cudagraph

---
## concrete CLI and config schema (added during step 0.1 refinement)

### CLI usage
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B \
    --heter-precision-config /path/to/heter_config.json
```

### heter_config.json schema
```json
{
    "groups": [
        {
            "name": "cold",
            "quant_method": "gptq",
            "num_bits": 4,
            "checkpoint": "/data/heter-moe/models/qwen3-30b-a3b-gptq-int4",
            "size_ratio": 0.8
        },
        {
            "name": "hot",
            "quant_method": null,
            "num_bits": 16,
            "checkpoint": "/data/heter-moe/models/qwen3-30b-a3b-bf16",
            "size_ratio": 0.2
        }
    ],
    "policy": "token_count",
    "policy_params": {}
}
```

### config fields
    groups[].name:          human label for the precision group
    groups[].quant_method:  "gptq", "awq", "w8a8_int8", or null (BF16)
    groups[].num_bits:      4 / 8 / 16
    groups[].checkpoint:    path to pre-quantized HF model weights
    groups[].size_ratio:    fraction of experts in this group (all must sum to 1.0)
    policy:                 "token_count" (primary), "random" (testing), "fixed" (manual assignment)
    policy_params:          policy-specific parameters (e.g., threshold values)

### where to add the flag
    python/sglang/srt/server_args.py — add heter_precision_config field to ServerArgs
    register in argparse with type=str, default=None
