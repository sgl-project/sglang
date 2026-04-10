to guarantee the best accuracy along with efficiency, we should load tuned models from different paths.
    (as a contrary, loading only bf16 models and provides naive quantization is suboptimal)
    (tuned model weights should have been processed with techniques like GPTQ/AWQ)


check if 
    1. INT4 weights / INT8 weight can be found on huggingface for qwen3-30b-a3b, if so download
    2. sglang includes support for weite quantization with GPTQ/AWQ, if all provided
        we need a pipeline that produces these quantized weights for our model

---
## huggingface availability and model dimensions (added during step 0.1 refinement)

### Qwen3-30B-A3B model dimensions
    hidden_size              = 2048
    moe_intermediate_size    = 768     (per expert)
    num_experts              = 128
    num_experts_per_tok      = 8       (top-8 routing)
    num_hidden_layers        = 48
    norm_topk_prob           = True

### expert weight shapes (per expert, single rank, TP=1)
    w13 (gate+up, fused): [1536, 2048]   (= 2 * 768, 2048)
    w2 (down):            [2048, 768]

### pre-quantized weights on HuggingFace — CONFIRMED AVAILABLE
    | model_id                                      | method    | bits | group_size | status          |
    |-----------------------------------------------|-----------|------|------------|-----------------|
    | Qwen/Qwen3-30B-A3B-GPTQ-Int4                 | GPTQ      | 4    | 128        | ✅ official     |
    | JunHowie/Qwen3-30B-A3B-GPTQ-Int8              | GPTQ      | 8    | 128        | ✅ community    |
    | JunHowie/Qwen3-30B-A3B-GPTQ-Int4              | GPTQ      | 4    | 128        | ✅ community    |
    | Sophia-AI/Qwen3-30B-A3B-Instruct-2507-AWQ-W4A16 | AWQ    | 4    | —          | ✅ community    |

    for this project:
        BF16 weights:  Qwen/Qwen3-30B-A3B                (original, ~60GB)
        INT4 weights:  Qwen/Qwen3-30B-A3B-GPTQ-Int4      (official GPTQ, ~17GB)
        INT8 weights:  JunHowie/Qwen3-30B-A3B-GPTQ-Int8   (community GPTQ, ~25GB)

### SGLang quantization support
    SGLang can LOAD pre-quantized GPTQ/AWQ weights but does NOT have built-in quantization scripts.
    for generating GPTQ/AWQ weights: use AutoGPTQ or AutoAWQ externally.
    SGLang quantization methods registry: python/sglang/srt/layers/quantization/__init__.py L54-80
    relevant entries: "gptq", "gptq_marlin", "awq", "awq_marlin", "moe_wna16", "w8a8_int8"

### weight loading strategy for heter-moe
    each precision group loads from a SEPARATE HF model checkpoint:
        group "cold" (INT4): loads from Qwen/Qwen3-30B-A3B-GPTQ-Int4
        group "hot"  (BF16): loads from Qwen/Qwen3-30B-A3B
    the heter_config JSON specifies per-group checkpoint paths (see goal.md for schema)
    at init time, each group's weights are loaded into separate parameter tensors
    at runtime, the dispatch policy selects which group's weights to use per expert

### download commands (for step 5)
    huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /data/heter-moe/models/qwen3-30b-a3b-bf16
    huggingface-cli download Qwen/Qwen3-30B-A3B-GPTQ-Int4 --local-dir /data/heter-moe/models/qwen3-30b-a3b-gptq-int4
    huggingface-cli download JunHowie/Qwen3-30B-A3B-GPTQ-Int8 --local-dir /data/heter-moe/models/qwen3-30b-a3b-gptq-int8
