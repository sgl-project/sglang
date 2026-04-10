we only require naive profiling at this stage:
    for different batch size, for each layer, what is the load per expert 

for datasets, we can simply use share gpt
for snapshots, take one snapshot for prefill and one snapshot for decoding
we need the load imbalance information for all layers in the model

but we don't need too much data points, simply pick batch size from [2^0, ..., 2^10]
 and one prefill one decode for each datapoint

each datapoint is a file, in such format 
{
    "transformer_block_{i}": [] : list of 128 interger numbers
}

all the datapoints should be collected in a folder with filenames indicating when(batch size; prefill/decode) is the data collected

---
## implementation approach (added during step 0.1 refinement)

### data collection method
    option A: instrument the router inside SGLang server to dump topk_ids per layer per batch
        hook into TopK.forward() at python/sglang/srt/layers/moe/topk.py
        after topk_ids = ... , do torch.bincount(topk_ids.flatten(), minlength=128)
        dump to JSON per layer
    option B: run model forward pass offline with ShareGPT prompts, capture routing decisions
        simpler, no server needed, just model.forward() with hooks

    recommendation: option B for data collection (simpler, offline)

### concrete steps
    1. download ShareGPT dataset (use sglang benchmark tooling or huggingface datasets)
    2. load Qwen3-30B-A3B in BF16 on available GPUs
    3. for each batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        a. construct a batch of ShareGPT prompts
        b. run prefill forward pass, hook each MoE layer to capture topk_ids
        c. compute per-expert token counts: torch.bincount(topk_ids.view(-1), minlength=128)
        d. save as JSON: {"transformer_block_0": [counts...], "transformer_block_1": [...], ...}
        e. repeat for decode (single new token per sequence)
    4. filenames: batch{N}_prefill.json, batch{N}_decode.json

### output directory
    /data/heter-moe/routing_stats/
    
### expected observation
    load imbalance increases with smaller batch sizes (fewer tokens → more variance)
    some experts are consistently hot across layers (popularity bias from training)
    this motivates dynamic per-batch precision assignment
