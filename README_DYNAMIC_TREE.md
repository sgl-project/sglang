## USAGE

```bash
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
python3 -m sglang.launch_server \
--model-path Qwen/Qwen3-30B-A3B \
--tp 2 \
--speculative-algorithm EAGLE3 \
--speculative-draft-model-path Tengyunw/qwen3_30b_moe_eagle3 \
--speculative-num-steps 2 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 3 \
--mem-fraction-static 0.65 \
--cuda-graph-max-bs 10 \
--attention-backend fa3 \
--speculative-eagle-mab-algorithm "DIRECT_MAP" \
--speculative-eagle-mab-config-path "test.json" \
--disable-radix-cache \
--disable-overlap-schedule
```

important args:

+ speculative-eagle-mab-algorithm: ["EG", "UCB1", "DIRECT_MAP"], default is "DIRECT_MAP"
+ speculative-eagle-mab-config-path: config path about speculative decoding


## config file format
``` json
{
    "batch_size_1": "<speculative_num_steps>_<topk>_<draft_tokens>",
    "batch_size_2": "<speculative_num_steps-1>_<topk-1>_<draft_tokens-1>",
    // ...
    "batch_size_n": "<speculative_num_steps-n>_<topk-n>_<draft_tokens-n>",
}
```

for example:
``` json
{
    "1": "6_8_10",
    "2": "5_6_8",
    "3": "3_4_6",
    "5": "2_1_3"
}
```

Select the largest configuration that does not exceed the current batch size. (if current batch size = 4, choose "3_4_6" for above config)

## Test method
``` bash
python3 -m sglang.test.send_one --batch-size bs
```
adjust `bs` and observe the accept length

