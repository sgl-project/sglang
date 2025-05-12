## Experiment - bench_serving | DeepSeek-V3

| Configuration | Input token throughput (tok/s) | Output token throughput (tok/s) |
|-------------|----------------|----------------|
| attn=fa3,random_input_len=1024,random_output_len=1024 | 2216.96    | 2268.68    |
| attn=flashinfer,random_input_len=1024,random_output_len=1024 | 2226.81    | 2278.76    |
| attn=fa3,random_input_len=8192,random_output_len=1024 | 8369.41    | 1078.81    |
| attn=flashinfer,random_input_len=8192,random_output_len=1024 | 8373.39    | 1079.32    |
| attn=fa3,random_input_len=8192,random_output_len=8192 | 1385.71    | 1482.43    |
| attn=flashinfer,random_input_len=8192,random_output_len=8192 | 1418.84    | 1517.88    |

### Commands

#### attn=fa3,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend fa3 --tp 8 --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 1024 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend flashinfer --tp 8 --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 1024 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend fa3 --tp 8 --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend flashinfer --tp 8 --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server --attention-backend fa3 --tp 8 --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 8192
```

#### attn=flashinfer,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server --attention-backend flashinfer --tp 8 --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 8192
```

## Experiment - bench_serving | Meta-Llama-3.1-8B-Instruct

| Configuration | Input token throughput (tok/s) | Output token throughput (tok/s) |
|-------------|----------------|----------------|
| attn=fa3,random_input_len=1024,random_output_len=1024 | 5232.96    | 5355.04    |
| attn=flashinfer,random_input_len=1024,random_output_len=1024 | 4819.61    | 4932.05    |
| attn=fa3,random_input_len=8192,random_output_len=1024 | 17652.56   | 2275.40    |
| attn=flashinfer,random_input_len=8192,random_output_len=1024 | 13886.84   | 1790.00    |
| attn=fa3,random_input_len=8192,random_output_len=8192 | 2419.37    | 2588.25    |
| attn=flashinfer,random_input_len=8192,random_output_len=8192 | 1641.00    | 1755.54    |

### Commands

#### attn=fa3,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend fa3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 1024 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend flashinfer --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 1024 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend fa3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend flashinfer --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server --attention-backend fa3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 8192
```

#### attn=flashinfer,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server --attention-backend flashinfer --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 8192
```
## Experiment - bench_serving_spec_decode | DeepSeek-V3

| Configuration | Input token throughput (tok/s) | Output token throughput (tok/s) |
|-------------|----------------|----------------|
| attn=fa3,random_input_len=1024,random_output_len=1024 | 739.87     | 757.13     |
| attn=flashinfer,random_input_len=1024,random_output_len=1024 | 712.76     | 729.38     |
| attn=fa3,random_input_len=8192,random_output_len=1024 | 4742.48    | 611.30     |
| attn=flashinfer,random_input_len=8192,random_output_len=1024 | 4673.64    | 602.43     |
| attn=fa3,random_input_len=8192,random_output_len=8192 | 862.90     | 923.14     |
| attn=flashinfer,random_input_len=8192,random_output_len=8192 | 870.02     | 930.75     |

### Commands

#### attn=fa3,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/public/sharing/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 1024 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/public/sharing/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 1024 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/public/sharing/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/public/sharing/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/public/sharing/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 8192
```

#### attn=flashinfer,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/public/sharing/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 8192
```

## Experiment - bench_serving_spec_decode | Meta-Llama-3.1-8B-Instruct

| Configuration | FAILED | Input token throughput (tok/s) | Output token throughput (tok/s) |
|-------------|----------------|----------------|----------------|
| attn=fa3,random_input_len=1024,random_output_len=1024 |  | 3456.01    | 3536.64    |
| attn=flashinfer,random_input_len=1024,random_output_len=1024 |  | 3375.78    | 3454.53    |
| attn=fa3,random_input_len=8192,random_output_len=1024 | Client failed with code 1 |  |  |
| attn=flashinfer,random_input_len=8192,random_output_len=1024 |  | 10417.88   | 1342.85    |
| attn=fa3,random_input_len=8192,random_output_len=8192 | Client failed with code 1 |  |  |
| attn=flashinfer,random_input_len=8192,random_output_len=8192 |  | 1230.00    | 1315.85    |

### Commands

#### attn=fa3,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 1024 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 1024 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 8192
```

#### attn=flashinfer,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --num-prompts 200 --random-input-len 8192 --random-output-len 8192
```
## Experiment - gsm8k | DeepSeek-V3

| Configuration | Accuracy | Output throughput |
|-------------|----------------|----------------|
| attn=fa3 | 0.970 | 573.756 token/s |
| attn=flashinfer | 0.960 | 570.449 token/s |

### Commands

#### attn=fa3

```
python3 -m sglang.launch_server --attention-backend fa3 --tp 8 --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205
python3 ../gsm8k/bench_sglang.py
```

#### attn=flashinfer

```
python3 -m sglang.launch_server --attention-backend flashinfer --tp 8 --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205
python3 ../gsm8k/bench_sglang.py
```

## Experiment - gsm8k | Meta-Llama-3.1-8B-Instruct

| Configuration | Accuracy | Output throughput |
|-------------|----------------|----------------|
| attn=fa3 | 0.755 | 3837.773 token/s |
| attn=flashinfer | 0.760 | 3068.449 token/s |

### Commands

#### attn=fa3

```
python3 -m sglang.launch_server --attention-backend fa3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct
python3 ../gsm8k/bench_sglang.py
```

#### attn=flashinfer

```
python3 -m sglang.launch_server --attention-backend flashinfer --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct
python3 ../gsm8k/bench_sglang.py
```
## Experiment - gsm8k_spec_decode | DeepSeek-V3

| Configuration | Accuracy | Output throughput |
|-------------|----------------|----------------|
| attn=fa3 | 0.960 | 409.504 token/s |
| attn=flashinfer | 0.970 | 390.942 token/s |

### Commands

#### attn=fa3

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/public/sharing/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 ../gsm8k/bench_sglang.py
```

#### attn=flashinfer

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/public/sharing/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 ../gsm8k/bench_sglang.py
```

## Experiment - gsm8k_spec_decode | Meta-Llama-3.1-8B-Instruct

| Configuration | Accuracy | Output throughput |
|-------------|----------------|----------------|
| attn=fa3 | 0.780 | 2377.701 token/s |
| attn=flashinfer | 0.775 | 2175.146 token/s |

### Commands

#### attn=fa3

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 ../gsm8k/bench_sglang.py
```

#### attn=flashinfer

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 ../gsm8k/bench_sglang.py
```
