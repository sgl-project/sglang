## Experiment - bench_serving | DeepSeek-V3

| Configuration | Output token throughput (tok/s) |
|-------------|----------------|
| attn=fa3,random_input_len=1024,random_output_len=1024 | 2858.33    |
| attn=flashinfer,random_input_len=1024,random_output_len=1024 | 2840.55    |
| attn=fa3,random_input_len=8192,random_output_len=1024 | 1201.67    |
| attn=flashinfer,random_input_len=8192,random_output_len=1024 | 1164.14    |
| attn=fa3,random_input_len=8192,random_output_len=8192 | 1632.74    |
| attn=flashinfer,random_input_len=8192,random_output_len=8192 | 1629.65    |

### Commands

#### attn=fa3,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend fa3 --tp 8 --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 1024 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend flashinfer --tp 8 --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 1024 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend fa3 --tp 8 --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend flashinfer --tp 8 --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server --attention-backend fa3 --tp 8 --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 8192
```

#### attn=flashinfer,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server --attention-backend flashinfer --tp 8 --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 8192
```

## Experiment - bench_serving | Meta-Llama-3.1-8B-Instruct

| Configuration | Output token throughput (tok/s) |
|-------------|----------------|
| attn=fa3,random_input_len=1024,random_output_len=1024 | 7221.69    |
| attn=flashinfer,random_input_len=1024,random_output_len=1024 | 6837.91    |
| attn=fa3,random_input_len=8192,random_output_len=1024 | 2322.57    |
| attn=flashinfer,random_input_len=8192,random_output_len=1024 | 1903.50    |
| attn=fa3,random_input_len=8192,random_output_len=8192 | 2881.56    |
| attn=flashinfer,random_input_len=8192,random_output_len=8192 | 1673.13    |

### Commands

#### attn=fa3,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend fa3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 1024 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend flashinfer --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 1024 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend fa3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server --attention-backend flashinfer --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server --attention-backend fa3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 8192
```

#### attn=flashinfer,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server --attention-backend flashinfer --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 8192
```
## Experiment - bench_serving_spec_decode | DeepSeek-V3

| Configuration | Output token throughput (tok/s) |
|-------------|----------------|
| attn=fa3,random_input_len=1024,random_output_len=1024 | 907.83     |
| attn=flashinfer,random_input_len=1024,random_output_len=1024 | 921.07     |
| attn=fa3,random_input_len=8192,random_output_len=1024 | 716.92     |
| attn=flashinfer,random_input_len=8192,random_output_len=1024 | 709.94     |
| attn=fa3,random_input_len=8192,random_output_len=8192 | 1020.97    |
| attn=flashinfer,random_input_len=8192,random_output_len=8192 | 1024.56    |

### Commands

#### attn=fa3,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/user/bhe/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 1024 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/user/bhe/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 1024 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/user/bhe/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/user/bhe/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/user/bhe/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 8192
```

#### attn=flashinfer,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/user/bhe/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 8192
```

## Experiment - bench_serving_spec_decode | Meta-Llama-3.1-8B-Instruct

| Configuration | FAILED | Output token throughput (tok/s) |
|-------------|----------------|----------------|
| attn=fa3,random_input_len=1024,random_output_len=1024 |  | 4239.73    |
| attn=flashinfer,random_input_len=1024,random_output_len=1024 |  | 4313.89    |
| attn=fa3,random_input_len=8192,random_output_len=1024 | Client failed with code 1 |  |
| attn=flashinfer,random_input_len=8192,random_output_len=1024 | Client failed with code 1 |  |
| attn=fa3,random_input_len=8192,random_output_len=8192 | Client failed with code 1 |  |
| attn=flashinfer,random_input_len=8192,random_output_len=8192 | Client failed with code 1 |  |

### Commands

#### attn=fa3,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --enable-torch-compile --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 1024 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=1024,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --enable-torch-compile --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 1024 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --enable-torch-compile --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 1024
```

#### attn=flashinfer,random_input_len=8192,random_output_len=1024

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --enable-torch-compile --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 1024
```

#### attn=fa3,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --enable-torch-compile --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 8192
```

#### attn=flashinfer,random_input_len=8192,random_output_len=8192

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --enable-torch-compile --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate 16 --random-input-len 8192 --random-output-len 8192
```
## Experiment - gsm8k | DeepSeek-V3

| Configuration | Accuracy | Output throughput |
|-------------|----------------|----------------|
| attn=fa3 | 0.960 | 545.436 token/s |
| attn=flashinfer | 0.985 | 543.785 token/s |

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
| attn=fa3 | 0.765 | 3813.443 token/s |
| attn=flashinfer | 0.770 | 3256.967 token/s |

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
| attn=fa3 | 0.955 | 390.234 token/s |
| attn=flashinfer | 0.965 | 379.337 token/s |

### Commands

#### attn=fa3

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/user/bhe/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 ../gsm8k/bench_sglang.py
```

#### attn=flashinfer

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --tp 8 --speculative-algo EAGLE --model-path /shared/public/elr-models/deepseek-ai/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205 --speculative-draft-model-path /shared/user/bhe/bijiang/models--SGLang--DeepSeek-V3-NextN/snapshots/a3b6ab058b5bb3df2bf48afb81312dafd7ea5e60
python3 ../gsm8k/bench_sglang.py
```

## Experiment - gsm8k_spec_decode | Meta-Llama-3.1-8B-Instruct

| Configuration | Accuracy | Output throughput |
|-------------|----------------|----------------|
| attn=fa3 | 0.775 | 2477.827 token/s |
| attn=flashinfer | 0.775 | 2356.153 token/s |

### Commands

#### attn=fa3

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend fa3 \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --enable-torch-compile --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 ../gsm8k/bench_sglang.py
```

#### attn=flashinfer

```
python3 -m sglang.launch_server \ --speculative-num-steps 3 \ --speculative-eagle-topk 1 \ --speculative-num-draft-tokens 4 \ --attention-backend flashinfer \ --trust-remote-code \ --mem-fraction-static 0.8 \ --cuda-graph-max-bs 2 --dtype float16 --enable-torch-compile --speculative-algo EAGLE3 --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct --speculative-draft-model-path /shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714
python3 ../gsm8k/bench_sglang.py
```
