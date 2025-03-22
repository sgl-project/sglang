# Evaluation of QwQ

Currently, this repository contains the code to reproduce the following scores.

| Datasets                 | SGLang   | QwQ-32B-official |
|--------------------------|----------|------------------|
| AIME24                   | 79.74    | 79.5             |
| AIME25                   |          | 69.5             |
| LiveCodeBench 2408-2502  | 61.9     | 63.4             |
| IFeval(Strict Prompt Acc)|          | 83.9             |

Kindly remind that, even with `n_samples` set the same as QwQ officially recommended, there's still some deviation in the results. For less deviation in the results please increase `n_samples` to a higher value.

## Download repo
```
git submodule init
git submodule update

```

## Run benchmark
```
pip install -r requirements.txt
python -m sglang_router.launch_server --model-path Qwen/QwQ-32B --dp-size 4 --host=0.0.0.0 --port=30000
```



### Run Inference

After starting the SGLang service, run the inference script to generate responses.

```bash
cd QwQ/eval

mkdir -p output

# aime24 (repeated sample 64 times)
python ./generate_api_answers/infer_multithread.py --input_file "./data/aime24.jsonl" --output_file "./output/aime24_bz64.jsonl"  --base_url "http://127.0.0.1:30000/v1" --model_name "Qwen/QwQ-32B"

# aime25 (repeated sample 64 times)
python ./generate_api_answers/infer_multithread.py --input_file "./data/aime25.jsonl" --output_file "./output/aime25_bz64.jsonl"  --base_url "http://127.0.0.1:30000/v1" --model_name "Qwen/QwQ-32B"

# livebench 2408-2502 (repeated sample 8 times)
python ./generate_api_answers/infer_multithread.py --input_file "./data/livecodebench_v5.jsonl" --output_file "./output/livecodebench_v5_bz8.jsonl"  --base_url "http://127.0.0.1:30000/v1" --model_name "Qwen/QwQ-32B" --n_samples 8

# IFEval
python ./generate_api_answers/infer_multithread.py --input_file "./data/ifeval.jsonl" --output_file "./output/ifeval_bz1.jsonl"  --base_url "http://127.0.0.1:30000/v1" --model_name "Qwen/QwQ-32B" --n_samples 1
```

**Note:** We apply repeated sampling to reduce evaluation variance, but it may take a long time to complete (more than 8 hours depending on your device).

#### Parameter Description

- `--base_url`: Base URL of the SGLang service
- `--model_name`: Must match the model name used in Step 1
- `--n_samples`: Number of samples per prompt
  - AIME24 / AIME 25: Recommended 64 samples
  - LiveCodeBench: Recommended 8 samples
  - IFEval: Recommended 1 sample
- `--input_file`: Input data file path
- `--output_file`: Output result file path, model responses will be stored in the `gen` field
- `--max_workers`: Maximum number of concurrent threads to control inference speed and resource usage

#### Sampling Parameters

We use ``top_p=0.95``, ``temperature=0.6``, ``top_k=40``, ``max_tokens=32768`` for sampling.

#### Resuming Interrupted Inference

If the inference process is interrupted, simply rerun the same command to resume. The script will automatically read the previous output file and process any prompts that haven't completed the required number of samples.

## Scoring

After completing the inference, use the following commands for scoring:

```bash
mkdir -p eval_res

python  ./eval/eval.py --input_path ./output/aime24_bz64.jsonl --cache_path ./eval_res/aime24_bz64.jsonl  --task_name "math_opensource/aime24" > ./eval_res/aime24_bz64_res_result.txt

python  ./eval/eval.py --input_path ./output/aime25_bz64.jsonl --cache_path ./eval_res/aime25_bz64.jsonl  --task_name "math_opensource/aime25" > ./eval_res/aime25_bz64_res_result.txt

# download all test cases
python ./data/process_data.py
# Note: running all code test cases can be very slow (more than 4 hours)
python  ./eval/eval.py --input_path ./output/livecodebench_v5_bz8.jsonl --cache_path ./eval_res/livecodebench_v5_bz8.jsonl  --task_name "livecodebench" > ./eval_res/livecodebench_v5_bz8_res_result.txt

python  ./eval/eval.py --input_path ./output/ifeval_bz1.jsonl --cache_path ./eval_res/ifeval_bz1.jsonl  --task_name "ifeval" > ./eval_res/ifeval_bz1_res_result.txt
```

### Parameter Description

- `--input_path`: Input file path, can directly use the output file from multi-threaded inference or other files with consistent format. Requirements:
  - JSONL format
  - Contains `prompt` and corresponding fields
  - Model responses stored in the `gen` field
- `--cache_path`: Cache directory for storing temporary files during evaluation
- `--task_name`: Evaluation task name, must be one of the following options:
  - `math_opensource/aime24`
  - `math_opensource/aime25`
  - `livecodebench`
  - `ifeval`
