# How to Measure Accuracy in SGLang

When contributing to SGLang, the [PR template](https://github.com/sgl-project/sglang/blob/main/.github/pull_request_template.md) requests contributors to report accuracy metrics when relevant. This guide demonstrates how to measure model accuracy using built-in benchmarks.

## Example: Evaluate on GSM8K

The [GSM8K mathematical reasoning dataset](https://huggingface.co/datasets/openai/gsm8k) is commonly used for LLM evaluation. SGLang provides a [benchmark script](https://github.com/sgl-project/sglang/tree/main/benchmark/gsm8k) for this task.

**Key Implementation Details**:

* Few-shot Prompting: Prepends N example Q/A pairs (configurable via `--num-shots`)

* Prompt Structure: Combines few-shot examples with current question using `Question: {text}\nAnswer:` format

* Answer Extraction:

    * Uses regex to find last numeric value in model response

    * Handles invalid outputs with fallback value (-9999999)

* Batch Processing: Leverages SGLang's `run_batch` for [continous batching](https://docs.sglang.ai/frontend/frontend.html#batching)

* Metric Tracking: Calculates accuracy, invalid response rate, and token throughput

The implementation remains model-agnostic - simply modify the `--model-path` server argument to test different models.

### Step 1: Launch Server

Launch server with desired [server arguments](https://docs.sglang.ai/backend/server_arguments.html)

```python
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-Math-1.5B-Instruct \
  --port 30000 \
  --mem-fraction-static 0.8
```

### Step 2: Run benchmark

```python
python3 bench_sglang.py --num-questions 200
```

This evaluates the first 200 questions from the dataset. Key arguments:

* `--num-questions`: Number of questions to evaluate

* `--data-path`: Custom dataset path in the format of the [default](https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl) which is used when no path is provided.

* `--num-shots`: Number of few shots to include into each prompt.

### Step 3: Read off accuracy

After running the benchmark a `result.jsonl` file containing the accuracy is created. Read it off from here.

## Example: Evaluate on MMLU

[MMLU](https://arxiv.org/pdf/2009.03300) is a common dataset to benchmark model abilities in a multiple choice setting. SGLang provides a [benchmark script](https://github.com/sgl-project/sglang/tree/main/benchmark/mmlu) for this task.
The implementation is similar to above. Contrary to GSM8K in MMLU we evaluate the models ability to reply with the letter corresponding to the correct answer.

### Step 1: Download Dataset

Execute the following code to download the dataset.

```bash
bash download_data.sh
```

### Step 2: Launch Server

Same as above.

```python
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-Math-1.5B-Instruct \
  --port 30000 \
  --mem-fraction-static 0.8
```

### Step 3: Run Benchmark

Similar to above.

```bash
python3 bench_sglang.py --nsub 10
```

This evaluates the model on 10 different subjects from the dataset. Key arguments:

* `--ntrain`: Number of few shot examples
* `--nsub`: Number of subjects to evaluate the model on

### Step 4: Read off accuracy

During script execution we get a breakdown on subject wise accuracy.
After running the benchmark a `result.jsonl` file containing the accuracy is created. Read it off from here.
