# Measuring Model Accuracy in SGLang

This guide shows how to evaluate model accuracy using SGLang's built-in benchmarks.

## Evalutating model accuracy with SGLang

SGLang provides many [benchmark scripts](https://github.com/sgl-project/sglang/tree/b045841baeff37a5601fcde23fa98bd09d942c36/benchmark) which can be used to evaluate a models accuracy.

Below we describe the workflow of evaluation with [MMLU benchmark](https://github.com/sgl-project/sglang/tree/main/benchmark/mmlu) as an example but it is very similar for every benchmark script.

### Step 1

Download the data

```bash
bash download_data.sh
```

### Step 2

Launch the server

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-Math-1.5B-Instruct \ # Define model
  --port 30000 \ # Define port
  --mem-fraction-static 0.8 # Avoid memory error
```

### Step 3

Run the benchmark script

```bash
python3 bench_sglang.py --nsub 10
```

This will produce a `result.jsonl` file with the key metrics.

### Step 4

Use `cat result.jsonl | grep -oP '"accuracy": \K\d+\.\d+` to extract the accuracy.

## Key implementation notes

Find a comparison of key parts of the implementation of [GSM8K](https://github.com/sgl-project/sglang/tree/main/benchmark/gsm8k) and [MMLU](https://github.com/sgl-project/sglang/tree/main/benchmark/mmlu) as example benchmarks below.

```python
# GSM8K Evaluation Script
# Prompt construction with few-shot examples
def build_prompt(question):
    # Format: [Few-shot Q/A pairs] + Current Question
    return few_shot_examples + f"Question: {question}\nAnswer:"

# Answer extraction logic
def parse_answer(response):
    # Extract last numeric value from response
    numbers = re.findall(r"\d+", response)
    return int(numbers[-1]) if numbers else -9999999  # Error flag

# Batch processing core
responses = sgl.run_batch(prompts)  # Efficient continuous batching
```

```python
# MMLU Evaluation Script
# Multiple-choice prompt template
def build_mmlu_prompt(question, choices):
    # Format: [Examples] + Current Question + Options
    return f"{examples}{question}\nOptions: {choices}\nAnswer:"

# Choice extraction
def parse_choice(response):
    # Get last alphabetical character
    return response.strip()[-1]  # A/B/C/D detection

# Batch processing core
states = few_shot_mmlu.run_batch(
        ...
        max_new_tokens=1, # Only one letter response
        ...
      )
```

We see that the core implementation is very similar. It only differs in how we define the prompt and extract the response. Other benchmarks can be analyzed similary.

## Customizing Benchmark Scripts

To match [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math)'s reported 76.8% GSM8K accuracy, customization is needed.

```python
for i in range(len(lines[num_shots:num_questions])): # Exclude the few shot examples for evaluation
    questions.append(get_one_example(lines, i, False))
    labels.append(get_answer_value(lines[i]["answer"]))
```

```python
@sgl.function
def few_shot_gsm8k(s, question):
    s += sgl.system("Please reason step by step, and put your final answer within \\boxed{}.") # Include system prompt
    s += few_shot_examples + question
    s += sgl.gen(
        "answer", max_tokens=2048, stop=["Question", "Assistant:", "</s>", "<|im_end|>", "<|endoftext|>"] # Adjust stopwords
    )
```

These adjustments give us the us the reported accuracy.

## Extending Evaluation Capabilities

1. **Contribute New Benchmarks**
   * Follow our [contribution guidelines](https://docs.sglang.ai/references/contribution_guide.html) to add new test scripts
2. **Request Implementations**
   * Feel free to open an issue describing your evaluation needs
3. **Use Alternative Tools**
   * [OpenCompass](https://opencompass.org.cn)
   * [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
