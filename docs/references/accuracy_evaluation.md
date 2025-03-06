# Measuring Model Accuracy in SGLang

This guide shows how to evaluate model accuracy using SGLang's [built-in benchmarks](https://github.com/sgl-project/sglang/tree/b045841baeff37a5601fcde23fa98bd09d942c36/benchmark). Please include accuracy on crucial benchmarks in your PR if you make modifications on the model side, like the kernel and model architecture.

## Benchmarking Model Accuracy

This is a reference workflow for the [MMLU benchmark](https://github.com/sgl-project/sglang/tree/main/benchmark/mmlu). For more details or other benchmarks, please refer to the README in each specific benchmark folder under [sglang/benchmark](https://github.com/sgl-project/sglang/tree/b045841baeff37a5601fcde23fa98bd09d942c36/benchmark).

```bash
# Step 1: Download the dataset
bash download_data.sh

# Step 2: Launch the server
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-Math-1.5B-Instruct \  # Model selection
  --port 30000 \  # Network configuration
  --mem-fraction-static 0.8  # Memory optimization

# Step 3: Run the benchmark script
python3 bench_sglang.py --nsub 10  # Test 10 subjects

# Step 4: Extract the accuracy
cat result.jsonl | grep -oP '"accuracy": \K\d+\.\d+'
```

## Customizing Benchmark Scripts

Some benchmark implementations may differ from ours, causing accuracy discrepancies. To match [[Qwen2.5-Math]](https://github.com/QwenLM/Qwen2.5-Math)'s reported 76.8% GSM8K accuracy, customization is required.

```python
# The GSM8K benchmark script includes few shot examples for evaluation by default.
# Here we exclude them.
for i in range(len(lines[num_shots:num_questions])):
    questions.append(get_one_example(lines, i, False))
    labels.append(get_answer_value(lines[i]["answer"]))
```

```python
@sgl.function
def few_shot_gsm8k(s, question):
    # System prompt given in https://github.com/QwenLM/Qwen2.5-Math
    s += sgl.system("Please reason step by step, and put your final answer within \\boxed{}.") # Include system prompt
    s += few_shot_examples + question
    # Stopwords given in evaluation/math_eval.py of the Qwen2.5-Math repo
    s += sgl.gen(
        "answer", max_tokens=2048, stop=["Question", "Assistant:", "</s>", "<|im_end|>", "<|endoftext|>"]
    )
```

These adjustments should return the desired accuracy.

## Extending Evaluation Capabilities

1. **Contribute New Benchmarks**
   * Follow our [contribution guidelines](../references/contribution_guide.md) to add new test scripts
2. **Request Implementations**
   * Feel free to open an issue describing your evaluation needs
3. **Use Alternative Tools**
   * [OpenCompass](https://opencompass.org.cn)
   * [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
