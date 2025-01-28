# Measuring Model Accuracy in SGLang

This guide shows how to evaluate model accuracy using SGLang's [built-in benchmarks](https://github.com/sgl-project/sglang/tree/b045841baeff37a5601fcde23fa98bd09d942c36/benchmark).

## Evalutating model accuracy with SGLang

This is a reference workflow for the [MMLU benchmark](). For more details or other benchmarks, please refer to the README in each specific benchmark folder under [sglang/benchmark](https://github.com/sgl-project/sglang/tree/b045841baeff37a5601fcde23fa98bd09d942c36/benchmark).

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

## Benchmark-Specific Implementation Details

Most benchmarks are similar with task-specific differences. Below, We compare [GSM8K](https://github.com/sgl-project/sglang/tree/main/benchmark/gsm8k) and [MMLU](https://github.com/sgl-project/sglang/tree/main/benchmark/mmlu).

```python
# GSM8K Evaluation Script

# Build few-shot prompt with examples
def get_one_example(lines, i, include_answer):
    # Basic Q&A format
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret

def get_few_shot_examples(lines, k):
    # Include k annotated examples
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret

# Create test dataset
for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))

# Assemble full prompt
@sgl.function
def few_shot_gsm8k(s, question):
    s += few_shot_examples + question
    s += sgl.gen(
        "answer", max_tokens=512, stop=["Question", "Assistant:", "<|separator|>"]
    )

# Extract numerical answer
def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID

# Run batch inference
states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=0,
        num_threads=args.parallel,
        progress_bar=True,
    )
```

```python
# MMLU Evaluation Script

# Format multiple-choice question
def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]  # Question text
    k = df.shape[1] - 2  # Number of options
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    # Create subject-specific header
    prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

# Assemble full prompt
@sgl.function
def few_shot_mmlu(s, examples, question):
    s += examples + question + sgl.gen("answer")

# Batch inference with letter prediction
states = few_shot_mmlu.run_batch(
    arguments,
    temperature=0,
    max_new_tokens=1,  # Generate only one token
    backend=backend,
    num_threads=args.parallel,
    progress_bar=True,
)

# Extract predicted choice
preds = [
    s["answer"].strip()[0] if len(s["answer"].strip()) > 0 else ""
    for s in states
]
```

The core implementation is largely similar, differing mainly in prompt definition and response extraction. Other benchmarks can be analyzed in a similar way.

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

These adjustments give us the us the reported accuracy.

## Extending Evaluation Capabilities

1. **Contribute New Benchmarks**
   * Follow our [contribution guidelines](https://docs.sglang.ai/references/contribution_guide.html) to add new test scripts
2. **Request Implementations**
   * Feel free to open an issue describing your evaluation needs
3. **Use Alternative Tools**
   * [OpenCompass](https://opencompass.org.cn)
   * [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
