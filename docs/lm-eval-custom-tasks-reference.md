# Custom Task Registration for lm-evaluation-harness v0.4.9.2

## Quick Reference: Custom Task Registration

### 1. Register Custom Task Directory

**Three methods (in order of preference):**

#### Method A: CLI Flag (Recommended for sglang)
```bash
lm-eval --tasks my_task --include_path /path/to/custom/tasks
```

#### Method B: Environment Variable
```bash
export LMEVAL_INCLUDE_PATH=/path/to/custom/tasks
lm-eval --tasks my_task
```

#### Method C: Programmatic (TaskManager)
```python
from lm_eval.tasks import TaskManager
import lm_eval

task_manager = TaskManager(include_path="/path/to/custom/tasks")
results = lm_eval.simple_evaluate(
    model=lm,
    tasks=["my_task"],
    task_manager=task_manager,
)
```

**Source:** [lm-eval/tasks/__init__.py:26-39](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.2/lm_eval/tasks/__init__.py#L26-L39)

---

## Minimum Viable Task YAML Schema

```yaml
# Required fields
task: my_task_name
dataset_path: huggingface/dataset_name  # or "json", "arrow", etc.
output_type: generate_until              # or "loglikelihood", "multiple_choice"
doc_to_text: "{{field_name}}"           # Jinja2 template or function
doc_to_target: "{{target_field}}"       # Jinja2 template or function

# Scoring
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true

# Optional but common
num_fewshot: 0
test_split: test
generation_kwargs:
  until: ["\n"]
  max_gen_toks: 512
  do_sample: false

metadata:
  version: 1.0
```

**Full schema reference:** [docs/task_guide.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.2/docs/task_guide.md#configurations)

---

## Custom Scoring via `process_results`

### Signature
```python
def process_results(doc: dict, results: list[str]) -> dict[str, Any]:
    """
    Args:
        doc: Single document from dataset
        results: List of model outputs (usually len=1 for generate_until)
    
    Returns:
        Dictionary mapping metric_name -> metric_value
        Can return multiple metrics at once.
    """
    response = results[0]
    # ... custom scoring logic ...
    return {
        "metric1": score1,
        "metric2": score2,
    }
```

### YAML Integration
```yaml
process_results: !function utils.process_results
metric_list:
  - metric: metric1
    aggregation: mean
    higher_is_better: true
  - metric: metric2
    aggregation: mean
    higher_is_better: true
```

**File location:** `lm_eval/tasks/<task_name>/utils.py`

**Example:** [IFEval process_results](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.2/lm_eval/tasks/ifeval/utils.py#L111-L128)

---

## Multi-Metric Aggregation

### Pattern: Prompt-Level + Instruction-Level (IFEval)

```python
def process_results(doc, results):
    response = results[0]
    
    # Compute per-instruction compliance
    strict_results = test_instruction_following_strict(doc, response)
    loose_results = test_instruction_following_loose(doc, response)
    
    return {
        "prompt_level_strict_acc": strict_results.follow_all_instructions,
        "inst_level_strict_acc": strict_results.follow_instruction_list,
        "prompt_level_loose_acc": loose_results.follow_all_instructions,
        "inst_level_loose_acc": loose_results.follow_instruction_list,
    }

def agg_inst_level_acc(items):
    """Flatten list of lists and compute mean."""
    flat_items = [item for sublist in items for item in sublist]
    return sum(flat_items) / len(flat_items)
```

### YAML Configuration
```yaml
metric_list:
  - metric: prompt_level_strict_acc
    aggregation: mean
    higher_is_better: true
  - metric: inst_level_strict_acc
    aggregation: !function utils.agg_inst_level_acc
    higher_is_better: true
  - metric: prompt_level_loose_acc
    aggregation: mean
    higher_is_better: true
  - metric: inst_level_loose_acc
    aggregation: !function utils.agg_inst_level_acc
    higher_is_better: true
```

**Source:** [IFEval utils.py:111-134](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.2/lm_eval/tasks/ifeval/utils.py#L111-L134)

---

## Task Archetype 1: Multiple Choice (MCQ) — SuperGPQA Pattern

### Directory Structure
```
lm_eval/tasks/gpqa/
├── cot_zeroshot/
│   ├── _gpqa_cot_zeroshot_yaml    # Base config (included by variants)
│   ├── gpqa_diamond_cot_zeroshot.yaml
│   ├── gpqa_main_cot_zeroshot.yaml
│   └── utils.py
└── README.md
```

### Base YAML (`_gpqa_cot_zeroshot_yaml`)
```yaml
dataset_path: Idavidrein/gpqa
tag: gpqa
output_type: generate_until
process_docs: !function utils.process_docs
training_split: train
validation_split: train
test_split: null

# Prompt format with choices
doc_to_text: |
  What is the correct answer to this question:{{Question}}
  Choices:
  (A) {{choice1}}
  (B) {{choice2}}
  (C) {{choice3}}
  (D) {{choice4}}
  Let's think step by step: 

doc_to_target: answer

# Multiple filter pipelines for different extraction strategies
filter_list:
  - name: "strict-match"
    filter:
      - function: "regex"
        regex_pattern: "(?<=The answer is )(.*)(?=.)"
      - function: "take_first"
  - name: "flexible-extract"
    filter:
      - function: "multi_choice_regex"
        group_select: -1
        ignore_case: true
        ignore_punctuation: true
        regex_pattern: "(\\([A-Z]\\))"
      - function: "take_first"

generation_kwargs:
  until: ["</s>"]
  do_sample: false
  temperature: 0.0

num_fewshot: 0
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: true

metadata:
  version: 1.0
```

### Variant YAML (`gpqa_diamond_cot_zeroshot.yaml`)
```yaml
dataset_name: gpqa_diamond
include: _gpqa_cot_zeroshot_yaml
task: gpqa_diamond_cot_zeroshot
```

### Key Concepts
- **Filter pipelines**: Extract answer from free-form text (regex, multi_choice_regex)
- **Multiple variants**: Same base config, different dataset_name
- **Metric options**: `ignore_case`, `ignore_punctuation` for flexible matching

**Source:** [GPQA cot_zeroshot](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.2/lm_eval/tasks/gpqa/cot_zeroshot/)

---

## Task Archetype 2: Code Execution (LiveCodeBench Pattern)

### YAML Configuration
```yaml
task: humaneval
dataset_path: openai/openai_humaneval
unsafe_code: true
output_type: generate_until
test_split: test

doc_to_text: "{{prompt}}"
doc_to_target: "{{test}}\ncheck({{entry_point}})"

# Custom metric function
metric_list:
  - metric: !function utils.pass_at_k
    aggregation: mean
    higher_is_better: true
    k: [1]

generation_kwargs:
  until:
    - "\nclass"
    - "\ndef"
    - "\n#"
    - "\nif"
    - "\nprint"
  max_gen_toks: 1024
  do_sample: false

# Custom filter to build complete code
filter_list:
  - name: "create_test"
    filter:
      - function: "custom"
        filter_fn: !function utils.build_predictions

repeats: 1
num_fewshot: 0

metadata:
  version: 1.0
```

### Custom Metric: `pass_at_k`
```python
import evaluate as hf_evaluate

# Load the code_eval metric (handles execution + timeout)
compute_ = hf_evaluate.load("code_eval")

def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    """
    Args:
        references: List of test cases (e.g., ["assert add(2,3)==5"])
        predictions: List of [code_samples] per problem
        k: List of k values to compute pass@k for
    
    Returns:
        pass@k score (0.0-1.0)
    """
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]

def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Prepend prompt to each response to form complete code."""
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]
```

### Key Concepts
- **`unsafe_code: true`**: Enables code execution (requires `HF_ALLOW_CODE_EVAL=1` env var)
- **`code_eval` metric**: From HuggingFace, handles sandboxed execution + timeouts
- **Custom filter**: Reconstructs complete code (prompt + generation)
- **`repeats`**: For pass@k, set to desired k value (e.g., 64 for pass@64)

**Source:** [HumanEval YAML](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.2/lm_eval/tasks/humaneval/humaneval.yaml) and [utils.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.2/lm_eval/tasks/humaneval/utils.py)

---

## Task Archetype 3: Constraint Verifier (IFBench Pattern)

### YAML Configuration
```yaml
task: ifeval
dataset_path: google/IFEval
dataset_name: null
output_type: generate_until
test_split: train
num_fewshot: 0

doc_to_text: prompt
doc_to_target: 0

generation_kwargs:
  until: []
  do_sample: false
  temperature: 0.0
  max_gen_toks: 1280

# Custom process_results returns multiple metrics
process_results: !function utils.process_results

# Four metrics: 2 strictness levels × 2 aggregation levels
metric_list:
  - metric: prompt_level_strict_acc
    aggregation: mean
    higher_is_better: true
  - metric: inst_level_strict_acc
    aggregation: !function utils.agg_inst_level_acc
    higher_is_better: true
  - metric: prompt_level_loose_acc
    aggregation: mean
    higher_is_better: true
  - metric: inst_level_loose_acc
    aggregation: !function utils.agg_inst_level_acc
    higher_is_better: true

metadata:
  version: 4.0
```

### Constraint Verifier Pattern
```python
# utils.py

@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]

@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]

def test_instruction_following_strict(inp, response):
    """Check if response follows ALL instructions (strict)."""
    instruction_list = inp.instruction_id_list
    is_following_list = []
    
    for index, instruction_id in enumerate(instruction_list):
        # Load constraint class from registry
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        
        # Build constraint description with kwargs
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        
        # Check if response satisfies constraint
        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)
    
    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )

def test_instruction_following_loose(inp, response):
    """Check if response follows ALL instructions (loose, with variants)."""
    # Try multiple response variants (remove first/last line, remove asterisks, etc.)
    all_responses = [
        response,
        response.replace("*", ""),
        "\n".join(response.split("\n")[1:]).strip(),
        # ... more variants ...
    ]
    
    instruction_list = inp.instruction_id_list
    is_following_list = []
    
    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        
        # Check if ANY variant satisfies constraint
        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break
        is_following_list.append(is_following)
    
    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )

def process_results(doc, results):
    """Return 4 metrics: prompt-level & instruction-level × strict & loose."""
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )
    response = results[0]
    
    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)
    
    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }

def agg_inst_level_acc(items):
    """Flatten list of lists (per-instruction) and compute mean."""
    flat_items = [item for sublist in items for item in sublist]
    return sum(flat_items) / len(flat_items)
```

### Constraint Registry Pattern
```python
# instructions_registry.py

INSTRUCTION_DICT = {
    "instruction_id_1": ConstraintClass1,
    "instruction_id_2": ConstraintClass2,
    # ... 25 constraint types for IFEval ...
}

# instructions.py

class ConstraintBase:
    def __init__(self, instruction_id: str):
        self.instruction_id = instruction_id
    
    def build_description(self, **kwargs):
        """Build constraint description from kwargs."""
        pass
    
    def check_following(self, response: str) -> bool:
        """Return True if response satisfies constraint."""
        pass
    
    def get_instruction_args(self) -> dict:
        """Return expected kwargs for this constraint."""
        pass
```

### Key Concepts
- **Constraint registry**: Map instruction IDs to verifier classes
- **Strict vs. loose**: Strict checks exact response; loose tries variants
- **Prompt-level**: All instructions followed (AND)
- **Instruction-level**: Per-instruction compliance (list of bools)
- **Custom aggregation**: Flatten instruction-level lists before averaging

**Source:** [IFEval utils.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.2/lm_eval/tasks/ifeval/utils.py) and [instructions.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.2/lm_eval/tasks/ifeval/instructions.py)

---

## SGLang Integration: Passing `--include_path`

### Current Status (sglang v0.4.9.2)
**`--include_path` is NOT currently exposed in sglang.bench_eval CLI.**

### Workaround 1: Environment Variable (Recommended)
```bash
export LMEVAL_INCLUDE_PATH=/path/to/custom/tasks
python -m sglang.bench_eval --task my_task --base-url http://127.0.0.1:30000 ...
```

### Workaround 2: Monkey-Patch (Temporary)
```python
# Before calling run_bench_eval
import os
os.environ["LMEVAL_INCLUDE_PATH"] = "/path/to/custom/tasks"

from sglang.bench_eval import run_bench_eval
results = run_bench_eval(task="my_task", ...)
```

### Workaround 3: Patch sglang (Permanent)
Add to `python/sglang/bench_eval.py`:

```python
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(...)
    # ... existing args ...
    p.add_argument("--include-path", default=None,
                   help="Additional path to include custom tasks.")
    return p

def run_bench_eval(
    *,
    task: str,
    # ... existing params ...
    include_path: Optional[str] = None,
    # ... rest of params ...
) -> Dict[str, Any]:
    import lm_eval
    from lm_eval.tasks import TaskManager
    
    # ... existing setup ...
    
    task_manager = None
    if include_path:
        task_manager = TaskManager(include_path=include_path)
    
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[task],
        # ... existing kwargs ...
        task_manager=task_manager,
    )
    
    # ... rest of function ...

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    report = run_bench_eval(
        # ... existing args ...
        include_path=args.include_path,
    )
    # ... rest of main ...
```

**Source:** [sglang bench_eval.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_eval.py) and [lm-eval evaluator.py:73](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.2/lm_eval/evaluator.py#L73)

---

## Complete Example: Custom Task YAML

### Directory Structure
```
custom_tasks/
├── livecode_v6/
│   ├── livecode_v6.yaml
│   └── utils.py
├── supergpqa/
│   ├── supergpqa.yaml
│   └── utils.py
└── ifbench/
    ├── ifbench.yaml
    ├── utils.py
    └── constraints.py
```

### Example: `custom_tasks/livecode_v6/livecode_v6.yaml`
```yaml
task: livecode_v6
dataset_path: livebench/livecode_v6
output_type: generate_until
test_split: test

doc_to_text: "{{problem_statement}}\n\ndef solution():\n    "
doc_to_target: "{{solution}}"

process_results: !function utils.process_results

metric_list:
  - metric: pass_at_1
    aggregation: mean
    higher_is_better: true

generation_kwargs:
  until: ["\ndef ", "\nclass ", "\n#"]
  max_gen_toks: 2048
  do_sample: false

num_fewshot: 0

metadata:
  version: 1.0
```

### Example: `custom_tasks/livecode_v6/utils.py`
```python
import subprocess
import tempfile
import os

def process_results(doc, results):
    """Execute generated code and check correctness."""
    response = results[0]
    
    try:
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(doc["problem_statement"] + "\n" + response)
            temp_file = f.name
        
        # Execute with timeout
        result = subprocess.run(
            ["python", temp_file],
            capture_output=True,
            timeout=5,
            text=True
        )
        
        # Check if output matches expected
        is_correct = result.stdout.strip() == doc["expected_output"].strip()
        
        return {"pass_at_1": 1.0 if is_correct else 0.0}
    
    except subprocess.TimeoutExpired:
        return {"pass_at_1": 0.0}
    except Exception as e:
        print(f"Error executing code: {e}")
        return {"pass_at_1": 0.0}
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
```

### Usage
```bash
# Register custom tasks
python -m sglang.bench_eval \
    --task livecode_v6 \
    --base-url http://127.0.0.1:30000 \
    --model qwen-32b \
    --tokenizer Qwen/Qwen2.5-32B-Instruct \
    --include-path /path/to/custom_tasks
```

---

## Debugging Tips

### Enable Debug Logging
```bash
export LMEVAL_LOG_LEVEL=DEBUG
python -m sglang.bench_eval --task my_task ...
```

### Validate YAML
```bash
python -c "
from lm_eval.utils import load_yaml_config
config = load_yaml_config('path/to/task.yaml', mode='simple')
print(config)
"
```

### Test Custom Function
```python
from lm_eval.tasks.my_task import utils
doc = {"field": "value"}
results = ["model output"]
metrics = utils.process_results(doc, results)
print(metrics)
```

### Check Task Registration
```bash
python -m lm_eval --tasks list --include_path /path/to/custom_tasks | grep my_task
```

---

## References

- **New Task Guide:** https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.2/docs/new_task_guide.md
- **Task Configuration Guide:** https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.2/docs/task_guide.md
- **TaskManager API:** https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.2/lm_eval/tasks/__init__.py#L20-L87
- **simple_evaluate API:** https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.2/lm_eval/evaluator.py#L50-L82
- **CLI Reference:** https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.2/lm_eval/__main__.py#L213-L218
