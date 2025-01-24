# Measure accuracy

[SGLang PR template](https://github.com/sgl-project/sglang/blob/main/.github/pull_request_template.md) asks the user to provide accuracy as needed. This document is an introduction for new users on how to perform such measurements.
SGLang provides lots of [Benchmarks](https://github.com/sgl-project/sglang/tree/main/benchmark) which provide an ideal starting point for the user to experiment with.

## GSM8K

[GSM8K](https://huggingface.co/datasets/openai/gsm8k) is a common dataset to evaluate mathematical reasoning in LLMs.
SGLang offers a [benchmark](https://github.com/sgl-project/sglang/tree/main/benchmark/gsm8k) on GSM8K.

The prompt we use is simply `"Question: " + lines[i]["question"] + "\nAnswer:"`. By default we will use `few shot prompting` that means before providing the Question we want to evaluate we will provide a number of question/answer pairs to give the model a sense of the expected kind of questions and answers.

Before running the benchmark a server needs to be started in a seperate terminal:
```python
python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-Math-1.5B-Instruct --port 30000 --mem-fraction-static 0.8
```
`mem_fraction_static` is to control the memory consumption of the model. Model is the excellent [Qwen2.5-1.5B version specialiced on math](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct?language=python).

For more commands on the server, please see [here](https://docs.sglang.ai/backend/server_arguments.html).

Once the server is running the benchmark script can be run.

`python3 bench_sglang.py --num-questions 200`

`num_questions=200` means that we take the first 200 questions in the dataset.
Running the script will produce `results.jsonl` which looks like this:
`{"task": "gsm8k", "backend": "srt", "num_gpus": 1, "latency": 10.513, "accuracy": 0.76, "num_requests": 200, "other": {"num_questions": 200, "parallel": 64}}`. From here we can read off the accuracy.
If we want to compare runs with multiple settings we can just rerun the benchmark and a new line will be appended to `results.jsonl`.
The script can be used to evaluate on a different dataset of the form `Question -> Answer with a number at the end`.
For this, place the dataset [in JSONL format like GSM8K](https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl) onto your disk and make `data_path` argument point to it. Further things that can be easily adjusted by modifying the code in the script are the prompt, method of answer extraction etc.
