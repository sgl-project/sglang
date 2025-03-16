import base64
import hashlib
import json
import pickle
import zlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

    SYSTEM_MESSAGE_GEMINI = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. Do NOT use system calls like `exit` in the generated program. Ensure that the first code block contains the solution."

    SYSTEM_MESSAGE_GEMINITHINK = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

    SYSTEM_MESSAGE_DEEPSEEK = f"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you answer questions related to computer science."

    SYSTEM_MESSAGE_CODEQWEN = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
    )

    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."


def load_code_generation_dataset(
    release_version="release_v5",
) -> list[CodeGenerationProblem]:
    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag=release_version,
        trust_remote_code=True,
    )
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset


def get_qwen_question_template_answer(question: CodeGenerationProblem):
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question: {question.question_content}\n\n"
    if question.starter_code:
        prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
    return prompt


def get_qwen_reasoning_question_template_answer(question: CodeGenerationProblem):
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    prompt += f"Question: {question.question_content}\n\n"
    if question.starter_code:
        prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
    return prompt


def calculate_string_md5(input_string: str):
    md5 = hashlib.md5()
    md5.update(input_string.encode("utf-8"))
    return md5.hexdigest()


if __name__ == "__main__":

    output_livecodebench_v5_tests_dir = (
        "/home/data/public/data/eval/code/livecodebench_v5_tests"
    )
    output_livecodebench_v5_data_path = (
        "/home/data/public/data/eval/code/livecodebench_v5.jsonl"
    )
    Path(output_livecodebench_v5_tests_dir).mkdir(parents=True, exist_ok=True)
    Path(output_livecodebench_v5_data_path).parent.mkdir(parents=True, exist_ok=True)

    dataset = load_code_generation_dataset(release_version="release_v5")
    num_samples = 10

    livecodebench_v5_inputs_outputs = []
    livecodebench_v5_dataset = []

    # template for general language model
    # prompt_template = get_qwen_question_template_answer
    # template for reasoning model
    prompt_template = get_qwen_reasoning_question_template_answer

    for global_id, sample in enumerate(tqdm(dataset)):
        inputs_outputs = sample.get_evaluation_sample()
        livecodebench_v5_dataset.append(
            {
                "global_id": global_id,
                "question_id": sample.question_id,
                "contest_id": sample.contest_id,
                "contest_date": sample.contest_date.isoformat(),
                "prompt": prompt_template(sample),
                "tests": {
                    "fname": f"{global_id}.json",
                    "md5": calculate_string_md5(json.dumps(inputs_outputs)),
                },
                "tags": "coding,en,python,core",
                "task": "livecodebench_v5",
                "source": "livecodebench_v5",
                "beam_size": num_samples,
                # "eval_args": eval_args,
            }
        )
        livecodebench_v5_inputs_outputs.append(inputs_outputs)

        # save test cases
        with open(
            Path(output_livecodebench_v5_tests_dir) / f"{global_id}.json", "w"
        ) as f:
            json.dump(inputs_outputs, f)

    # save dataset
    with open(output_livecodebench_v5_data_path, "w") as f:
        for sample in livecodebench_v5_dataset:
            f.write(json.dumps(sample) + "\n")
