import time
import datasets
import argparse
import json
import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text


@sgl.function
def context_qa(s, refs, question):
    s += sgl.system()
    prompt = "Please answer a question according to the following references.\n"
    for i, refs in enumerate(refs):
        prompt += f"Ref {i}: {refs}\n"
    prompt += "The questions is: " + question + "\n"
    prompt += "Please provide a single-paragraph answer. "
    prompt += "Focus on the provided references and the answer to the question. "
    prompt += 'End your answer paragraph with the word "END"\n'
    s += sgl.user(prompt)
    s += sgl.assistant(sgl.gen("answer", stop="END"))


@sgl.function
def mt_bench(s, question_1, question_2):
    s += sgl.system()
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1"))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument(
        "--mode", type=str, choices=["contextqa", "mtbench"], default="contextqa"
    )
    args = add_common_sglang_args_and_parse(parser)
    backend = select_sglang_backend(args)

    sgl.set_default_backend(backend)
    completion_tokens = 0
    decode_tokens = 0

    if args.mode == "contextqa":
        dataset = datasets.load_dataset(
            "miracl/hagrid", split="dev", trust_remote_code=True
        )
        dataset = dataset.select(range(args.num))
        arguments = [
            {"refs": [q["text"] for q in d["quotes"]], "question": d["query"]}
            for d in dataset
        ]
        tic = time.time()
        states = context_qa.run_batch(
            arguments,
            max_new_tokens=256,
            temperature=0,
            num_threads=args.parallel,
            progress_bar=True,
        )
        toc = time.time()
        for s in states:
            meta = s.get_meta_info("answer")
            completion_tokens += meta["completion_tokens"]
            decode_tokens += meta["decode_tokens"]
    elif args.mode == "mtbench":
        arguments = []
        with open("question.jsonl", "r") as lines:
            for line in lines:
                obj = json.loads(line)
                arguments.append(
                    {"question_1": obj["turns"][0], "question_2": obj["turns"][1]}
                )
        arguments = arguments[: args.num]
        tic = time.time()
        states = mt_bench.run_batch(
            arguments,
            temperature=0,
            max_new_tokens=256,
            num_threads=args.parallel,
            progress_bar=True,
        )
        toc = time.time()
        for s in states:
            meta1 = s.get_meta_info("answer_1")
            meta2 = s.get_meta_info("answer_2")
            completion_tokens += meta1["completion_tokens"] + meta2["completion_tokens"]
            decode_tokens += meta1["decode_tokens"] + meta2["decode_tokens"]
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    print(f"Latency: {toc - tic:.3f}")

    print(f"Completion tokens: {completion_tokens}")
    print(f"Decode tokens: {decode_tokens}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "reference_speculative",
            "backend": args.backend,
            "mode": args.mode,
            "latency": round(toc - tic, 3),
            "num_requests": args.num,
            "completion_tokens": completion_tokens,
            "decode_tokens": decode_tokens,
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    main()
