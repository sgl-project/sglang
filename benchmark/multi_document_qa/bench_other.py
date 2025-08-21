import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from tqdm import tqdm

from sglang.test.test_utils import add_common_other_args_and_parse, get_call_generate
from sglang.utils import dump_state_text, read_jsonl

USER_PREFIX = "[INST] "
USER_SUFFIX = " [/INST]"
ASSISTANT_PREFIX = ""
ASSISTANT_SUFFIX = " </s><s>"


def multi_document_qa(docs, question, generate):
    s = USER_PREFIX
    s += "Please answer a question according to given documents.\n"
    s += "Question:" + question + "Documents begin.\n"

    s += "".join(docs)

    s += "\nDocuments end."
    s += (
        "\n\nBased on the above documents, please answer this question:\n"
        + question
        + "\nAnswer in three words or fewer."
    )
    s += USER_SUFFIX
    s += ASSISTANT_PREFIX
    answer = generate(s, max_tokens=16, stop=None)
    return answer


def main(args):
    lines = read_jsonl(args.data_path)
    l = lines[0]
    arguments = []
    labels = []

    num_docs = 10
    if args.backend == "guidance":
        num_docs = 7  # due to OOM

    for i in range(len(l["questions"][: args.num_questions])):
        arguments.append(
            {
                "docs": l["documents"][:num_docs],
                "question": l["questions"][i],
            }
        )
        labels.append(l["answers"][i])
    states = [None] * len(arguments)

    # Select backend
    call_generate = partial(get_call_generate(args), temperature=0)

    # Run requests
    def get_one_answer(i):
        states[i] = multi_document_qa(generate=call_generate, **arguments[i])

    tic = time.perf_counter()
    if args.parallel == 1:
        for i in tqdm(range(len(labels))):
            get_one_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            list(
                tqdm(
                    executor.map(get_one_answer, list(range(len(labels)))),
                    total=len(labels),
                )
            )

    latency = time.perf_counter() - tic

    # Compute accuracy
    print(states)
    correct = 0
    for s, label in zip(states, labels):
        answer = s.lower()
        if all(x in answer for x in label.lower().split(" ")):
            correct += 1
    accuracy = correct / len(labels)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Latency: {latency:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "multi_document_qa",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "num_requests": args.num_questions,
            "accuracy": accuracy,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="questions.jsonl")
    parser.add_argument("--num-questions", type=int, default=100)
    args = add_common_other_args_and_parse(parser)
    main(args)
