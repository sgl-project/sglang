import argparse
import json
import time

import numpy as np
import sglang as sgl
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend
from sglang.utils import read_jsonl, dump_state_text


@sgl.function
def multi_document_qa(s, docs, question):
    s += sgl.user_begin()
    s += "Pleaes answer a question according to given documents.\n"
    s += "Question:" + question + "Documents begin.\n"

    forks = s.fork(len(docs))
    forks += lambda i: docs[i]
    forks.join("concate_and_append")

    s += "\nDocuments end."
    s += ("\n\nBased on the above documents, please answer this question:\n" + question + "\nAnswer in three words or fewer.")
    s += sgl.user_end()
    s += sgl.assistant(sgl.gen("answer", max_tokens=16))


def main(args):
    lines = read_jsonl(args.data_path)
    l = lines[0]
    arguments = []
    labels = []
    for i in range(len(l["questions"][:args.num_questions])):
        arguments.append({
            "docs": l["documents"][:10],
            "question": l["questions"][i],
        })
        labels.append(l["answers"][i])

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Run requests
    tic = time.time()
    states = multi_document_qa.run_batch(
        arguments, temperature=0, num_threads=args.parallel)
    latency = time.time() - tic

    # Compute accuracy 
    print([s["answer"] for s in states])
    correct = 0
    for s, label in zip(states, labels):
        answer = s["answer"].lower()
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
            }
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="questions.jsonl")
    parser.add_argument("--num-questions", type=int, default=100)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
