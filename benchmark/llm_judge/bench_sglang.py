import argparse
import json
import time

import numpy as np
import sglang as sgl
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend
from sglang.utils import read_jsonl, dump_state_text


system_prompt = (
"Please serve as an impartial judge and rigorously evaluate the quality of the following article. Apply the most stringent standards possible, showing no leniency."
)

dimension_prompts = [
"Content: This refers to the essences of the essay. The substance should be well researched, accurate, relevant to the topic and should show a thorough understanding of the subject. The essay should also reflect a clear goal or purpose.",
"Organization and Structure: An essay needs to be properly structured with a clear introduction, body, and conclusion. The essay should flow naturally, with one paragraph leading seamlessly into the next.",
"Argument and Analysis: The argument made in the essay should be logical, coherent and clearly articulated. Each point made should be backed up by solid evidence and thorough analysis.",
"Clarity and Precision: The essay should be written in a clear and concise manner. The points made should be easily understood by the reader. The language used should also be precise and unambiguous.",
"Grammar and Punctuation: Proper use of grammar and punctuation is vital in an academic essay. Errors in grammar and punctuation not only distract the reader but can also negatively impact the meaning and interpretation of the content.",
"Referencing and Citation: An essay should contain proper citations and references for all sources used. This not only prevents accusations of plagiarism but also gives credit to the authors of the works that have contributed to the essay. The citation should adhere to a specific format as required by the academic institution or specified by the professor.",
]


@sgl.function
def multi_dimension_judge(s, article):
    s += system_prompt
    s += "\n```\n" + article + "\n```\n\n"

    forks = s.fork(len(dimension_prompts))
    for i in range(len(dimension_prompts)):
        forks[i] += ("USER: Please judge the quality based on the following metric. " +
                     dimension_prompts[i] + " Please provide a single-paragraph judgement. " +
                     "Focus on the provided metric and do not say other things. " 
                     'End your judgement paragraph with the word "END"\nJUDGE:')
        forks[i] += sgl.gen("judgement", max_tokens=256, stop="END")
    forks.join()

    s += "I will judge the quality based on the following metrics.\n"
    for i in range(len(dimension_prompts)):
        s += dimension_prompts[i].split(":")[0] + ": " + forks[i]["judgement"].strip() + "\n"

    s += "In summary, on a scale of 1 to 10, I would give the article a score of"
    s += sgl.gen("score", max_tokens=2)


def main(args):
    lines = read_jsonl(args.data_path)[:args.num_questions]
    arguments = [{"article": l} for l in lines]

    # Select backend
    backend = select_sglang_backend(args)

    # Run requests
    tic = time.time()
    states = multi_dimension_judge.run_batch(
        arguments, temperature=0, backend=backend, num_threads=args.parallel)
    latency = time.time() - tic

    print(f"Latency: {latency:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "llm_judge",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            }
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="articles.jsonl")
    parser.add_argument("--num-questions", type=int, default=20)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
