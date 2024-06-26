import argparse
import json
import time
import os

import sglang as sgl
import tqdm
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend
from sglang.utils import read_jsonl, dump_state_text
from PIL import Image

# This function is adapted to handle two questions, simulating a Chain of Thought (CoT) reasoning process.
@sgl.function
def image_qa(s, image_file, question1, question2):
    """
    Enhanced Image QA function supporting Chain of Thought reasoning with two questions.

    Args:
        s: sglang object
        image_file: The name of the image file
        question1: The first question, initiating the CoT reasoning
        question2: The second question, continuing the CoT reasoning
    
    Performs two rounds of Q&A for deeper understanding or reasoning.
    """
    s += sgl.user(sgl.image(image_file) + question1)
    s += sgl.assistant(sgl.gen("answer1", max_tokens=args.max_tokens))  # First round of QA
    s += sgl.user(question2)
    s += sgl.assistant(sgl.gen("answer2", max_tokens=args.max_tokens))  # Second round of QA
    # s += sgl.user(question_3)
    # s += sgl.assistant(sgl.gen("answer", max_tokens=args.max_tokens)) 

def main(args):
    lines = read_jsonl(args.question_file)[:args.num_questions]
    # Adjust the structure to include two questions per image for CoT reasoning
    arguments = [
        {"image_file":
            os.path.abspath(args.image_folder + "/" + l["image"]),
          "question1": l["text1"], "question2": l["text2"]} for l in lines
    ]

    # Process the images and questions
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)
    states = [None] * len(lines)

    # Processing the images and questions with CoT reasoning
    tic = time.time()
    if args.parallel == 1:
        for i in tqdm.tqdm(range(len(lines))):
            image_file = arguments[i]["image_file"]
            question1 = arguments[i]["question1"]
            question2 = arguments[i]["question2"] 
            states[i] = image_qa.run(image_file=image_file, question1=question1, question2=question2, temperature=0)
    else:
        states = image_qa.run_batch(arguments, temperature=0, num_threads=args.parallel, progress_bar=True)
    latency = time.time() - tic

    print(f"Latency: {latency:.3f}")

    # Write results, now including answers to two questions per image
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    print(f"Write output to {args.answer_file}")
    with open(args.answer_file, "w") as fout:
        for i in range(len(lines)):
            value = {"question_id": lines[i]["question_id"], 
                      "prompt1": lines[i]["text1"],
                      "text1": states[i]["answer1"].strip(),
                      "prompt2": lines[i]["text2"], 
                      "text2": states[i]["answer2"].strip(),
                      "model_id": backend.model_info["model_path"],
                      "answer_id": lines[i]["question_id"], "metadata": {}}
            fout.write(json.dumps(value) + '\n')

    # Optionally, log the task and performance metrics
    with open(args.result_file, "a") as fout:
        value = {
            "task": "llava_bench_cot",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "num_requests": len(lines),
            "parallel": args.parallel,
        }
        fout.write(json.dumps(value) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="questions_cot.jsonl")
    parser.add_argument("--answer-file", type=str, default="answers_cot.jsonl")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=768)
    args = add_common_sglang_args_and_parse(parser)
    main(args)