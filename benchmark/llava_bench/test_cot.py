import json
import os
import sglang as sgl
from sglang.test.test_utils import select_sglang_backend, add_common_sglang_args_and_parse
import tqmd
import time
import argparse
from sglang.utils import read_jsonl, dump_state_text

@sgl.function
def image_qa(s, image_file, question1, question2 ) -> sgl.SglangObject:
    """
    图像问答函数

    Args:
        s: sglang对象
        image_file: 图像文件名
        question_1: 问题1
        question_2: 问题2
        question_3: 问题3
    
    Returns:
        s: sglang对象
    """
    s += sgl.user(sgl.image(image_file) + question1)
    s += sgl.assistant(sgl.gen("answer1", max_tokens=args.max_tokens))  
    s += sgl.user(question2)
    s += sgl.assistant(sgl.gen("answer2", max_tokens=args.max_tokens))  
    # s += sgl.user(question_3)
    # s += sgl.assistant(sgl.gen("answer", max_tokens=args.max_tokens))  

def main() -> None:
    arguments = [
        {"image_file":
            os.path.abspath(args.image_folder + "/" + l["image"]),
          "text1": l["text1"], "text2": l["text2"]} for l in lines
    ]

    # print(arguments)

    states = [None] * len(lines)

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Run requests
    tic = time.time()
    if args.parallel == 1:
        for i in tqdm.tqdm(range(len(lines))):
            image_file = arguments[i]["image_file"]
            question = arguments[i]["question"]
            ret = image_qa.run(
                image_file=image_file,
                question=question,
                temperature=0)
            states[i] = ret
    else:
        states = image_qa.run_batch(
            arguments,
            temperature=0,
            num_threads=args.parallel,
            progress_bar=True)
    latency = time.time() - tic

    print(f"Latency: {latency:.3f}")

# Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    print(f"Write output to {args.answer_file}")
    with open(args.answer_file, "w") as fout:
        for i in range(len(lines)):
            value = {
                "question_id": lines[i]["question_id"],
                "prompt": lines[i]["text"],
                "text": states[i]["answer"].strip(),
                "model_id": backend.model_info["model_path"],
                "answer_id": i,
                "metadata": {},
            }
            fout.write(json.dumps(value) + "\n")

    with open(args.result_file, "a") as fout:
        value = {
            "task": "llava_bench",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "num_requests": len(lines),
            "parallel": args.parallel,
        }
        fout.write(json.dumps(value) + "\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="/home/users/ntu/chih0001/scratch/VLM/sglang_fork/benchmark/llava_bench/questions_cot.jsonl")
    parser.add_argument("--answer-file", type=str, default="/home/users/ntu/chih0001/scratch/VLM/sglang_fork/benchmark/llava_bench/test_answers.jsonl")
    # parser.add_argument("--image-folder", type=str, default="./images")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=768)
    args = add_common_sglang_args_and_parse(parser)
    main(args)