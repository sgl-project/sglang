import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from tqdm import tqdm

from sglang.test.test_utils import add_common_other_args_and_parse, get_call_generate
from sglang.utils import dump_state_text, read_jsonl

number = 5


def expand_tip(topic, tip, generate):
    s = (
        """Please expand a tip for a topic into a detailed paragraph.

Topic: staying healthy
Tip: Regular Exercise
Paragraph: Incorporate physical activity into your daily routine. This doesn't necessarily mean intense gym workouts; it can be as simple as walking, cycling, or yoga. Regular exercise helps in maintaining a healthy weight, improves cardiovascular health, boosts mental health, and can enhance cognitive function, which is crucial for fields that require intense intellectual engagement.

Topic: building a campfire
Tip: Choose the Right Location
Paragraph: Always build your campfire in a safe spot. This means selecting a location that's away from trees, bushes, and other flammable materials. Ideally, use a fire ring if available. If you're building a fire pit, it should be on bare soil or on a bed of stones, not on grass or near roots which can catch fire underground. Make sure the area above is clear of low-hanging branches.

Topic: writing a blog post
Tip: structure your content effectively
Paragraph: A well-structured post is easier to read and more enjoyable. Start with an engaging introduction that hooks the reader and clearly states the purpose of your post. Use headings and subheadings to break up the text and guide readers through your content. Bullet points and numbered lists can make information more digestible. Ensure each paragraph flows logically into the next, and conclude with a summary or call-to-action that encourages reader engagement.

Topic: """
        + topic
        + "\nTip: "
        + tip
        + "\nParagraph:"
    )
    return generate(s, max_tokens=128, stop=["\n\n"])


def suggest_tips(topic, generate):
    s = "Please act as a helpful assistant. Your job is to provide users with useful tips on a specific topic.\n"
    s += "USER: Give some tips for " + topic + ".\n"
    s += (
        "ASSISTANT: Okay. Here are "
        + str(number)
        + " concise tips, each under 8 words:\n"
    )

    tips = []
    for i in range(1, 1 + number):
        s += f"{i}."
        tip = generate(s, max_tokens=24, stop=[".", "\n"])
        s += tip + ".\n"
        tips.append(tip)

    paragraphs = [expand_tip(topic, tip, generate=generate) for tip in tips]

    for i in range(1, 1 + number):
        s += f"Tip {i}:" + paragraphs[i - 1] + "\n"
    return s


def main(args):
    lines = read_jsonl(args.data_path)[: args.num_questions]
    states = [None] * len(lines)

    # Select backend
    call_generate = partial(get_call_generate(args), temperature=0)

    # Run requests
    tic = time.perf_counter()
    if args.backend != "lmql":

        def get_one_answer(i):
            states[i] = suggest_tips(lines[i]["topic"], call_generate)

        if args.parallel == 1:
            for i in tqdm(range(len(lines))):
                get_one_answer(i)
        else:
            with ThreadPoolExecutor(args.parallel) as executor:
                list(
                    tqdm(
                        executor.map(get_one_answer, list(range(len(lines)))),
                        total=len(lines),
                    )
                )

    else:
        import asyncio

        from lmql_funcs import suggest_tips_async

        async def get_one_answer_async(i):
            states[i] = await suggest_tips_async(lines[i]["topic"], call_generate)

        batches = []
        for i in range(0, len(lines), args.parallel):
            batches.append(list(range(i, min(i + args.parallel, len(lines)))))
        loop = asyncio.get_event_loop()
        for batch in tqdm(batches):
            loop.run_until_complete(
                asyncio.gather(*[get_one_answer_async(i) for i in batch])
            )
    latency = time.perf_counter() - tic

    # Compute accuracy
    print(f"Latency: {latency:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "tip_suggestion",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="topic.jsonl")
    parser.add_argument("--num-questions", type=int, default=100)
    args = add_common_other_args_and_parse(parser)
    main(args)
