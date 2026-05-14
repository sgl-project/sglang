import argparse
import json
import time

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text, read_jsonl

number = 5


@sgl.function
def expand_tip(s, topic, tip):
    s += (
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
    s += sgl.gen("paragraph", max_tokens=128, stop=["\n\n"], temperature=0)


@sgl.function
def suggest_tips(s, topic):
    s += "Please act as a helpful assistant. Your job is to provide users with useful tips on a specific topic.\n"
    s += "USER: Give some tips for " + topic + ".\n"
    s += (
        "ASSISTANT: Okay. Here are "
        + str(number)
        + " concise tips, each under 8 words:\n"
    )

    paragraphs = []
    for i in range(1, 1 + number):
        s += f"{i}." + sgl.gen(f"tip_{i}", max_tokens=24, stop=[".", "\n"]) + ".\n"
        paragraphs.append(expand_tip(topic=topic, tip=s[f"tip_{i}"]))

    for i in range(1, 1 + number):
        s += f"Tip {i}:" + paragraphs[i - 1]["paragraph"] + "\n"


def main(args):
    lines = read_jsonl(args.data_path)[: args.num_questions]
    arguments = [{"topic": l["topic"]} for l in lines]

    # Select backend
    sgl.set_default_backend(select_sglang_backend(args))

    # Run requests
    tic = time.perf_counter()
    states = suggest_tips.run_batch(
        arguments, temperature=0, num_threads=args.parallel, progress_bar=True
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
    args = add_common_sglang_args_and_parse(parser)
    main(args)
