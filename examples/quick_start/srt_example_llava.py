"""
Usage: python3 srt_example_llava.py
"""
import sglang as sgl


@sgl.function
def image_qa(s, image_path, question):
    s += sgl.user(sgl.image(image_path) + question)
    s += sgl.assistant(sgl.gen("answer"))


runtime = sgl.Runtime(model_path="liuhaotian/llava-v1.5-7b",
                      tokenizer_path="llava-hf/llava-1.5-7b-hf")
sgl.set_default_backend(runtime)


# Single
state = image_qa.run(
    image_path="images/cat.jpeg",
    question="What is this?",
    max_new_tokens=64)
print(state["answer"], "\n")


# Batch
states = image_qa.run_batch(
    [
        {"image_path": "images/cat.jpeg", "question":"What is this?"},
        {"image_path": "images/dog.jpeg", "question":"What is this?"},
    ],
    max_new_tokens=64,
)
for s in states:
    print(s["answer"], "\n")


runtime.shutdown()
