"""
Usage: python3 srt_example_llava.py
"""
import sglang as sgl


@sgl.function
def image_qa(s, image_path, question):
    s += sgl.user(sgl.image(image_path) + question)
    s += sgl.assistant(sgl.gen("answer"))


runtime = sgl.Runtime(model_path="/home/ec2-user/Yi-VL-6B-hf/Yi-VL-6B", tokenizer_path="/home/ec2-user/Yi-VL-6B-hf/Yi-VL-6B")
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
