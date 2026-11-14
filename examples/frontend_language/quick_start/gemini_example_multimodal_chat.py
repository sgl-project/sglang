"""
Usage:
export GCP_PROJECT_ID=******
python3 gemini_example_multimodal_chat.py
"""

import sglang as sgl


@sgl.function
def image_qa(s, image_file1, image_file2, question):
    s += sgl.user(sgl.image(image_file1) + sgl.image(image_file2) + question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=256))


if __name__ == "__main__":
    sgl.set_default_backend(sgl.VertexAI("gemini-pro-vision"))

    state = image_qa.run(
        image_file1="./images/cat.jpeg",
        image_file2="./images/dog.jpeg",
        question="Describe difference of the two images in one sentence.",
        stream=True,
    )

    for out in state.text_iter("answer"):
        print(out, end="", flush=True)
    print()

    print(state["answer"])
