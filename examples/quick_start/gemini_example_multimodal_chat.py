from sglang import function, user, assistant, gen, image, set_default_backend, VertexAI


@function
def image_qa(s, image_file1, image_file2, question):
    s += user(image(image_file1) + image(image_file2) + question)
    s += assistant(gen("answer_1", max_tokens=256))

set_default_backend(VertexAI("gemini-pro-vision"))

state = image_qa.run(
    image_file1="./images/cat.jpeg",
    image_file2="./images/dog.jpeg",
    question="Describe difference of the 2 images in one sentence.",
    stream=True
)

for out in state.text_iter():
    print(out, end="", flush=True)
