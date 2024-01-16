from sglang import function, user, model, gen, image, set_default_backend, Gemini


@function
def image_qa(s, image_file1, image_file2, question):
    # s += user(image(image_file1) + image(image_file2) + question)
    s += image(image_file1) + image(image_file2) + question
    s += gen("answer_1", max_tokens=256)

set_default_backend(Gemini("gemini-pro-vision"))

state = image_qa.run(
    image_file1="./images/cat.jpeg",
    image_file2="./images/rat.jpeg",
    question="Describe difference of the 2 images using one sentence.",
    stream=True
)

for out in state.text_iter():
    print(out, end="", flush=True)