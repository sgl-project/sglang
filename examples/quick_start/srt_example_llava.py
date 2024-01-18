import sglang as sgl


@sgl.function
def image_qa(s, image_path, question):
    s += sgl.user(sgl.image(image_path) + question)
    s += sgl.assistant(sgl.gen("answer"))


runtime = sgl.Runtime(model_path="liuhaotian/llava-v1.5-7b",
                      tokenizer_path="llava-hf/llava-1.5-7b-hf")
sgl.set_default_backend(runtime)


state = image_qa.run(image_path="images/cat.jpeg", question="What is this?")
print(state["answer"])

runtime.shutdown()
