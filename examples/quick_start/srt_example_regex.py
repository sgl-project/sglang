from sglang import function, gen, set_default_backend, Runtime


IP_ADDR_REGEX = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"


@function
def regex_gen(s):
    s += "Q: What is the IP address of the Google DNS servers?\n"
    s += "A: " + gen(
        "answer",
        temperature=0,
        regex=IP_ADDR_REGEX,
    )


runtime = Runtime(model_path="meta-llama/Llama-2-7b-chat-hf")
set_default_backend(runtime)

state = regex_gen.run()

print(state.text())

runtime.shutdown()
