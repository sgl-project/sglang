import sglang as sgl


@sgl.function
def decode(s):
    s += sgl.user("What is the capital of the United States?")
    s += sgl.assistant(sgl.gen("answer"))


@sgl.function
def decode_int(s):
    s += "The number of hours in a day is " + sgl.gen_int("hours") + "\n"
    s += "The number of days in a year is " + sgl.gen_int("days") + "\n"


@sgl.function
def decode_int_and_decode(s):
    s += "The number of hours in a day is " + sgl.gen_int("hours") + "\n"
    s += "The number of days in a year is " + sgl.gen_int("days") + "\n"
    s += sgl.user("What is the capital of the United States?")
    s += sgl.assistant(sgl.gen("answer"))


model_path = "/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/"

if __name__ == "__main__":
    runtime = sgl.Runtime(model_path=model_path)
    sgl.set_default_backend(runtime)

    print("==== Run decode_int and decode in two sgl.function ====")
    state = decode_int.run()
    print(state.text())
    state = decode.run()
    print(state.text())

    print("==== Run decode_int and decode in one sgl.function ====")
    state = decode_int_and_decode.run()
    print(state.text())

    runtime.shutdown()
