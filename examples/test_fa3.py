import sglang as sgl


@sgl.function
def few_shot_qa1(s, question):
    s += """The following are questions with answers.
        Q: What is the capital of France?
        A: Paris
        Q: What is the capital of Germany?
        A: Berlin
        Q: What is the capital of Italy?
        A: Rome
    """
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n", temperature=0)


def single1():
    state = few_shot_qa1.run(question="What is the capital of the United States?")
    answer = state["answer"].strip().lower()
    print(state.get_meta_info("answer"))
    print(state.text())
    assert "washington" in answer, f"answer: {state['answer']}"


@sgl.function
def few_shot_qa2(s, question):
    s += """The following are questions with answers.
        Q: What is the capital of France?
        A: Paris
        Q: What is the capital of Germany?
        A: Berlin
        Q: What is the capital of Italy?
        A: Rome
    """
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n", temperature=0)

def single2():
    state = few_shot_qa2.run(question="What is the capital of the United States?")
    
    answer = state["answer"].strip().lower()
    print(state.get_meta_info("answer"))
    print(state.text())
    assert "washington" in answer, f"answer: {state['answer']}"

if __name__ == "__main__":
    # model_path = "/shared/public/models/meta-llama/Llama-3.2-3B-Instruct"
    model_path = "/shared/public/models/meta-llama/Llama-3.2-1B-Instruct"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    runtime = sgl.Runtime(model_path=model_path,
                        mem_fraction_static=0.70, disable_cuda_graph=True,
                        disable_radix_cache=False, chunked_prefill_size=-1, # , is_embedding=True
                        attention_backend="flashattention",
                        )
    sgl.set_default_backend(runtime)

    # Run a single request
    print("\n========== single1 ==========\n")
    single1()

    print("\n========== single2 ==========\n")
    single2()

    runtime.shutdown()
