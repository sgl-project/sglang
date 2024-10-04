"""
Usage:
python3 local_example_chat.py
"""

import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer


@sgl.function
def long_answer(s):
    s += sgl.user("What is the capital of the United States?")
    s += sgl.assistant(sgl.gen("answer", min_tokens=64, max_tokens=128))


def assert_min_token_length(tokenizer, s, min_tokens):
    token_ids = tokenizer.encode(s)
    print(len(token_ids))
    assert len(token_ids) > min_tokens


if __name__ == "__main__":
    runtime = sgl.Runtime(
        model_path="/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/"
    )
    sgl.set_default_backend(runtime)

    tokenizer = get_tokenizer(
        "/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/"
    )

    print("===================================")
    state = long_answer.run()
    assert_min_token_length(tokenizer, state["answer"], 64)

    runtime.shutdown()
