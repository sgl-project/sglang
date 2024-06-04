from pprint import pformat

import sglang as sgl


@sgl.function
def cot_decoding(s, question, get_top_k):
    """CoT Decoding: http://arxiv.org/abs/2402.10200"""

    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant_begin()
    s += " Let's think step by step. "
    s += sgl.gen(
        "get_top_k",
        max_tokens=0,
        return_logprob=True,
        top_logprobs_num=get_top_k,
        return_text_in_logprobs=True,
    )

    forks = s.fork(get_top_k)
    logprobs = s.get_meta_info("get_top_k")["decode_top_logprobs"][0]
    for idx, (f, token) in enumerate(zip(forks, logprobs)):
        logprob, token_id, text = token
        f += text + sgl.gen("answer", max_tokens=256)
        f += sgl.assistant_end()
        print(f"Path #{idx}", logprob, pformat(text + f["answer"]))


sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

state = cot_decoding.run(
    question=
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    get_top_k=10,
)
