# NOTE: Currently this can only be run through HTTP requests.
import json
from concurrent.futures import ThreadPoolExecutor

from json_decode import character_regex

from sglang.utils import http_request

character_names = ["Hermione Granger", "Ron Weasley", "Harry Potter"]

base_url = "http://localhost:30000"

prompt = "is a character in Harry Potter. Please fill in the following information about this character.\n"


def openai_api_request(name):
    data = {
        "model": "",
        "prompt": name + prompt,
        "temperature": 0,
        "max_tokens": 128,
        "regex": character_regex,
        "logprobs": 3,
    }
    res = http_request(base_url + "/v1/completions", json=data).json()
    with open(f"json_logprobs_{name.replace(' ', '_')}_tmp.json", "w") as fout:
        fout.write(json.dumps(res, indent=4))


def srt_api_request(name):
    data = {
        "text": name + prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 128,
            "regex": character_regex,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 3,
        "return_text_in_logprobs": True,
    }

    res = http_request(base_url + "/generate", json=data).json()

    # with open(f"json_logprobs_{name.replace(' ', '_')}_tmp.json", "w") as fout:
    #     fout.write(json.dumps(res, indent=4))

    meta_info = res["meta_info"]
    assert len(meta_info["prefill_token_logprobs"]) == len(
        meta_info["prefill_top_logprobs"]
    )
    assert len(meta_info["decode_token_logprobs"]) == len(
        meta_info["decode_top_logprobs"]
    )

    return res


def pretty_print(res):
    meta_info = res["meta_info"]

    print("\n\n", "=" * 30, "Prefill", "=" * 30)
    for i in range(len(meta_info["prefill_token_logprobs"])):
        print(f"{str(meta_info['prefill_token_logprobs'][i][2].encode()): <20}", end="")
        top_ks = (
            [str(t[2].encode()) for t in meta_info["prefill_top_logprobs"][i]]
            if meta_info["prefill_top_logprobs"][i]
            else []
        )
        for top_k in top_ks:
            print(f"{top_k: <15}", end="")
        print()

    print("\n\n", "=" * 30, "Decode", "=" * 30)
    for i in range(len(meta_info["decode_token_logprobs"])):
        print(f"{str(meta_info['decode_token_logprobs'][i][2].encode()): <20}", end="")
        top_ks = [str(t[2].encode()) for t in meta_info["decode_top_logprobs"][i]]
        for top_k in top_ks:
            print(f"{top_k: <15}", end="")
        print()

    print(res["text"])


if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        ress = executor.map(srt_api_request, character_names)

    for res in ress:
        pretty_print(res)

    openai_api_request("Hermione Granger")
