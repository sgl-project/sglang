import argparse
import os
import time

import openai

"""
# Edit the code file srt/models/deepseek_v2.py in the Python site package and add the logic for saving topk_ids:
# import get_tensor_model_parallel_rank
# DeepseekV2MoE::forward_normal
if hidden_states.shape[0] >= 4096 and get_tensor_model_parallel_rank() == 0:
    topk_ids_dir = xxxx
    if not hasattr(self, "save_idx"):
        self.save_idx = 0
    if self.save_idx <= 1:
        torch.save(topk_output.topk_ids, f"{topk_ids_dir}/topk_ids_layer{self.layer_id}_idx{self.save_idx}.pt")
    self.save_idx += 1
"""


def read_long_prompt():
    import json

    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{current_dir}/tuning_text.json", "r") as fp:
        text = fp.read()
    rst = json.loads(text)
    return rst["prompt"]


def openai_stream_test(model, ip, port):
    client = openai.Client(base_url=f"http://{ip}:{port}/v1", api_key="None")
    qst = read_long_prompt()

    messages = [
        {"role": "user", "content": qst},
    ]
    msg2 = dict(
        model=model,
        messages=messages,
        temperature=0.6,
        top_p=0.75,
        max_tokens=100,
    )
    response = client.chat.completions.create(**msg2, stream=True)
    time_start = time.time()
    time_cost = []
    for chunk in response:
        time_end = time.time()
        # if chunk.choices[0].delta.content:
        #    print(chunk.choices[0].delta.content, end="", flush=True)
        time_cost.append(time_end - time_start)
        time_start = time.time()

    ttft = time_cost[0] + time_cost[1]
    tpot = sum(time_cost[2:]) / len(time_cost[2:])
    print(f"\nTTFT {ttft}, TPOT {tpot}")
    return ttft, tpot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="auto")
    parser.add_argument(
        "--ip",
        type=str,
        default="127.0.0.1",
    )
    parser.add_argument("--port", type=int, default=8188)
    args = parser.parse_args()
    openai_stream_test(args.model, args.ip, args.port)
