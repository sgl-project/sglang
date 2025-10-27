"""
Run one test prompt with support for multi-turn conversations and client-side tokenization.

Features:
- Multi-turn conversations with "Can you elaborate more?" follow-ups
- Client-side tokenization with input_ids instead of raw text
- Token-based conversation continuity (appends output_ids for subsequent turns)

Usage:
python3 -m sglang.test.send_one
python3 -m sglang.test.send_one --turns 3  # For 3-turn conversation
python3 -m sglang.test.send_one --tokenizer-path /path/to/tokenizer --turns 2  # Use tokenization

When using --tokenizer-path:
- First turn: text -> tokens -> input_ids sent to server
- Subsequent turns: previous output_ids + follow-up tokens -> input_ids sent to server
- Maintains token-space conversation continuity without text roundtrips
"""

import argparse
import dataclasses
import json

import requests

from sglang.srt.utils.hf_transformers_utils import get_tokenizer


@dataclasses.dataclass
class BenchArgs:
    host: str = "localhost"
    port: int = 30000
    batch_size: int = 1
    temperature: float = 0.0
    max_new_tokens: int = 512
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    json: bool = False
    return_logprob: bool = False
    prompt: str = (
        "Human: Give me a fully functional FastAPI server. Show the python code.\n\nAssistant:"
    )
    prompt_suffix: str = ""
    image: bool = False
    many_images: bool = False
    stream: bool = False
    turns: int = 1
    tokenizer_path: str = ""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--host", type=str, default=BenchArgs.host)
        parser.add_argument("--port", type=int, default=BenchArgs.port)
        parser.add_argument("--batch-size", type=int, default=BenchArgs.batch_size)
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature)
        parser.add_argument(
            "--max-new-tokens", type=int, default=BenchArgs.max_new_tokens
        )
        parser.add_argument(
            "--frequency-penalty", type=float, default=BenchArgs.frequency_penalty
        )
        parser.add_argument(
            "--presence-penalty", type=float, default=BenchArgs.presence_penalty
        )
        parser.add_argument("--json", action="store_true")
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument("--prompt", type=str, default=BenchArgs.prompt)
        parser.add_argument(
            "--prompt-suffix", type=str, default=BenchArgs.prompt_suffix
        )
        parser.add_argument("--image", action="store_true")
        parser.add_argument("--many-images", action="store_true")
        parser.add_argument("--stream", action="store_true")
        parser.add_argument(
            "--turns",
            type=int,
            default=BenchArgs.turns,
            help="Number of conversation turns",
        )
        parser.add_argument(
            "--tokenizer-path",
            type=str,
            default=BenchArgs.tokenizer_path,
            help="Path to tokenizer for client-side tokenization",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def make_request(args, prompt, image_data=None, json_schema=None, tokenizer=None):
    """Helper function to make a single request to the server."""

    # Check if prompt is already tokenized (list of ints) or text
    if isinstance(prompt, list) and tokenizer and args.tokenizer_path:
        # prompt is already tokenized input_ids
        if args.batch_size > 1:
            input_ids = [prompt] * args.batch_size
        else:
            input_ids = prompt

        json_data = {
            "input_ids": input_ids,  # Use pre-tokenized input_ids
            "image_data": image_data,
            "sampling_params": {
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "frequency_penalty": args.frequency_penalty,
                "presence_penalty": args.presence_penalty,
                "json_schema": json_schema,
                "stop": ["Question", "Assistant:", "<|separator|>", "<|eos|>"],
            },
            "return_logprob": args.return_logprob,
            "stream": args.stream,
        }
    elif tokenizer and args.tokenizer_path:
        # Tokenize the text prompt
        if args.batch_size > 1:
            if isinstance(prompt, str):
                prompts = [prompt] * args.batch_size
            else:
                prompts = prompt
            input_ids = [tokenizer.encode(p) for p in prompts]
        else:
            input_ids = tokenizer.encode(prompt)

        json_data = {
            "input_ids": input_ids,  # Use input_ids instead of text
            "image_data": image_data,
            "sampling_params": {
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "frequency_penalty": args.frequency_penalty,
                "presence_penalty": args.presence_penalty,
                "json_schema": json_schema,
                "stop": ["Question", "Assistant:", "<|separator|>", "<|eos|>"],
            },
            "return_logprob": args.return_logprob,
            "stream": args.stream,
        }
    else:
        # Fallback to original text-based approach
        if args.batch_size > 1:
            prompt = [prompt] * args.batch_size

        json_data = {
            "text": prompt,
            "image_data": image_data,
            "sampling_params": {
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "frequency_penalty": args.frequency_penalty,
                "presence_penalty": args.presence_penalty,
                "json_schema": json_schema,
                "stop": ["Question", "Assistant:", "<|separator|>", "<|eos|>"],
            },
            "return_logprob": args.return_logprob,
            "stream": args.stream,
        }

    response = requests.post(
        f"http://{args.host}:{args.port}/generate",
        json=json_data,
        stream=args.stream,
    )

    if args.stream:
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                ret = json.loads(chunk[5:].strip("\n"))
    else:
        ret = response.json()

    if args.batch_size > 1:
        ret = ret[0]

    return response, ret


def send_one_prompt(args):
    # Initialize tokenizer if path is provided
    tokenizer = None
    if args.tokenizer_path:
        try:
            tokenizer = get_tokenizer(args.tokenizer_path)
            print(f"Loaded tokenizer from: {args.tokenizer_path}")
        except Exception as e:
            print(f"Failed to load tokenizer from {args.tokenizer_path}: {e}")
            print("Falling back to text-based requests")
            tokenizer = None
            raise e

    if args.image:
        args.prompt = (
            "Human: Describe this image in a very short sentence.\n\nAssistant:"
        )
        image_data = "https://raw.githubusercontent.com/sgl-project/sglang/main/test/lang/example_image.png"
    elif args.many_images:
        args.prompt = (
            "Human: I have one reference image and many images."
            "Describe their relationship in a very short sentence.\n\nAssistant:"
        )
        image_data = [
            "https://raw.githubusercontent.com/sgl-project/sglang/main/test/lang/example_image.png",
            "https://raw.githubusercontent.com/sgl-project/sglang/main/test/lang/example_image.png",
            "https://raw.githubusercontent.com/sgl-project/sglang/main/test/lang/example_image.png",
            "https://raw.githubusercontent.com/sgl-project/sglang/main/test/lang/example_image.png",
        ]
    else:
        image_data = None

    prompt = args.prompt

    if args.json:
        prompt = (
            "Human: What is the capital of France and how is that city like. "
            "Give me 3 trivial information about that city. "
            "Write in a format of json.\nAssistant:"
        )
        json_schema = "$$ANY$$"
    else:
        json_schema = None

    # Initialize variables for multi-turn conversation
    conversation_history = []
    total_acc_length = 0.0
    total_speed = 0.0

    # Keep track of input_ids for tokenized requests
    current_input_ids = None
    if tokenizer and args.tokenizer_path:
        current_input_ids = tokenizer.encode(prompt)
        follow_up_text = "\n\nHuman: Can you elaborate more?\n\nAssistant:"
        follow_up_tokens = tokenizer.encode(follow_up_text)

    # Perform multiple turns
    for turn in range(args.turns):
        print(f"\n{'='*20} Turn {turn + 1} {'='*20}")

        # For tokenized requests, use current_input_ids if available
        if tokenizer and args.tokenizer_path and current_input_ids is not None:
            request_prompt = current_input_ids
        else:
            request_prompt = prompt

        # Make the request with tokenizer
        response, ret = make_request(
            args, request_prompt, image_data, json_schema, tokenizer
        )

        if response.status_code != 200:
            print(ret)
            return 0, 0

        # Calculate metrics for this turn
        latency = ret["meta_info"]["e2e_latency"]

        if "spec_verify_ct" in ret["meta_info"]:
            acc_length = (
                ret["meta_info"]["completion_tokens"]
                / ret["meta_info"]["spec_verify_ct"]
            )
        else:
            acc_length = 1.0

        speed = ret["meta_info"]["completion_tokens"] / latency

        # Accumulate metrics
        total_acc_length += acc_length
        total_speed += speed

        # Display the response for this turn
        print(ret["text"])
        print(f"Turn {turn + 1} - {acc_length=:.2f}, {speed=:.2f} token/s")

        # Store the conversation
        conversation_history.append(
            {
                "turn": turn + 1,
                "prompt": (
                    request_prompt
                    if isinstance(request_prompt, str)
                    else f"<tokenized: {len(request_prompt)} tokens>"
                ),
                "response": ret["text"],
                "acc_length": acc_length,
                "speed": speed,
            }
        )

        # Prepare for next turn (if not the last turn)
        if turn < args.turns - 1:
            if tokenizer and args.tokenizer_path and current_input_ids is not None:
                # For tokenized requests: append output_ids and follow-up tokens
                if "output_ids" in ret:
                    assistant_output_ids = ret["output_ids"]
                    print(f"Assistant output_ids length: {len(assistant_output_ids)}")
                    # Append assistant response tokens + follow-up question tokens
                    current_input_ids.extend(assistant_output_ids)
                    current_input_ids.extend(follow_up_tokens)
                    print(f"Extended input_ids length: {len(current_input_ids)}")
                else:
                    print(
                        "Warning: output_ids not found in response, falling back to text concatenation"
                    )
                    # Fallback to text-based approach
                    assistant_response = ret["text"]
                    prompt = f"{prompt}{assistant_response}\n\nHuman: Can you elaborate more?\n\nAssistant:"
                    current_input_ids = None  # Switch to text mode
            else:
                # For text-based requests: concatenate text as before
                assistant_response = ret["text"]
                # print(f"Assistant response: {ret=}")
                prompt = f"{prompt}{assistant_response}\n\nHuman: Can you elaborate more?\n\nAssistant:"

    # Display summary
    print(f"\n{'='*20} Conversation Summary {'='*20}")
    avg_acc_length = total_acc_length / args.turns
    avg_speed = total_speed / args.turns

    print(f"Total turns: {args.turns}")
    print(f"Average acc_length: {avg_acc_length:.2f}")
    print(f"Average speed: {avg_speed:.2f} token/s")

    if tokenizer:
        print(f"Used tokenizer: {args.tokenizer_path}")
        print("Sent input_ids instead of raw text")
    else:
        print("Used raw text (no tokenizer)")

    return avg_acc_length, avg_speed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    send_one_prompt(args)
