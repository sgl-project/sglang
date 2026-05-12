import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import jsonschema
import requests
from datasets import load_dataset

import sglang as sgl
from sglang.global_config import global_config
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text, normalize_base_url


@sgl.function
def schema_gen(s, message: Tuple[str, str], json_schema: str, max_tokens: int):
    system, user = message
    s += sgl.system(system)
    s += sgl.user(user)
    s += sgl.assistant(
        sgl.gen(
            "json_output",
            temperature=0,
            max_tokens=max_tokens,
            json_schema=json_schema,
        )
    )


def contains_formats(schema, formats: List[str]):
    if isinstance(schema, dict):
        if schema.get("format", None) in formats:
            return True
        for value in schema.values():
            if contains_formats(value, formats):
                return True
    elif isinstance(schema, list):
        for item in schema:
            if contains_formats(item, formats):
                return True
    return False


def convert_dataset(path: str):
    raw_dataset = load_dataset(path)
    dataset = []
    for data in raw_dataset["train"]:
        messages = data["prompt"]
        schema = data["schema"]
        obj = json.loads(schema)

        # skip some corrupted examples
        if obj.get("type", None) is None:
            continue

        # skip schema with format "email"
        # which is not supported by outlines for now
        if contains_formats(obj, ["email"]):
            continue

        system = messages[0]
        user = messages[1]
        assert system["role"] == "system", "invalid role"
        assert user["role"] == "user", "invalid role"
        assert len(messages) == 2, "invalid message length"
        message = json.dumps(system["content"]), json.dumps(user["content"])
        dataset.append(
            {
                "message": message,
                "chat_messages": [
                    {"role": "system", "content": system["content"]},
                    {"role": "user", "content": user["content"]},
                ],
                "json_schema": schema,
            }
        )

    return dataset


def load_arguments(args):
    arguments = convert_dataset(args.data_path)

    if args.num_jsons < 0 or args.num_jsons > len(arguments):
        args.num_jsons = len(arguments)
    return arguments[: args.num_jsons]


def bench_schema_frontend(args, arguments):
    frontend_arguments = [
        {
            "message": argument["message"],
            "json_schema": argument["json_schema"],
            "max_tokens": args.max_tokens,
        }
        for argument in arguments
    ]

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Run requests
    tic = time.perf_counter()
    states = schema_gen.run_batch(
        frontend_arguments,
        temperature=0,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic
    outputs = [state["json_output"] for state in states]
    return states, outputs, latency


def _send_chat_request(args, argument: Dict[str, Any]) -> Dict[str, Any]:
    base_url = normalize_base_url(args.host, args.port)
    schema = json.loads(argument["json_schema"])
    response = None
    payload = {
        "model": args.model_id,
        "messages": argument["chat_messages"],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "separate_reasoning": args.thinking,
        "chat_template_kwargs": {"enable_thinking": args.thinking},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "schema",
                "schema": schema,
            },
        },
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=args.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        return {
            "ok": True,
            "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content"),
            "finish_reason": data["choices"][0].get("finish_reason"),
            "response": data,
        }
    except Exception as e:
        response_text = getattr(response, "text", None)
        return {
            "ok": False,
            "content": "",
            "reasoning_content": None,
            "finish_reason": None,
            "error": repr(e),
            "status_code": getattr(response, "status_code", None),
            "response_text": response_text[:4096] if response_text else None,
        }


def bench_schema_chat(args, arguments):
    results = [None] * len(arguments)
    tic = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_to_index = {
            executor.submit(_send_chat_request, args, argument): i
            for i, argument in enumerate(arguments)
        }
        completed = 0
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            results[i] = future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(arguments):
                print(f"Completed {completed}/{len(arguments)} requests")
    latency = time.perf_counter() - tic
    outputs = [result["content"] for result in results]
    return results, outputs, latency


def json_safe_raw_result(raw_result, output):
    if (
        isinstance(raw_result, (dict, list, str, int, float, bool))
        or raw_result is None
    ):
        return raw_result
    return {"json_output": output}


def get_finish_reason(raw_result):
    if isinstance(raw_result, dict):
        return raw_result.get("finish_reason")
    return None


def validate_outputs(arguments, outputs, raw_results):
    failed_indexes = []
    failures = []

    # Check if the outputs are valid
    for i, output in enumerate(outputs):
        try:
            schema = json.loads(arguments[i]["json_schema"])
            obj = json.loads(output)
            assert jsonschema.validate(obj, schema) is None
        except Exception as e:
            print(e)
            failed_indexes.append(i)
            failures.append(
                {
                    "index": i,
                    "exception": repr(e),
                    "output": output,
                    "schema": arguments[i]["json_schema"],
                    "messages": arguments[i]["chat_messages"],
                    "raw_result": json_safe_raw_result(raw_results[i], output),
                }
            )

    return failed_indexes, failures


def bench_schema(args):
    arguments = load_arguments(args)
    if args.chat_mode:
        raw_results, outputs, latency = bench_schema_chat(args, arguments)
    else:
        raw_results, outputs, latency = bench_schema_frontend(args, arguments)

    failed_indexes, failures = validate_outputs(arguments, outputs, raw_results)
    return arguments, raw_results, outputs, latency, failed_indexes, failures


def get_tokenizer_for_run(args):
    if args.chat_mode:
        server_info = requests.get(
            f"{normalize_base_url(args.host, args.port)}/server_info",
            timeout=30,
        ).json()
        return get_tokenizer(server_info["tokenizer_path"])
    return get_tokenizer(
        global_config.default_backend.get_server_info()["tokenizer_path"]
    )


def write_jsonl(path, rows):
    with open(path, "w") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")


def main(args):
    (
        _arguments,
        raw_results,
        outputs,
        latency,
        failed_indexes,
        failures,
    ) = bench_schema(args)

    # Compute accuracy
    tokenizer = get_tokenizer_for_run(args)
    num_output_tokens = sum(len(tokenizer.encode(x)) for x in outputs)
    num_valid = len(outputs) - len(failed_indexes)
    accuracy = num_valid / len(outputs) if outputs else 0.0
    length_failed_indexes = [
        failure["index"]
        for failure in failures
        if get_finish_reason(failure["raw_result"]) == "length"
    ]
    print(f"Latency: {latency:.3f}")
    print(f"Output throughput: {num_output_tokens / latency:.3f} token/s")
    print(f"#output tokens: {num_output_tokens}")
    print(f"Accuracy: {accuracy:.4f} ({num_valid}/{len(outputs)})")

    # Write results
    os.makedirs(args.output_dir, exist_ok=True)
    if args.chat_mode:
        write_jsonl(os.path.join(args.output_dir, "raw_responses.jsonl"), raw_results)
    else:
        dump_state_text(
            os.path.join(args.output_dir, f"tmp_output_{args.backend}.txt"),
            raw_results,
        )

    with open(os.path.join(args.output_dir, f"{args.backend}.jsonl"), "w") as fout:
        for output in outputs:
            fout.write(output + "\n")

    write_jsonl(os.path.join(args.output_dir, "failures.jsonl"), failures)

    value = {
        "task": "json_schema",
        "backend": args.backend,
        "chat_mode": args.chat_mode,
        "thinking": args.thinking,
        "latency": round(latency, 3),
        "num_jsons": args.num_jsons,
        "num_valid": num_valid,
        "num_failed": len(failed_indexes),
        "accuracy": round(accuracy, 6),
        "failed_indexes": failed_indexes,
        "num_length_failures": len(length_failed_indexes),
        "length_failed_indexes": length_failed_indexes,
        "parallel": args.parallel,
        "max_tokens": args.max_tokens,
    }

    with open(args.result_file, "a") as fout:
        fout.write(json.dumps(value) + "\n")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as fout:
        json.dump(value, fout, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="NousResearch/json-mode-eval")
    parser.add_argument("--num-jsons", type=int, default=-1)
    parser.add_argument("--chat-mode", action="store_true")
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen3 thinking mode in --chat-mode.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--model-id", type=str, default="default")
    parser.add_argument("--request-timeout", type=int, default=600)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
