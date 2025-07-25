# Benchmark with lots of common prefixes. Used to benchmark prefix caching performance.
#
# Launch a server:
# python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --log-level-http warning

import random
import string
import time
import os
import dataclasses
from typing import Tuple
import argparse
import multiprocessing
import json
import itertools

from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Union
import requests
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

import sglang as sgl
from sglang import set_default_backend
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.srt.entrypoints.http_server import launch_server
from sglang.bench_serving import get_tokenizer, sample_random_requests
from sglang.profiler import run_profile
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import is_in_ci, write_github_step_summary

@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (32,)
    temperature: float = 0.0
    return_logprob: bool = False
    input_len_step_percentage: float = 0.0
    result_filename: str = "result.jsonl"
    base_url: str = ""
    skip_warmup: bool = False
    show_report: bool = False
    profile: bool = False
    profile_by_stage: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature)
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument(
            "--input-len-step-percentage",
            type=float,
            default=BenchArgs.input_len_step_percentage,
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--base-url", type=str, default=BenchArgs.base_url)
        parser.add_argument("--skip-warmup", action="store_true")
        parser.add_argument("--show-report", action="store_true")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument("--profile-by-stage", action="store_true")

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def generate_random_string(token_length: int) -> str:
    random_string = "".join(
        random.choices(string.ascii_letters + string.digits, k=token_length * 100)
    )
    tokenized_output = tokenizer.encode(random_string, add_special_tokens=False)[
        :token_length
    ]

    if len(tokenized_output) < token_length:
        tokenized_output = tokenized_output + [tokenizer.pad_token_id] * (
            token_length - len(tokenized_output)
        )

    decoded_string = tokenizer.decode(tokenized_output, skip_special_tokens=False)
    return decoded_string


def generate_unique_prefix(base_text, index):
    return str(index) + base_text[len(str(index)) :]


@sgl.function
def text_qa(s, question, gen_len):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n", temperature=0, max_tokens=gen_len)


def prepare_prompts(num_prefix, num_samples_per_prefix, prefix_length, suffix_length):
    base_prefix = generate_random_string(prefix_length)

    tot_input_len = 0
    all_prompts = []
    for i in tqdm(range(num_prefix), desc="prepare prompts"):
        unique_prefix = generate_unique_prefix(base_prefix, i)
        prompt_list = []
        for j in range(num_samples_per_prefix[i]):
            suffix = generate_random_string(suffix_length)
            prompt = unique_prefix + suffix
            prompt_list.append(tokenizer.encode(prompt))
            # tot_input_len += len(tokenizer.encode(prompt))
        all_prompts += prompt_list
    return all_prompts, tot_input_len

def run_one_case(
    url: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    temperature: float,
    return_logprob: bool,
    input_len_step_percentage: float,
    run_name: str,
    result_filename: str,
    tokenizer,
    profile: bool = False,
    profile_by_stage: bool = False,
):
    requests.post(url + "/flush_cache")
    input_requests = sample_random_requests(
        input_len=input_len,
        output_len=output_len,
        num_prompts=batch_size,
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path="",
        random_sample=True,
        return_text=False,
    )

    use_structured_outputs = False
    if use_structured_outputs:
        texts = []
        for _ in range(batch_size):
            texts.append(
                "Human: What is the capital city of france? can you give as many trivial information as possible about that city? answer in json.\n"
                * 50
                + "Assistant:"
            )
        json_schema = "$$ANY$$"
    else:
        json_schema = None

    profile_link = None
    if profile:
        profile_link: str = run_profile(
            url, 3, ["CPU", "GPU"], None, None, profile_by_stage
        )
    tic = time.perf_counter()
    # print(f"input ids: {[req.prompt for req in input_requests]}")
    response = requests.post(
        url + "/generate",
        json={
            "input_ids": [req.prompt for req in input_requests],
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": output_len,
                "ignore_eos": True,
                "json_schema": json_schema,
            },
            "return_logprob": return_logprob,
            "stream": True,
        },
        stream=True,
    )

    # The TTFT of the last request in the batch
    ttft = 0.0
    for chunk in response.iter_lines(decode_unicode=False):
        chunk = chunk.decode("utf-8")
        if chunk and chunk.startswith("data:"):
            if chunk == "data: [DONE]":
                break
            data = json.loads(chunk[5:].strip("\n"))
            if "error" in data:
                raise RuntimeError(f"Request has failed. {data}.")

            assert (
                data["meta_info"]["finish_reason"] is None
                or data["meta_info"]["finish_reason"]["type"] == "length"
            )
            if data["meta_info"]["completion_tokens"] == 1:
                ttft = time.perf_counter() - tic

    latency = time.perf_counter() - tic
    input_throughput = batch_size * input_len / ttft
    output_throughput = batch_size * output_len / (latency - ttft)
    overall_throughput = batch_size * (input_len + output_len) / latency

    server_info = requests.get(url + "/get_server_info").json()
    acc_length = server_info["internal_states"][0].get("avg_spec_accept_length", None)
    last_gen_throughput = server_info["internal_states"][0]["last_gen_throughput"]

    print(f"batch size: {batch_size}")
    print(f"input_len: {input_len}")
    print(f"output_len: {output_len}")
    print(f"latency: {latency:.2f} s")
    print(f"ttft: {ttft:.2f} s")
    print(f"last generation throughput: {last_gen_throughput:.2f} tok/s")
    print(f"input throughput: {input_throughput:.2f} tok/s")
    if output_len != 1:
        print(f"output throughput: {output_throughput:.2f} tok/s")

    if result_filename:
        with open(result_filename, "a") as fout:
            res = {
                "run_name": run_name,
                "batch_size": batch_size,
                "input_len": input_len,
                "output_len": output_len,
                "latency": round(latency, 4),
                "output_throughput": round(output_throughput, 2),
                "overall_throughput": round(overall_throughput, 2),
                "last_gen_throughput": round(last_gen_throughput, 2),
            }
            fout.write(json.dumps(res) + "\n")

    return (
        batch_size,
        latency,
        ttft,
        input_throughput,
        output_throughput,
        overall_throughput,
        last_gen_throughput,
        acc_length,
        profile_link if profile else None,
    )

def run_one_batch(
    url: str,
    batch_size: int,
    input_ids: list,
    output_len: int,
    temperature: float,
    return_logprob: bool,
    input_len_step_percentage: float,
    run_name: str,
    result_filename: str,
    tokenizer,
    profile: bool = False,
    profile_by_stage: bool = False,
):
    # requests.post(url + "/flush_cache")
    # input_requests = sample_random_requests(
    #     input_len=input_len,
    #     output_len=output_len,
    #     num_prompts=batch_size,
    #     range_ratio=1.0,
    #     tokenizer=tokenizer,
    #     dataset_path="",
    #     random_sample=True,
    #     return_text=False,
    # )
    # for _ in range(3):
    use_structured_outputs = False
    if use_structured_outputs:
        texts = []
        for _ in range(batch_size):
            texts.append(
                "Human: What is the capital city of france? can you give as many trivial information as possible about that city? answer in json.\n"
                * 50
                + "Assistant:"
            )
        json_schema = "$$ANY$$"
    else:
        json_schema = None
    profile_link = None
    if profile:
        profile_link: str = run_profile(
            url, 3, ["CPU", "GPU"], None, None, profile_by_stage
        )
    # print(f"err input ids: {[input_ids[0]]}")
    tic = time.perf_counter()
    
    response = requests.post(
        url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": output_len,
                "ignore_eos": True,
                "json_schema": json_schema,
            },
            "return_logprob": return_logprob,
            "stream": True,
        },
        stream=True,
    )
    # The TTFT of the last request in the batch
    ttft = 0.0
    for chunk in response.iter_lines(decode_unicode=False):
        chunk = chunk.decode("utf-8")
        if chunk and chunk.startswith("data:"):
            if chunk == "data: [DONE]":
                break
            data = json.loads(chunk[5:].strip("\n"))
            if "error" in data:
                raise RuntimeError(f"Request has failed. {data}.")
            assert (
                data["meta_info"]["finish_reason"] is None
                or data["meta_info"]["finish_reason"]["type"] == "length"
            )
            if data["meta_info"]["completion_tokens"] == 1:
                ttft = time.perf_counter() - tic
    latency = time.perf_counter() - tic
    input_throughput = sum([len(input_id) for input_id in input_ids]) / ttft
    output_throughput = batch_size * output_len / (latency - ttft)
    overall_throughput = (batch_size *  output_len + sum([len(input_id) for input_id in input_ids])) / latency
    server_info = requests.get(url + "/get_server_info").json()
    acc_length = server_info["internal_states"][0].get("avg_spec_accept_length", None)
    last_gen_throughput = server_info["internal_states"][0]["last_gen_throughput"]
    print(f"batch size: {batch_size}")
    print(f"input_len: {[len(input_id) for input_id in input_ids]}")
    print(f"output_len: {output_len}")
    print(f"latency: {latency:.2f} s")
    print(f"ttft: {ttft:.2f} s")
    print(f"last generation throughput: {last_gen_throughput:.2f} tok/s")
    print(f"input throughput: {input_throughput:.2f} tok/s")
    if output_len != 1:
        print(f"output throughput: {output_throughput:.2f} tok/s")
    if result_filename:
        with open(result_filename, "a") as fout:
            res = {
                "run_name": run_name,
                "batch_size": batch_size,
                "input_len": [len(input_id) for input_id in input_ids],
                "output_len": output_len,
                "latency": round(latency, 4),
                "output_throughput": round(output_throughput, 2),
                "overall_throughput": round(overall_throughput, 2),
                "last_gen_throughput": round(last_gen_throughput, 2),
            }
            fout.write(json.dumps(res) + "\n")

    return (
        batch_size,
        latency,
        ttft,
        input_throughput,
        output_throughput,
        overall_throughput,
        last_gen_throughput,
        acc_length,
        profile_link if profile else None,
    )


def get_model(pretrained_model_name_or_path: str) -> str:
    if os.getenv("SGLANG_USE_MODELSCOPE", "false").lower() == "true":
        import huggingface_hub.constants
        from modelscope import snapshot_download

        model_path = snapshot_download(
            model_id=pretrained_model_name_or_path,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
        )

        return model_path
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    assert (
        pretrained_model_name_or_path is not None
        and pretrained_model_name_or_path != ""
    )
    if pretrained_model_name_or_path.endswith(
        ".json"
    ) or pretrained_model_name_or_path.endswith(".model"):
        from sglang.srt.hf_transformers_utils import get_tokenizer

        return get_tokenizer(pretrained_model_name_or_path)

    if pretrained_model_name_or_path is not None and not os.path.exists(
        pretrained_model_name_or_path
    ):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )

def launch_server_internal(server_args):
    try:
        launch_server(server_args)
    except Exception as e:
        print("exception e")
        raise e
    finally:
        print("kill precess tree")
        kill_process_tree(os.getpid(), include_parent=False)


def launch_server_process(server_args: ServerArgs):
    proc = multiprocessing.Process(target=launch_server_internal, args=(server_args,))
    proc.start()
    base_url = f"http://{server_args.host}:{server_args.port}"
    timeout = 600

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
            }
            response = requests.get(f"{base_url}/v1/models", headers=headers)
            if response.status_code == 200:
                return proc, base_url
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("Server failed to start within the timeout period.")

def run_benchmark(server_args: ServerArgs, bench_args: BenchArgs):
    if bench_args.base_url:
        proc, base_url = None, bench_args.base_url
    else:
        proc, base_url = launch_server_process(server_args)

    # warmup
    if not bench_args.skip_warmup:
        print("=" * 8 + " Warmup Begin " + "=" * 8)
        run_one_case(
            base_url,
            batch_size=16,
            input_len=1024,
            output_len=16,
            temperature=bench_args.temperature,
            return_logprob=bench_args.return_logprob,
            input_len_step_percentage=bench_args.input_len_step_percentage,
            run_name="",
            result_filename="",
            tokenizer=tokenizer,
        )
        print("=" * 8 + " Warmup End   " + "=" * 8 + "\n")

    # benchmark
    result = []
    bench_result = []
    try:
        for bs, ol in itertools.product(
            bench_args.batch_size,  bench_args.output_len
        ):
            # for prompts in _all_prompts:
            #     print("running for no prefetch")
            #     bench_result.append(
            #         (
            #             run_one_batch(
            #                 base_url,
            #                 bs,
            #                 prompts,
            #                 ol,
            #                 temperature=bench_args.temperature,
            #                 return_logprob=bench_args.return_logprob,
            #                 input_len_step_percentage=bench_args.input_len_step_percentage,
            #                 run_name=bench_args.run_name,
            #                 result_filename=bench_args.result_filename,
            #                 tokenizer=tokenizer,
            #                 profile=bench_args.profile,
            #                 profile_by_stage=bench_args.profile_by_stage,
            #             )[-1],
            #         )
            #     )
            for prompts in all_prompts:
                # print("running for prefetch")
                result.append(
                    run_one_batch(
                        base_url,
                        bs,
                        prompts,
                        ol,
                        temperature=bench_args.temperature,
                        return_logprob=bench_args.return_logprob,
                        input_len_step_percentage=bench_args.input_len_step_percentage,
                        run_name=bench_args.run_name,
                        result_filename=bench_args.result_filename,
                        tokenizer=tokenizer,
                    )
                )

        if bench_args.profile:
            try:
                for bs, ol in itertools.product(
                    bench_args.batch_size, bench_args.output_len
                ):
                    for prompts in _all_prompts:
                        bench_result.append(
                            (
                                run_one_batch(
                                    base_url,
                                    bs,
                                    prompts,
                                    ol,
                                    temperature=bench_args.temperature,
                                    return_logprob=bench_args.return_logprob,
                                    input_len_step_percentage=bench_args.input_len_step_percentage,
                                    run_name=bench_args.run_name,
                                    result_filename=bench_args.result_filename,
                                    tokenizer=tokenizer,
                                    profile=bench_args.profile,
                                    profile_by_stage=bench_args.profile_by_stage,
                                )[-1],
                            )
                        )
                    for prompts in all_prompts:
                        bench_result.append(
                            (
                                run_one_batch(
                                    base_url,
                                    bs,
                                    prompts,
                                    ol,
                                    temperature=bench_args.temperature,
                                    return_logprob=bench_args.return_logprob,
                                    input_len_step_percentage=bench_args.input_len_step_percentage,
                                    run_name=bench_args.run_name,
                                    result_filename=bench_args.result_filename,
                                    tokenizer=tokenizer,
                                    profile=bench_args.profile,
                                    profile_by_stage=bench_args.profile_by_stage,
                                )[-1],
                            )
                        )
                result = [t1[:-1] + t2 for t1, t2 in zip(result, bench_result)]
            except Exception as e:
                print(f"Error profiling, there will be no profile trace dump: {e}")
    finally:
        if proc:
            kill_process_tree(proc.pid)

    print(f"\nResults are saved to {bench_args.result_filename}")

    if not bench_args.show_report:
        return

    summary = (
        f"\nInput lens: {bench_args.input_len}. Output lens: {bench_args.output_len}.\n"
    )
    summary += "| batch size | latency (s) | input throughput (tok/s)  | output throughput (tok/s) | acc length | ITL (ms) | input cost ($/1M) | output cost ($/1M) |"

    if bench_args.profile:
        summary += " profile |"

    summary += "\n"
    summary += "| ---------- | ----------- | ------------------------- | ------------------------- | ---------- | -------- | ----------------- | ------------------ |"

    if bench_args.profile:
        summary += "-------------|"
    summary += "\n"

    for (
        batch_size,
        latency,
        ttft,
        input_throughput,
        output_throughput,
        overall_throughput,
        last_gen_throughput,
        acc_length,
        trace_link,
    ) in result:
        hourly_cost = 2 * server_args.tp_size  # $2/hour for one H100
        input_util = 0.7
        accept_length = round(acc_length, 2) if acc_length is not None else "n/a"
        line = (
            f"| {batch_size} | "
            f"{latency:.2f} | "
            f"{input_throughput:.2f} | "
            f"{output_throughput:.2f} | "
            f"{accept_length} | "
            f"{1 / (output_throughput/batch_size) * 1000:.2f} | "
            f"{1e6 / (input_throughput * input_util) / 3600 * hourly_cost:.2f} | "
            f"{1e6 / output_throughput / 3600 * hourly_cost:.2f} |"
        )
        if trace_link:
            line += f" [Profile]({trace_link}) |"
        line += "\n"
        summary += line

    # print metrics table
    print(summary)

    if is_in_ci():
        write_github_step_summary(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    tokenizer_id = server_args.tokenizer_path or server_args.model_path
    print(f"tokenizer id: {tokenizer_id}")
    tokenizer = get_tokenizer(tokenizer_id)

    random.seed(1)
    _num_prefix = 1
    _num_samples_per_prefix = [128]
    _prefix_length = 1024
    _suffix_length = 1024
    _all_prompts, _tot_input_len = prepare_prompts(
        _num_prefix, _num_samples_per_prefix, _prefix_length, _suffix_length
    )

    random.seed(0)
    num_prefix = 32
    # num_samples_per_prefix = [128] + [32 for _ in range(num_prefix - 1)]
    num_samples_per_prefix = [8 for _ in range(num_prefix)]
    print(f"num samples per prefix: {num_samples_per_prefix}")
    prefix_length = 1024
    suffix_length = 512
    gen_len = 1
    all_prompts, tot_input_len = prepare_prompts(
        num_prefix, num_samples_per_prefix, prefix_length, suffix_length
    )
    all_prompts = [all_prompts]
    print(f"Total input token length: {tot_input_len}\n")


    run_benchmark(server_args, bench_args)
    
