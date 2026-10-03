"""Reproduce prefill backend output changes with training capture absent."""

import argparse
import json
from pathlib import Path

import requests
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase, free_port
from sglang.test.test_utils import popen_launch_server


def run(model, backend):
    url = f"http://127.0.0.1:{free_port()}"
    graph = {
        "prefill": {"backend": backend, "bs": [16, 32, 64, 128], "max_bs": 128},
        "decode": {
            "backend": "disabled" if backend == "disabled" else "full",
            "bs": [1, 2, 4],
            "max_bs": 4,
        },
    }
    if backend == "full":
        graph["prefill"]["full_prefill_max_req"] = 4
    server = popen_launch_server(
        model,
        url,
        timeout=600,
        other_args=[
            "--tp-size",
            "4",
            "--pp-max-micro-batch-size",
            "4",
            "--skip-server-warmup",
            "--skip-tokenizer-init",
            "--attention-backend",
            "flashinfer",
            "--disable-flashinfer-autotune",
            "--mem-fraction-static",
            "0.25",
            "--max-total-tokens",
            "4096",
            "--max-running-requests",
            "4",
            "--max-prefill-tokens",
            "128",
            "--chunked-prefill-size",
            "128",
            "--cuda-graph-config",
            json.dumps(graph),
            *(["--disable-overlap-schedule"] if backend == "disabled" else []),
            *(
                ["--enable-torch-compile-debug-mode"]
                if backend == "tc_piecewise"
                else []
            ),
        ],
    )
    records = {}

    def generate(name, prompts, lengths, *, biased=True):
        response = requests.post(
            url + "/generate",
            json={
                "rid": [f"control-{backend}-{name}-{i}" for i in range(len(prompts))],
                "input_ids": prompts,
                "sampling_params": [
                    {
                        "temperature": 0,
                        "max_new_tokens": length,
                        "ignore_eos": True,
                        **({"logit_bias": {"100": 100.0}} if biased else {}),
                    }
                    for length in lengths
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        values = response.json()
        for value, length in zip(values, lengths, strict=True):
            assert len(value["output_ids"]) == length
            if biased:
                assert value["output_ids"] == [100] * length
        records[name] = values

    try:
        info = requests.get(url + "/server_info", timeout=10)
        info.raise_for_status()
        assert info.json()["training_capture_config"] is None
        prompt = [100, 200, 300, 400] * 67 + [501, 502, 503]
        length = 8 if backend == "disabled" else 4
        generate("chunked", [prompt], [length])
        generate("cached-one-token", [prompt], [1])
        generate("cached-extension", [prompt + list(range(600, 619))], [length])
        for iteration in range(2):
            generate(
                f"batch-{iteration}",
                [
                    list(
                        range(
                            2000 + iteration * 1000 + i * 100,
                            2000 + iteration * 1000 + i * 100 + count,
                        )
                    )
                    for i, count in enumerate([33, 37, 43])
                ],
                [length, length - 1, 1],
            )
        generate("reject", [list(range(8000, 8013))], [6], biased=False)
    finally:
        PDCaptureRuntimeBase.stop_process(server)
    return {"backend": backend, "capture_config": None, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [run(args.model, backend) for backend in ("disabled", "tc_piecewise")]
    report = {
        "model": args.model,
        "results": results,
        "reject_outputs": [
            row["records"]["reject"][0]["output_ids"] for row in results
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"capture_off_prefill": report["reject_outputs"]}), flush=True)


if __name__ == "__main__":
    main()
