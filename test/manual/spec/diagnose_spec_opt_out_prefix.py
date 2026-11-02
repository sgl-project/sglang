"""Manual same-prefix diagnostic; CUDA required only when run, not on import.

Captures pre-sampling target logits at the two previously observed divergence
sites. Synchronization can perturb execution: warm_plain is deliberately unprobed.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

from test_per_request_spec_opt_out_e2e import PROMPTS, response_token_ids

# Prefixes from the saved 2026-11-02 run, before the FIRST divergent token.
PREFIX_OUTPUTS = {
    1: [
        429,
        979,
        39020,
        28833,
        279,
        9237,
        594,
        16566,
        11,
        432,
        374,
        36967,
        553,
        279,
        13673,
        18730,
        304,
        279,
        3720,
        11,
        1741,
        438,
        15797,
        323,
        3015,
        6973,
        89492,
        13,
        8697,
        3100,
        374,
        36967,
        803,
        1091,
        1008,
        7987,
        1576,
        432,
        34192,
        304,
        23327,
        16876,
        323,
        374,
        2686,
        4363,
        311,
        387,
        41001,
        553,
        279,
        16566,
        13,
        1096,
        11137,
        279,
        12884,
        311,
        4994,
        6303,
        13,
        576,
        7015,
    ],
    0: [220, 20, 11, 220],
}


def install_probes():
    import numpy as np
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative import eagle_utils

    root = Path(os.environ["PREFIX_RESULTS"])

    def capture(logits, positions, inputs, seq_lens, pool_indices, kind):
        case = json.loads((root / "active.json").read_text())
        if not case["capture"]:
            return []
        pos = positions.tolist()
        ids = inputs.tolist()
        output = []
        for row, (position, token) in enumerate(zip(pos, ids)):
            if [position, token] not in case["sites"]:
                continue
            values = logits[row].detach().float().cpu().numpy().copy()
            name = f"{case['name']}-{kind}-{time.time_ns()}-{row}"
            np.save(root / (name + ".npy"), values)
            order = np.argsort(values)[-5:][::-1].tolist()
            output.append(
                dict(
                    name=name,
                    row=row,
                    kind=kind,
                    position=position,
                    input_id=token,
                    seq_lens=seq_lens.tolist(),
                    req_pool_indices=pool_indices.tolist(),
                    top_ids=order,
                    top_logits=values[order].tolist(),
                )
            )
        return output

    def finish(records, selected):
        for record in records:
            record["selected"] = int(selected[record["row"]].item())
            with open(root / "captures.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")

    original = ModelRunner.sample

    def sample(self, output, batch):
        records = []
        if not self.is_draft_worker:
            if batch.forward_mode.is_decode():
                positions, ids = batch.positions, batch.input_ids
            else:
                positions = batch.seq_lens - 1
                ends = batch.extend_seq_lens.cumsum(0).long() - 1
                ids = batch.input_ids[ends]
            records = capture(
                output.next_token_logits,
                positions,
                ids,
                batch.seq_lens,
                batch.req_pool_indices,
                "decode" if batch.forward_mode.is_decode() else "prefill",
            )
        selected = original(self, output, batch)
        finish(records, selected)
        return selected

    ModelRunner.sample = sample
    original_roots = eagle_utils._sample_spec_opt_out_roots

    def roots(logits, verify, batch, sampler):
        idx = verify.retrieve_index[:, 0].long()
        records = capture(
            logits[idx],
            verify.positions[idx],
            verify.draft_token[idx],
            batch.seq_lens,
            batch.req_pool_indices,
            "verify_root",
        )
        selected = original_roots(logits, verify, batch, sampler)
        finish(records, selected)
        return selected

    eagle_utils._sample_spec_opt_out_roots = roots


def run():
    from transformers import AutoTokenizer

    root = Path(os.environ.get("PREFIX_RESULTS", "/workspace/prefix-results"))
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    prompts = [tokenizer.encode(p) for p in PROMPTS]
    prefixes = {i: prompts[i] + output for i, output in PREFIX_OUTPUTS.items()}
    sites = [[len(p) - 1, p[-1]] for p in prefixes.values()]
    (root / "inputs.json").write_text(
        json.dumps(dict(prompts=prompts, prefixes=prefixes, sites=sites))
    )
    bootstrap = root / "serve_prefix_probe.py"
    bootstrap.write_text(
        f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        "from diagnose_spec_opt_out_prefix import install_probes\ninstall_probes()\n"
        "if __name__ == '__main__':\n"
        "    from sglang.launch_server import run_server\n"
        "    from sglang.srt.server_args import prepare_server_args\n"
        "    run_server(prepare_server_args(sys.argv[1:]))\n"
    )
    url = "http://127.0.0.1:30000"
    for mode in ("target", "spec"):
        (root / "active.json").write_text(
            json.dumps(dict(name="warmup", capture=False, sites=sites))
        )
        env = dict(os.environ, PREFIX_RESULTS=str(root))
        args = [
            sys.executable,
            str(bootstrap),
            "--model-path",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "--host",
            "127.0.0.1",
            "--port",
            "30000",
            "--random-seed",
            "0",
            "--attention-backend",
            os.environ.get("PREFIX_ATTENTION", "triton"),
            "--page-size",
            os.environ.get("PREFIX_PAGE_SIZE", "16"),
            "--mem-fraction-static",
            "0.65",
            "--max-total-tokens",
            "4096",
            "--context-length",
            "1024",
            "--max-running-requests",
            "8",
            "--disable-prefill-cuda-graph",
            "--disable-decode-cuda-graph",
        ]
        if os.environ.get("PREFIX_NO_OVERLAP") == "1":
            args += ["--disable-overlap-schedule"]
        if mode == "spec":
            args += [
                "--speculative-algorithm",
                "STANDALONE",
                "--speculative-draft-model-path",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
            ]
        (root / f"{mode}-args.json").write_text(json.dumps(args))
        with open(root / f"{mode}.log", "w") as log:
            proc = subprocess.Popen(
                args, env=env, stdout=log, stderr=log, start_new_session=True
            )
            try:
                deadline = time.monotonic() + 420
                while time.monotonic() < deadline:
                    assert proc.poll() is None, f"{mode} exited"
                    try:
                        if requests.get(url + "/health", timeout=2).ok:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError(mode)

                def generate(name, inputs, limits, flags, capture):
                    assert requests.get(url + "/flush_cache", timeout=30).ok
                    (root / "active.json").write_text(
                        json.dumps(
                            dict(name=f"{mode}-{name}", capture=capture, sites=sites)
                        )
                    )
                    body = dict(
                        input_ids=inputs,
                        sampling_params=[
                            dict(
                                temperature=0,
                                ignore_eos=True,
                                max_new_tokens=n,
                                disable_speculative_decoding=flag,
                            )
                            for n, flag in zip(limits, flags)
                        ],
                        return_logprob=True,
                        top_logprobs_num=5,
                    )
                    response = requests.post(url + "/generate", json=body, timeout=180)
                    (root / f"{mode}-{name}.json").write_text(
                        json.dumps(
                            dict(
                                request=body,
                                status=response.status_code,
                                response_text=response.text,
                            )
                        )
                    )
                    assert response.ok, response.text
                    result = response.json()
                    tokens = [response_token_ids(r, n) for r, n in zip(result, limits)]
                    print(mode, name, tokens, flush=True)
                    return tokens

                flags = [True, True, False] if mode == "spec" else [False] * 3
                for name, capture in (("warm_plain", False), ("warm_probe", True)):
                    generate(name, prompts, [8, 68, 68], flags, capture)
                for i, prefix in prefixes.items():
                    generate(f"fresh-{i}", [prefix], [1], [mode == "spec"], True)
                    # Recompute all but the last known prefix token, then decode.
                    # The emitted prefix token is checked before interpreting the next score.
                    out = generate(
                        f"one-decode-{i}", [prefix[:-1]], [2], [mode == "spec"], True
                    )
                    print(
                        "PREFIX_PREDECESSOR_MATCH",
                        mode,
                        i,
                        out[0][0] == prefix[-1],
                        flush=True,
                    )
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
    print("PREFIX_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    run()
