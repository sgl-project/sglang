# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream timing/load generation; only adapt chat framing for V4.1.

The upstream parser is executed unchanged from its AST __main__ block. This
avoids copying a CLI or editing vendor code. Serial generation makes the
process-local encoder override explicit and reproducible.
"""

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path


def load_encoder(checkpoint: Path):
    path = checkpoint / "encoding/encoding.py"
    spec = importlib.util.spec_from_file_location("official_dsv41_encoding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_messages


def install_adapter(bench, encode, thinking: str, effort: int, manifest: Path | None):
    original_sample = bench.sample_random_requests

    def frame(prompt, tokenizer, dsv4):
        if dsv4:
            raise ValueError("Do not use the V4 encoder for V4.1")
        return encode(
            [{"role": "user", "content": prompt}],
            thinking_mode=thinking,
            reasoning_effort=effort,
        )

    def sample(*args, **kwargs):
        if kwargs.get("num_workers") != 1 or not kwargs.get("use_chat_template"):
            raise ValueError(
                "V4.1 adapter requires --random-num-workers 1 --use-chat-template"
            )
        requests = original_sample(*args, **kwargs)
        if manifest:
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "thinking": thinking,
                        "reasoning_effort": effort,
                        "encoder": "checkpoint/encoding/encoding.py",
                        "requests": [
                            {
                                "index": i,
                                "prompt_tokens": int(r[1]),
                                "requested_output_tokens": int(r[2]),
                                "prompt_sha256": hashlib.sha256(
                                    r[0].encode()
                                ).hexdigest(),
                            }
                            for i, r in enumerate(requests)
                        ],
                        "first_prompt_prefix": requests[0][0][:240],
                        "first_prompt_suffix": requests[0][0][-80:],
                    },
                    indent=2,
                )
                + "\n"
            )
        return requests

    bench._apply_chat_template = frame
    bench.sample_random_requests = sample


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inferencex-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thinking", choices=["thinking", "chat"], default="thinking")
    parser.add_argument("--reasoning-effort", type=int, default=100)
    parser.add_argument("--request-manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    own, upstream = parser.parse_known_args()
    if not 1 <= own.reasoning_effort <= 100:
        raise ValueError("reasoning effort must be between 1 and 100")
    sys.path.insert(0, str(own.inferencex_dir.resolve()))
    bench = importlib.import_module("infx.bench_serving.benchmark_serving")
    encode = load_encoder(own.checkpoint)
    install_adapter(
        bench, encode, own.thinking, own.reasoning_effort, own.request_manifest
    )
    if own.self_test:
        tokenizer = bench._load_tokenizer(str(own.checkpoint), "auto", True)
        for mode, suffix in [("chat", "</think>"), ("thinking", "<think>")]:
            rendered = encode(
                [{"role": "user", "content": "What is 17*19?"}],
                thinking_mode=mode,
                reasoning_effort=100,
            )
            assert rendered.startswith("<｜begin▁of▁sentence｜>")
            assert rendered.endswith("<｜Assistant｜>" + suffix)
            assert tokenizer.encode(rendered, add_special_tokens=False)[0] == 0
        bench.np.random.seed(0)
        req = bench.sample_random_requests(
            prefix_len=0,
            input_len=8192,
            output_len=1024,
            num_prompts=2,
            range_ratio=1.0,
            tokenizer=tokenizer,
            use_chat_template=True,
            dsv4=False,
            tokenizer_id=str(own.checkpoint),
            tokenizer_mode="auto",
            trust_remote_code=True,
            num_workers=1,
        )
        assert all(abs(r[1] - 8192) <= 8 and r[2] == 1024 for r in req), req
        print("PASS: official V4.1 chat/thinking framing and 8k/1k prompt generation")
        return
    source = Path(bench.__file__)
    tree = ast.parse(source.read_text())
    guards = [
        n
        for n in tree.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"
    ]
    if len(guards) != 1:
        raise RuntimeError("Upstream CLI changed; review adapter before running")
    sys.argv = [str(source), *upstream]
    exec(
        compile(ast.Module(body=guards[0].body, type_ignores=[]), str(source), "exec"),
        vars(bench),
    )


if __name__ == "__main__":
    main()
