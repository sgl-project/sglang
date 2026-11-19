"""Generate a TensorRT-LLM py_executor override for stable torch-profiler capture."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

START_MARKER = "torch_profiler = torch.profiler.profile("


@dataclass
class ProfileCallSpan:
    start: int
    end: int
    block: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a py_executor.py override that enables with_stack=True for "
            "TensorRT-LLM torch-profiler traces."
        )
    )
    parser.add_argument("--source", required=True, help="Original py_executor.py path.")
    parser.add_argument("--output", required=True, help="Override file path to write.")
    return parser.parse_args()


def find_profile_call_span(text: str) -> ProfileCallSpan:
    start = text.find(START_MARKER)
    if start == -1:
        raise SystemExit("Could not find torch profiler setup in source file.")

    open_paren = text.find("(", start)
    if open_paren == -1:
        raise SystemExit("Malformed torch profiler setup in source file.")

    depth = 0
    for index in range(open_paren, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return ProfileCallSpan(
                    start=start,
                    end=index + 1,
                    block=text[start : index + 1],
                )
    raise SystemExit("Could not find the end of the torch profiler call.")


def inject_with_stack(block: str) -> str:
    if "with_stack=" in block:
        return block

    lines = block.splitlines()
    if not lines:
        raise SystemExit("Unexpected torch profiler block format.")

    last_line = lines[-1]
    if not last_line.strip():
        raise SystemExit("Unexpected torch profiler block terminator.")

    if last_line.strip() == ")":
        if len(lines) < 2:
            raise SystemExit("Could not find the last torch profiler argument line.")
        last_arg_index = len(lines) - 2
        last_arg_line = lines[last_arg_index]
        indent = last_arg_line[: len(last_arg_line) - len(last_arg_line.lstrip())]
        if not last_arg_line.rstrip().endswith(","):
            lines[last_arg_index] = last_arg_line.rstrip() + ","
        lines.insert(len(lines) - 1, f"{indent}with_stack=True")
        return "\n".join(lines)

    if not last_line.rstrip().endswith(")"):
        raise SystemExit("Unexpected torch profiler block terminator.")

    indent = last_line[: len(last_line) - len(last_line.lstrip())]
    last_arg_text = last_line.rstrip()[:-1].rstrip()
    if not last_arg_text.endswith(","):
        last_arg_text += ","
    lines[-1] = last_arg_text
    lines.append(f"{indent}with_stack=True)")
    return "\n".join(lines)


def inject_rank0_trace_guard(text: str) -> str:
    needle = (
        "        enable_torch_trace = bool(torch_trace_path and profile_start_stop)\n"
    )
    replacement = (
        "        # Multi-rank PyTorch backend workers race on the same chrome-trace "
        "path.\n"
        "        # Keep the full torch-profiler trace on rank 0 and let the other "
        "ranks\n"
        "        # continue with CUDA-profiler gating only.\n"
        "        enable_torch_trace = bool(\n"
        "            torch_trace_path and profile_start_stop and self.dist.rank == 0\n"
        "        )\n"
    )
    if replacement in text:
        return text
    if needle not in text:
        raise SystemExit("Could not find enable_torch_trace assignment in source file.")
    return text.replace(needle, replacement, 1)


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    text = source.read_text(encoding="utf-8")
    span = find_profile_call_span(text)
    patched_block = inject_with_stack(span.block)
    patched = (
        text
        if patched_block == span.block
        else (text[: span.start] + patched_block + text[span.end :])
    )
    patched = inject_rank0_trace_guard(patched)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(patched, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
