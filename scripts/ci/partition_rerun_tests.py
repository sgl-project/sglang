import ast
import json
import math
import os
from pathlib import Path

# Run test has a 60-minute timeout; reserve a third for runtime variance/retries.
PARTITION_SECONDS = 40 * 60


def estimate_seconds(command, root, mode):
    filename = command.split()[0].split("::", 1)[0]
    path = root / filename if mode == "multimodal_gen" else root / "test" / filename
    register_name = {"cuda": "register_cuda_ci", "cpu": "register_cpu_ci"}.get(mode)
    if register_name is None:
        return None
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return None
    estimates = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == register_name
        ):
            continue
        value = next(
            (kw.value for kw in node.keywords if kw.arg == "est_time"),
            node.args[0] if node.args else None,
        )
        try:
            estimate = ast.literal_eval(value)
        except (ValueError, TypeError):
            return None
        if (
            type(estimate) not in (int, float)
            or not math.isfinite(estimate)
            or estimate <= 0
        ):
            return None
        estimates.append(estimate)
    # A file may register on multiple pools; use the largest estimate conservatively.
    return max(estimates) if estimates else None


def partition_commands(commands, root, mode):
    partitions, totals = [], []
    for command in commands.splitlines():
        command = command.strip()
        if not command:
            continue
        estimate = estimate_seconds(command, root, mode)
        if estimate is None or estimate > PARTITION_SECONDS:
            partitions.append([command])
            totals.append(PARTITION_SECONDS)
            continue
        for idx, total in enumerate(totals):
            if total + estimate <= PARTITION_SECONDS:
                partitions[idx].append(command)
                totals[idx] += estimate
                break
        else:
            partitions.append([command])
            totals.append(estimate)
    if not partitions:
        raise ValueError("No test commands supplied")
    if len(partitions) > 256:
        raise ValueError("Test commands exceed the 256-job matrix limit")
    return {
        "include": [
            {"partition": idx + 1, "test_command": "\n".join(batch)}
            for idx, batch in enumerate(partitions)
        ]
    }


if __name__ == "__main__":
    matrix = partition_commands(
        os.environ["TEST_COMMAND"], Path(os.environ["TEST_ROOT"]), os.environ["MODE"]
    )
    for partition in matrix["include"]:
        print(f"Partition {partition['partition']}:\n{partition['test_command']}")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"matrix={json.dumps(matrix, separators=(',', ':'))}\n")
