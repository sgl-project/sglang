# SPDX-License-Identifier: Apache-2.0
"""Compare exported state from two weight-cache daemon groups.

Pair corresponding ranks with the same parallel layout to validate transferred
weights against an independently disk-loaded reference. Both groups must be ready.
"""

import argparse
import socket

import torch
from sglang.srt.platforms import current_platform

from sglang.srt.weight_cache.protocol import (
    get_socket_path,
    recv_msg,
    send_msg,
)
from sglang.srt.weight_cache.transport import (
    TORCH_IPC_BACKEND,
    get_client_transport_backend,
)


def fetch_state(gpu_id: int):
    socket_path = get_socket_path(current_platform.get_device_uuid(gpu_id))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(60)
        client.connect(socket_path)
        send_msg(client, {"type": "query_config"})
        config_response = recv_msg(client)
    if config_response.get("status") != "ok":
        raise RuntimeError(f"query_config failed for {socket_path}: {config_response}")

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(60)
        client.connect(socket_path)
        send_msg(
            client,
            {
                "type": "fetch_state",
                "config": config_response["config"],
            },
        )
        response = recv_msg(client)
        backend = get_client_transport_backend(
            response.get("transport_backend", TORCH_IPC_BACKEND)
        )
        response = backend.recv_fetch_state_response(client, response)
    if response.get("status") != "ok":
        raise RuntimeError(f"fetch_state failed for {socket_path}: {response}")
    return response["entries"], backend


def compare_rank(left_gpu_id: int, right_gpu_id: int, rank: int) -> tuple[int, int]:
    left, left_backend = fetch_state(left_gpu_id)
    right, right_backend = fetch_state(right_gpu_id)
    if set(left) != set(right):
        raise AssertionError(
            f"rank {rank} state keys differ: "
            f"left_only={sorted(set(left) - set(right))[:10]}, "
            f"right_only={sorted(set(right) - set(left))[:10]}"
        )

    compared_bytes = 0
    for name in sorted(left):
        left_entry = left[name]
        right_entry = right[name]
        for field in ("is_plain_value", "is_plain_tensor"):
            if left_entry.get(field, False) != right_entry.get(field, False):
                raise AssertionError(f"rank {rank} {name} metadata {field} differs")
        if left_entry.get("is_plain_value", False):
            left_value, right_value = left_entry["value"], right_entry["value"]
            if type(left_value) is not type(right_value) or left_value != right_value:
                raise AssertionError(f"rank {rank} {name} derived values differ")
            continue
        for field in ("shape", "dtype", "is_param"):
            if left_entry[field] != right_entry[field]:
                raise AssertionError(
                    f"rank {rank} {name} metadata {field} differs: "
                    f"{left_entry[field]!r} != {right_entry[field]!r}"
                )
        left_tensor = left_backend.import_tensor(left_entry)
        right_tensor = right_backend.import_tensor(right_entry)
        right_on_left = right_tensor.to(left_tensor.device)
        if not torch.equal(left_tensor, right_on_left):
            detail = "values differ"
            if left_tensor.is_floating_point():
                max_abs = (left_tensor.float() - right_on_left.float()).abs().max()
                detail += f", max_abs={max_abs.item()}"
            raise AssertionError(f"rank {rank} tensor {name} {detail}")
        compared_bytes += left_tensor.numel() * left_tensor.element_size()
        del right_on_left, right_tensor, left_tensor
    return len(left), compared_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-base-gpu-id", type=int, required=True)
    parser.add_argument("--right-base-gpu-id", type=int, required=True)
    parser.add_argument("--gpu-id-step", type=int, default=1)
    parser.add_argument(
        "--num-ranks",
        type=int,
        required=True,
        help="Number of paired daemon ranks (TP * PP).",
    )
    args = parser.parse_args()

    total_entries = total_bytes = 0
    for rank in range(args.num_ranks):
        entries, nbytes = compare_rank(
            args.left_base_gpu_id + rank * args.gpu_id_step,
            args.right_base_gpu_id + rank * args.gpu_id_step,
            rank,
        )
        total_entries += entries
        total_bytes += nbytes
        print(f"rank {rank}: exactly matched {entries} state entries ({nbytes} bytes)")
    print(f"PASS: exactly matched {total_entries} rank-state entries ({total_bytes} bytes)")


if __name__ == "__main__":
    main()
