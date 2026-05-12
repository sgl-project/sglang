"""Measure raw IPC overhead: ZMQ pickle vs queue reference passing.

This directly measures the cost we're trying to eliminate.
"""

import os
import sys
import time
import queue
import pickle
import threading
import multiprocessing as mp

import numpy as np
import zmq


def build_typical_tokenized_request():
    """Build a payload similar to TokenizedGenerateReqInput."""
    return {
        "input_ids": list(range(128)),
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 128,
            "stop": ["</s>"],
        },
        "rid": "req-0001",
        "input_text": "Hello " * 64,
        "mm_inputs": None,
    }


def build_typical_batch_output(batch_size=16):
    """Build a payload similar to BatchTokenIDOutput."""
    return {
        "rids": [f"req-{i:04d}" for i in range(batch_size)],
        "finished_reasons": [None] * batch_size,
        "decode_ids": [[42]] * batch_size,
        "output_ids": [list(range(10)) for _ in range(batch_size)],
        "read_offsets": [0] * batch_size,
        "output_strs": ["hello world "] * batch_size,
        "num_output_tokens": [10] * batch_size,
    }


def build_mm_tokenized_request():
    """Build a multimodal request with image features."""
    import torch
    return {
        "input_ids": list(range(128)),
        "sampling_params": {"temperature": 0.7, "max_new_tokens": 16},
        "rid": "req-mm-0001",
        "input_text": "Describe this image " * 6,
        "mm_inputs": {
            "pixel_values": torch.randn(8, 3, 448, 448),
            "image_sizes": [(448, 448)] * 8,
        },
    }


def bench_zmq_ipc(payload, num_iters=1000, label=""):
    """Measure ZMQ IPC round-trip: send_pyobj + recv_pyobj."""
    ctx = zmq.Context()
    sender = ctx.socket(zmq.PUSH)
    sender.bind("ipc:///tmp/sglang_bench_ipc")
    receiver = ctx.socket(zmq.PULL)
    receiver.connect("ipc:///tmp/sglang_bench_ipc")

    # warmup
    for _ in range(10):
        sender.send_pyobj(payload)
        receiver.recv_pyobj()

    start = time.perf_counter()
    for _ in range(num_iters):
        sender.send_pyobj(payload)
        receiver.recv_pyobj()
    elapsed = time.perf_counter() - start

    sender.close()
    receiver.close()
    ctx.term()
    avg_us = elapsed / num_iters * 1e6
    print(f"  [{label}] ZMQ send+recv: {avg_us:.1f}µs/op ({num_iters} ops in {elapsed*1000:.1f}ms)")
    return avg_us


def bench_zmq_cross_process(payload, num_iters=1000, label=""):
    """Measure ZMQ IPC across actual processes."""
    def receiver_proc(addr, n):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PULL)
        sock.bind(addr)
        for _ in range(n + 10):  # warmup + real
            sock.recv_pyobj()
        sock.close()
        ctx.term()

    addr = "ipc:///tmp/sglang_bench_ipc_xproc"
    p = mp.Process(target=receiver_proc, args=(addr, num_iters))
    p.start()
    time.sleep(0.5)

    ctx = zmq.Context()
    sender = ctx.socket(zmq.PUSH)
    sender.connect(addr)

    # warmup
    for _ in range(10):
        sender.send_pyobj(payload)
    time.sleep(0.1)

    start = time.perf_counter()
    for _ in range(num_iters):
        sender.send_pyobj(payload)
    elapsed = time.perf_counter() - start

    sender.close()
    ctx.term()
    p.join()

    avg_us = elapsed / num_iters * 1e6
    print(f"  [{label}] ZMQ cross-proc send: {avg_us:.1f}µs/op")
    return avg_us


def bench_queue_pass(payload, num_iters=1000, label=""):
    """Measure queue.SimpleQueue put+get (reference passing, no pickle)."""
    q = queue.SimpleQueue()

    # warmup
    for _ in range(10):
        q.put(payload)
        q.get()

    start = time.perf_counter()
    for _ in range(num_iters):
        q.put(payload)
        q.get()
    elapsed = time.perf_counter() - start

    avg_us = elapsed / num_iters * 1e6
    print(f"  [{label}] Queue put+get: {avg_us:.1f}µs/op ({num_iters} ops in {elapsed*1000:.1f}ms)")
    return avg_us


def bench_pickle_only(payload, num_iters=1000, label=""):
    """Measure just pickle.dumps + pickle.loads."""
    # warmup
    for _ in range(10):
        b = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.loads(b)

    start = time.perf_counter()
    for _ in range(num_iters):
        b = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.loads(b)
    elapsed = time.perf_counter() - start

    avg_us = elapsed / num_iters * 1e6
    size_kb = len(b) / 1024
    print(f"  [{label}] pickle dumps+loads: {avg_us:.1f}µs/op, size={size_kb:.1f}KB")
    return avg_us


def main():
    gil_status = "DISABLED" if hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled() else "ENABLED/N/A"
    print(f"Python {sys.version}")
    print(f"GIL: {gil_status}")
    print()

    text_req = build_typical_tokenized_request()
    batch_out = build_typical_batch_output(batch_size=16)

    print("=" * 60)
    print("TEXT REQUEST (TokenizedGenerateReqInput-like)")
    print("=" * 60)
    bench_pickle_only(text_req, 5000, "text-req")
    bench_zmq_ipc(text_req, 5000, "text-req")
    bench_queue_pass(text_req, 5000, "text-req")
    print()

    print("=" * 60)
    print("BATCH OUTPUT (BatchTokenIDOutput-like, bs=16)")
    print("=" * 60)
    bench_pickle_only(batch_out, 5000, "batch-out")
    bench_zmq_ipc(batch_out, 5000, "batch-out")
    bench_queue_pass(batch_out, 5000, "batch-out")
    print()

    try:
        mm_req = build_mm_tokenized_request()
        print("=" * 60)
        print("MULTIMODAL REQUEST (8 images, 448x448)")
        print("=" * 60)
        bench_pickle_only(mm_req, 500, "mm-req")
        bench_zmq_ipc(mm_req, 500, "mm-req")
        bench_queue_pass(mm_req, 500, "mm-req")
        print()
    except Exception as e:
        print(f"Skipping MM test: {e}")

    # Per-request IPC budget analysis
    print("=" * 60)
    print("PER-REQUEST IPC BUDGET ANALYSIS")
    print("=" * 60)
    print("Each request goes through 3 IPC hops:")
    print("  1. tokenizer → scheduler (send request)")
    print("  2. scheduler → detokenizer (per decode step)")
    print("  3. detokenizer → tokenizer (per decode step)")
    print()
    print("For output_len=16:")

    zmq_text = bench_zmq_ipc(text_req, 2000, "hop1-text")
    zmq_batch = bench_zmq_ipc(batch_out, 2000, "hop2+3-batch")
    q_text = bench_queue_pass(text_req, 2000, "hop1-text")
    q_batch = bench_queue_pass(batch_out, 2000, "hop2+3-batch")

    zmq_total = zmq_text + zmq_batch * 16 * 2  # 16 decode steps, 2 hops each
    q_total = q_text + q_batch * 16 * 2
    saved = zmq_total - q_total

    print(f"\n  ZMQ total per request:   {zmq_total/1000:.2f}ms")
    print(f"  Queue total per request: {q_total/1000:.2f}ms")
    print(f"  Saved per request:       {saved/1000:.2f}ms")
    print(f"  At QPS=32: saved/s =     {saved * 32 / 1000:.1f}ms/s")


if __name__ == "__main__":
    main()
