"""Validate paged split attention with shuffled pages and mutable graph lengths."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import flashinfer
import torch


def reference(q, k, v, length, qk_scale, v_scale):
    if length == 0:
        return torch.zeros_like(q, dtype=torch.float64)
    scores = q.double().transpose(0, 1) @ k.double().T * qk_scale
    ends = length - q.shape[0] + torch.arange(q.shape[0], device=q.device)
    tokens = torch.arange(length, device=q.device)
    scores.masked_fill_(tokens[None, None, :] > ends[None, :, None], -float("inf"))
    return (scores.softmax(-1) @ (v.double() * v_scale)).transpose(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="42,1729,20261015")
    parser.add_argument("--heads", default="4,8")
    parser.add_argument("--lengths", default="2048,131072,168960")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("split_attention_check", args.kernel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    workspace = torch.zeros(256 * 1024**2, device="cuda", dtype=torch.uint8)
    counter = torch.zeros(8 * 1024**2, device="cuda", dtype=torch.uint8)
    metadata = dict(
        torch=torch.__version__,
        flashinfer=flashinfer.__version__,
        device=torch.cuda.get_device_name(),
        args=vars(args),
        kernel_md5=hashlib.md5(Path(args.kernel).read_bytes()).hexdigest(),
        script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
        mode="numerical validation; no timing claim",
    )
    seeds = [42] if args.quick else list(map(int, args.seeds.split(",")))
    heads_values = [4, 8] if args.quick else list(map(int, args.heads.split(",")))
    lengths = [2048, 4096] if args.quick else list(map(int, args.lengths.split(",")))
    for seed in seeds:
        torch.manual_seed(seed)
        for heads in heads_values:
            for length in lengths:
                pages = (length + 63) // 64
                q = torch.randn(7, heads, 256, device="cuda").to(torch.float8_e4m3fn)
                kv = tuple(
                    (torch.randn(pages, 1, 64, 256, device="cuda") * 64).to(q.dtype)
                    for _ in range(2)
                )
                table = torch.randperm(pages, device="cuda", dtype=torch.int32)[None]
                seq = torch.tensor([length], device="cuda", dtype=torch.int32)
                out = torch.empty_like(q, dtype=torch.bfloat16)
                stock = torch.empty_like(out)
                scratch = module.allocate_scratch(q, 1, 148, 7)
                qk_scale, v_scale = 1 / (16 * 64), 1 / 64

                def candidate():
                    module.split_kv_verify(
                        q,
                        kv,
                        table,
                        seq,
                        out,
                        scratch,
                        splits=148,
                        qlen=7,
                        block_m=64,
                        block_n=128,
                        qk_scale=qk_scale,
                        v_scale=v_scale,
                        num_stages=2,
                        skip_empty=True,
                        short_merge=True,
                        short_tiles=True,
                        merge_tile=64,
                    )

                candidate()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    candidate()
                graph.replay()
                flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                    q,
                    kv,
                    workspace,
                    table,
                    seq,
                    262144,
                    bmm1_scale=qk_scale,
                    bmm2_scale=v_scale,
                    out=stock,
                    q_len_per_req=7,
                    multi_ctas_kv_counter_buffer=counter,
                )
                selected = table[0].long()
                ref = reference(
                    q,
                    kv[0].index_select(0, selected).reshape(-1, 256)[:length],
                    kv[1].index_select(0, selected).reshape(-1, 256)[:length],
                    length,
                    qk_scale,
                    v_scale,
                )
                row = dict(seed=seed, heads=heads, length=length, q_len=7, batch=1)
                for label, value in (("stock", stock), ("candidate", out)):
                    error = value.double() - ref
                    row[label] = dict(
                        mean_row_rel_l2=(error.norm(dim=-1) / ref.norm(dim=-1))
                        .mean()
                        .item(),
                        max_abs=error.abs().max().item(),
                    )
                row["numerical_pass"] = (
                    row["candidate"]["mean_row_rel_l2"]
                    <= 1.05 * row["stock"]["mean_row_rel_l2"]
                    and row["candidate"]["max_abs"] <= 2 * row["stock"]["max_abs"]
                )
                expected = out.clone()
                seq.zero_()
                graph.replay()
                torch.cuda.synchronize()
                row["empty_pass"] = torch.count_nonzero(out).item() == 0
                seq.fill_(length)
                graph.replay()
                torch.cuda.synchronize()
                row["restore_pass"] = torch.equal(out, expected)
                row["verdict"] = (
                    "PASS"
                    if row["numerical_pass"]
                    and row["empty_pass"]
                    and row["restore_pass"]
                    else "FAIL"
                )
                records.append(row)
                print(json.dumps(row), flush=True)
                Path(args.output).write_text(
                    json.dumps(dict(metadata=metadata, records=records), indent=2)
                    + "\n"
                )
    assert all(row["verdict"] == "PASS" for row in records), "validation failed"


if __name__ == "__main__":
    main()
