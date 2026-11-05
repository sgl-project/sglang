"""Native BF16 Qwen4 same-slot A/B/A replacement and optional GPU fault injection."""

import argparse
import json
import os
import re
import shutil
import signal
from pathlib import Path

import psutil
import torch
from safetensors.torch import load_file
from test_qwen_moe_expert_profile import PROMPTS, compare
from test_qwen_moe_saliency import inspect_fixture

import sglang as sgl


def scheduler_pids():
    pids = []
    for child in psutil.Process().children(recursive=True):
        try:
            if child.status() != psutil.STATUS_ZOMBIE and child.name().startswith(
                "sglang::sched"
            ):
                pids.append(child.pid)
        except psutil.NoSuchProcess:
            continue
    return sorted(pids)


# Spawned scheduler processes re-import this main file. Only the last TP rank fails once.
if os.environ.get("SGLANG_TEST_HOT_SWAP_WRITE_FAILURE") == "1":
    from sglang.srt.eplb.expert_swap_weights import ExpertRowPatch
    from sglang.srt.runtime_context import get_parallel

    _copy_new = ExpertRowPatch.copy_new
    _fail_next_write = True

    def _injected_copy(self):
        global _fail_next_write
        if (
            _fail_next_write
            and get_parallel().moe_tp_rank == get_parallel().moe_tp_size - 1
        ):
            _fail_next_write = False
            self.module.w13_weight.data[self.slot].fill_(123)
            raise RuntimeError("injected partial expert row write")
        return _copy_new(self)

    ExpertRowPatch.copy_new = _injected_copy


def generate(engine):
    return [
        engine.generate(
            input_ids=prompt,
            sampling_params={"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
            return_logprob=True,
            logprob_start_len=0,
        )
        for prompt in PROMPTS
    ]


def export(engine, folder, tp_size):
    folder.mkdir()
    engine.save_sharded_model(
        path=str(folder), pattern="rank-{rank}.safetensors", max_size=None
    )
    return [
        load_file(str(folder / f"rank-{rank}.safetensors")) for rank in range(tp_size)
    ]


def check_weights(
    actual, reference, actual_map, reference_map, previous=None, old_map=None
):
    changed = 0
    for rank, (left, right) in enumerate(zip(actual, reference, strict=True)):
        assert set(left) == set(right)
        for name, tensor in left.items():
            match = re.search(
                r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(w13_weight|w2_weight)$", name
            )
            expected = right[name]
            if match:
                layer = int(match.group(1))
                permutation = [
                    reference_map[layer].index(expert) for expert in actual_map[layer]
                ]
                expected = expected[permutation]
                if previous is not None:
                    for slot, expert in enumerate(actual_map[layer]):
                        if expert in old_map[layer]:
                            assert old_map[layer][slot] == expert
                            torch.testing.assert_close(
                                tensor[slot], previous[rank][name][slot], rtol=0, atol=0
                            )
                        elif not torch.equal(tensor[slot], previous[rank][name][slot]):
                            changed += 1
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    if previous is not None:
        assert changed > 0, "The test must observe actual expert-weight replacement"
    return changed


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, choices=(1, 4), default=4)
    parser.add_argument("--port", type=int, default=31100)
    parser.add_argument("--decode-graph", action="store_true")
    parser.add_argument("--fault-prepare", action="store_true")
    parser.add_argument("--fault-write", action="store_true")
    args = parser.parse_args()
    source = inspect_fixture(args.model_path)
    config = json.loads((args.model_path / "config.json").read_text())
    if (
        config["model_type"] != "qwen4_exp"
        or source["num_experts"] < 8
        or source["top_k"] > 4
    ):
        parser.error(
            "Use a tiny fused Qwen4 fixture with >=8 experts and token Top-K <=4"
        )
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    # Own the source copy so a prepare fault never renames files used by another run.
    model = args.work_dir / "model"
    model.mkdir()
    for relative in source["files"]:
        destination = model / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source["path"]) / relative, destination)
    (args.work_dir / "model-source.json").write_text(json.dumps(source, indent=2))
    expert_count = source["num_experts"]
    initial = [
        [(offset + layer) % expert_count for offset in (7, 1, 5, 3)]
        for layer in range(source["num_layers"])
    ]
    target = []
    for row in initial:
        missing = [expert for expert in range(expert_count) if expert not in row]
        target.append([row[2], missing[0], row[0], missing[1]])
    for label, keep in (("a", initial), ("b", target)):
        (args.work_dir / f"profile-{label}.json").write_text(
            json.dumps({"num_experts": expert_count, "keep": keep})
        )
    common = dict(
        model_path=str(model),
        tp_size=args.tp_size,
        port=args.port,
        dtype="bfloat16",
        attention_backend="fa3",
        moe_runner_backend="triton",
        skip_tokenizer_init=True,
        context_length=512,
        max_total_tokens=1024,
        max_running_requests=2,
        chunked_prefill_size=256,
        mem_fraction_static=0.15,
        cuda_graph_backend_decode="full" if args.decode_graph else "disabled",
        cuda_graph_backend_prefill="disabled",
        cuda_graph_max_bs_decode=2,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        ple_offload_embedding=False,
        disable_shared_experts_fusion=True,
        random_seed=417,
    )
    references, reference_weights = {}, {}
    for label in ("a", "b"):
        engine = sgl.Engine(
            **common, moe_expert_profile=str(args.work_dir / f"profile-{label}.json")
        )
        try:
            references[label] = generate(engine)
            reference_weights[label] = export(
                engine, args.work_dir / f"reference-{label}-weights", args.tp_size
            )
        finally:
            engine.shutdown()
    old_fault = os.environ.get("SGLANG_TEST_HOT_SWAP_WRITE_FAILURE")
    if args.fault_write:
        os.environ["SGLANG_TEST_HOT_SWAP_WRITE_FAILURE"] = "1"
    try:
        engine = sgl.Engine(
            **common, moe_expert_profile=str(args.work_dir / "profile-a.json")
        )
    finally:
        if old_fault is None:
            os.environ.pop("SGLANG_TEST_HOT_SWAP_WRITE_FAILURE", None)
        else:
            os.environ["SGLANG_TEST_HOT_SWAP_WRITE_FAILURE"] = old_fault
    result = {
        "status": "running",
        "tp_size": args.tp_size,
        "decode_graph": args.decode_graph,
        "source_sha256": source["sha256"],
    }
    try:
        result["before"] = compare(generate(engine), references["a"], 0.05)
        result["scheduler_pids_before"] = scheduler_pids()
        assert len(result["scheduler_pids_before"]) == args.tp_size, result
        before_weights = export(engine, args.work_dir / "before-weights", args.tp_size)
        if args.fault_prepare:
            renamed = []
            try:
                for filename in model.glob("*.safetensors"):
                    missing = filename.with_suffix(".missing")
                    filename.rename(missing)
                    renamed.append((missing, filename))
                try:
                    engine.hot_swap_expert_set(target, memory_reserve_bytes=0)
                except AssertionError as error:
                    assert "old weights retained" in str(error), error
                else:
                    raise AssertionError("Prepare fault did not reject the update")
            finally:
                for missing, filename in renamed:
                    missing.rename(filename)
            result["after_prepare_failure"] = compare(
                generate(engine), references["a"], 0.05
            )
        if args.fault_write:
            try:
                engine.hot_swap_expert_set(target, memory_reserve_bytes=0)
            except AssertionError as error:
                assert "rolled back" in str(error), error
            else:
                raise AssertionError("Write fault did not reject the update")
            result["after_write_failure"] = compare(
                generate(engine), references["a"], 0.05
            )
            restored = export(engine, args.work_dir / "rollback-weights", args.tp_size)
            check_weights(restored, before_weights, initial, initial)
        result["a_to_b"] = engine.hot_swap_expert_set(
            target, max_node_staging_bytes=128 * 2**20, memory_reserve_bytes=0
        )
        result["b_output"] = compare(generate(engine), references["b"], 0.05)
        actual = result["a_to_b"]["physical_to_logical_map"]
        after_b = export(engine, args.work_dir / "after-b-weights", args.tp_size)
        result["changed_weight_rows"] = check_weights(
            after_b, reference_weights["b"], actual, target, before_weights, initial
        )
        result["b_to_a"] = engine.hot_swap_expert_set(
            initial, max_node_staging_bytes=128 * 2**20, memory_reserve_bytes=0
        )
        result["a_output"] = compare(generate(engine), references["a"], 0.05)
        after_a = export(engine, args.work_dir / "after-a-weights", args.tp_size)
        check_weights(
            after_a,
            reference_weights["a"],
            result["b_to_a"]["physical_to_logical_map"],
            initial,
        )
        result["scheduler_pids_after"] = scheduler_pids()
        assert result["scheduler_pids_after"] == result["scheduler_pids_before"], result
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        engine.shutdown()
        (args.work_dir / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
