"""Run the full PD/MTP GSM8K acceptance three times per NVFP4 configuration."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("baseline", "layer_split")
POOL_MARKER = "DSA NVFP4 pool bytes: "


def compare_results(runs, allocations):
    """Require all six full runs and all four prefill target-pool records."""
    medians = {}
    checks = {"accuracy": True, "metrics": True, "memory": True}
    memory = {}
    for phase in PHASES:
        samples = runs.get(phase, [])
        checks["accuracy"] &= len(samples) == 3 and all(
            sample.get("score", 0) >= 0.935 for sample in samples
        )
        valid = len(samples) == 3 and all(
            math.isfinite(sample.get(key, 0)) and sample.get(key, 0) > 0
            for sample in samples
            for key in ("latency", "output_throughput")
        )
        checks["metrics"] &= valid
        if valid:
            medians[phase] = {
                key: statistics.median(float(sample[key]) for sample in samples)
                for key in ("latency", "output_throughput")
            }
        records = allocations.get(phase, [])
        target_layers = max((record["layers"] for record in records), default=0)
        targets = {
            record["device"]: record
            for record in records
            if record["layers"] == target_layers
        }
        checks["memory"] &= len(targets) == 4 and target_layers > 1
        for record in targets.values():
            checks["memory"] &= record["total_bytes"] == sum(
                record[key]
                for key in (
                    "main_bytes",
                    "index_bytes",
                    "remote_main_bytes",
                    "remote_index_bytes",
                    "global_scale_bytes",
                )
            )
            checks["memory"] &= record["layer_split"] == (phase == "layer_split")
        memory[phase] = targets
    if checks["memory"]:
        before, after = memory["baseline"], memory["layer_split"]
        checks["memory"] &= before.keys() == after.keys() and all(
            after[device]["capacity_tokens"] == before[device]["capacity_tokens"]
            and after[device]["layers"] == before[device]["layers"]
            and after[device]["total_bytes"] < before[device]["total_bytes"]
            for device in before
        )
    checks["throughput"] = checks["metrics"] and (
        medians["layer_split"]["output_throughput"]
        >= 0.95 * medians["baseline"]["output_throughput"]
    )
    checks["latency"] = checks["metrics"] and (
        medians["layer_split"]["latency"] <= 1.05 * medians["baseline"]["latency"]
    )
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "medians": medians,
        "prefill_target_pools": memory,
    }


def _read_allocations(output_dir, phase):
    records = []
    for stream in ("stdout", "stderr"):
        path = output_dir / f"{phase}.prefill.{stream}.log"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if POOL_MARKER in line:
                # Log prefixes can include rank labels and terminal colors.
                payload = line.split(POOL_MARKER, 1)[1]
                record, _ = json.JSONDecoder().raw_decode(payload)
                records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-total-tokens", type=int, default=131072)
    args = parser.parse_args()
    if args.max_total_tokens <= 0 or args.max_total_tokens % 64:
        parser.error("--max-total-tokens must be a positive multiple of 64")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty; existing results are preserved")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_root = str(ROOT / "python")
    sys.path.insert(0, source_root)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [source_root, *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    report = {
        "status": "running",
        "stage": "preflight",
        "runs": {},
        "allocations": {},
        "model": args.model_path,
        "max_total_tokens": args.max_total_tokens,
    }

    def save_report():
        temporary = args.output_dir / "results.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, default=str) + "\n")
        temporary.replace(args.output_dir / "results.json")

    save_report()
    try:
        import requests
        import torch

        from sglang.test.kits.eval_accuracy_kit import prepare_full_gsm8k_data
        from sglang.test.run_eval import run_eval
        from sglang.test.server_fixtures.disaggregation_fixture import (
            PDDisaggregationServerBase,
        )

        if torch.cuda.device_count() != 8 or any(
            "B300" not in torch.cuda.get_device_name(i) for i in range(8)
        ):
            raise RuntimeError("Acceptance requires exactly eight visible B300 GPUs")
        from sgl_kernel import flashmla_nvfp4_ops  # noqa: F401

        if not hasattr(torch.ops.sgl_kernel, "sparse_decode_fwd_nvfp4"):
            raise RuntimeError("Rebuild flashmla_nvfp4_ops with sparse_decode_fwd_nvfp4")
        report["versions"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(8)],
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        }
        (args.output_dir / "changes.diff").write_text(
            subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT, text=True)
        )
        path = (
            ROOT
            / "test/registered/e2e/models/test_dsa_glm52_pd_mtp_cp_layersplit_nvfp4.py"
        )
        # A working tree can contain new files that git diff does not include.
        # Keep the two acceptance entry points and hash the runtime sources.
        for source in (Path(__file__), path):
            (args.output_dir / source.name).write_bytes(source.read_bytes())
        report["source_sha256"] = {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "python/sglang/srt/arg_groups/overrides.py",
                "python/sglang/srt/mem_cache/memory_pool.py",
                "python/sglang/srt/mem_cache/dsa_cache_layer_split.py",
                "python/sglang/srt/model_executor/pool_configurator.py",
                "python/sglang/test/kits/eval_accuracy_kit.py",
                "scripts/bench_glm52_nvfp4_layersplit.py",
                "test/registered/e2e/models/test_dsa_glm52_pd_mtp_cp_layersplit_nvfp4.py",
            )
        }
        spec = importlib.util.spec_from_file_location("glm52_nvfp4_acceptance", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        base = module.TestGLM52NVFP4KVLayerSplit
        dataset = prepare_full_gsm8k_data(args.output_dir / "full_gsm8k.jsonl")
        report["dataset"] = {
            "path": dataset,
            "sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
            "train_demonstrations": 20,
            "test_examples": 1319,
        }

        def make_case(phase):
            prefill_args = [
                flag
                for flag in base.extra_prefill_args
                if flag != "--enable-dsa-cache-layer-split"
            ]
            if phase == "layer_split":
                prefill_args.append("--enable-dsa-cache-layer-split")
            capacity = ["--max-total-tokens", str(args.max_total_tokens)]
            report[phase + "_config"] = {
                "prefill_args": prefill_args + capacity,
                "decode_args": base.extra_decode_args + capacity,
                "prefill_tp_size": 4,
                "decode_tp_size": 4,
                "decode_base_gpu_id": 4,
                "transfer_backend": "mooncake",
                "gsm8k": {
                    "questions": 1319,
                    "shots": 20,
                    "threads": 200,
                    "max_tokens": 512,
                    "accuracy_threshold": 0.935,
                },
            }

            class Acceptance(base):
                model = args.model_path
                extra_prefill_args = prefill_args + capacity
                extra_decode_args = base.extra_decode_args + capacity

                @classmethod
                def setUpClass(cls):
                    # Use file sinks so logs survive setup failures and teardown.
                    PDDisaggregationServerBase.setUpClass.__func__(cls)
                    for side in ("prefill", "decode"):
                        for stream in ("stdout", "stderr"):
                            attr = f"_{side}_{stream}_buf"
                            getattr(cls, attr).close()
                            setattr(
                                cls,
                                attr,
                                (
                                    args.output_dir / (f"{phase}.{side}.{stream}.log")
                                ).open("w", buffering=1),
                            )
                    cls.transfer_backend = [
                        "--disaggregation-transfer-backend",
                        "mooncake",
                    ]
                    try:
                        cls.launch_all()
                    except BaseException:
                        cls.tearDownClass()
                        raise

                def test_gsm8k(self):
                    report["runs"][phase] = []
                    for _ in range(3):
                        # Flush both workers directly. Router cache endpoints
                        # differ across versions and must not change run state.
                        for url in (self.prefill_url, self.decode_url):
                            requests.post(
                                url + "/flush_cache",
                                params={"timeout": 60},
                                timeout=65,
                            ).raise_for_status()
                        metrics = run_eval(
                            SimpleNamespace(
                                base_url=self.base_url,
                                model=self.model,
                                eval_name="gsm8k",
                                num_examples=1319,
                                num_threads=200,
                                api="completion",
                                max_tokens=512,
                                num_shots=20,
                                temperature=0,
                                gsm8k_data_path=dataset,
                            )
                        )
                        report["runs"][phase].append(metrics)
                        save_report()
                        self.assertGreaterEqual(metrics["score"], 0.935)

            return Acceptance

        report["stage"] = "golden_acceptance"
        for phase in PHASES:
            case = make_case(phase)
            save_report()
            with (args.output_dir / f"{phase}.unittest.log").open("w") as log:
                outcome = unittest.TextTestRunner(stream=log, verbosity=2).run(
                    unittest.defaultTestLoader.loadTestsFromTestCase(case)
                )
            report["allocations"][phase] = _read_allocations(args.output_dir, phase)
            save_report()
            if not outcome.wasSuccessful():
                raise RuntimeError(
                    f"{phase} GSM8K failed; see its unittest and server logs"
                )
        report["comparison"] = compare_results(report["runs"], report["allocations"])
        report["status"] = "passed" if report["comparison"]["passed"] else "failed"
    except Exception as error:
        report["status"] = "blocked" if report["stage"] == "preflight" else "failed"
        report["error"] = str(error)
        raise
    finally:
        save_report()
    print(json.dumps(report["comparison"], indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
