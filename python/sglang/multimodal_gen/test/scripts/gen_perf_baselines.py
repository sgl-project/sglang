import argparse
import inspect
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

from sglang.multimodal_gen.test.server.test_server_utils import (
    ServerManager,
    WarmupRunner,
    download_image_from_url,
    get_generate_fn,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    BASELINE_CONFIG,
    DiffusionTestCase,
)
from sglang.multimodal_gen.test.test_utils import (
    get_dynamic_server_port,
    is_image_url,
    wait_for_req_perf_record,
)


def _all_cases() -> list[DiffusionTestCase]:
    import sglang.multimodal_gen.test.server.testcase_configs as cfg

    cases: list[DiffusionTestCase] = []
    for _, v in inspect.getmembers(cfg):
        if isinstance(v, list) and v and isinstance(v[0], DiffusionTestCase):
            cases.extend(v)

    seen: set[str] = set()
    out: list[DiffusionTestCase] = []
    for c in cases:
        if c.id not in seen:
            seen.add(c.id)
            out.append(c)
    return out


def _baseline_path() -> Path:
    import sglang.multimodal_gen.test.server.testcase_configs as cfg

    return Path(cfg.__file__).with_name("perf_baselines.json")


def _openai_client(port: int) -> OpenAI:
    return OpenAI(api_key="sglang-anything", base_url=f"http://localhost:{port}/v1")


def _build_server_extra_args(case: DiffusionTestCase) -> str:
    a = os.environ.get("SGLANG_TEST_SERVE_ARGS", "")
    a += f" --num-gpus {case.server_args.num_gpus}"
    if case.server_args.tp_size is not None:
        a += f" --tp-size {case.server_args.tp_size}"
    if case.server_args.ulysses_degree is not None:
        a += f" --ulysses-degree {case.server_args.ulysses_degree}"
    if case.server_args.dit_layerwise_offload:
        a += " --dit-layerwise-offload true"
    if case.server_args.ring_degree is not None:
        a += f" --ring-degree {case.server_args.ring_degree}"
    if case.server_args.lora_path:
        a += f" --lora-path {case.server_args.lora_path}"
    if case.server_args.enable_warmup:
        a += " --enable-warmup"
    return a


def _build_env_vars(case: DiffusionTestCase) -> dict[str, str]:
    if case.server_args.enable_cache_dit:
        return {"SGLANG_CACHE_DIT_ENABLED": "true"}
    return {}


def _torch_cleanup() -> None:
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _run_case(case: DiffusionTestCase) -> dict:
    default_port = get_dynamic_server_port()
    port = int(os.environ.get("SGLANG_TEST_SERVER_PORT", default_port))
    mgr = ServerManager(
        model=case.server_args.model_path,
        port=port,
        wait_deadline=float(os.environ.get("SGLANG_TEST_WAIT_SECS", "1200")),
        extra_args=_build_server_extra_args(case),
        env_vars=_build_env_vars(case),
    )
    ctx = mgr.start()
    try:
        sp = case.sampling_params
        output_size = os.environ.get("SGLANG_TEST_OUTPUT_SIZE", sp.output_size)
        w = WarmupRunner(
            port=ctx.port,
            model=case.server_args.model_path,
            prompt=sp.prompt or "A colorful raccoon icon",
            output_size=output_size,
            output_format=sp.output_format,
        )
        if case.server_args.warmup > 0:
            if sp.image_path and sp.prompt:
                image_path_list = sp.image_path
                if not isinstance(image_path_list, list):
                    image_path_list = [image_path_list]
                new_image_path_list = []
                for p in image_path_list:
                    if is_image_url(p):
                        new_image_path_list.append(download_image_from_url(str(p)))
                    else:
                        pp = Path(p)
                        if not pp.exists():
                            raise FileNotFoundError(str(pp))
                        new_image_path_list.append(pp)
                w.run_edit_warmups(
                    count=case.server_args.warmup,
                    edit_prompt=sp.prompt,
                    image_path=new_image_path_list,
                )
            else:
                w.run_text_warmups(case.server_args.warmup)

        client = _openai_client(ctx.port)
        gen = get_generate_fn(
            model_path=case.server_args.model_path,
            modality=case.server_args.modality,
            sampling_params=sp,
        )
        rid = gen(case.id, client)
        rec = wait_for_req_perf_record(
            rid,
            ctx.perf_log_path,
            timeout=float(os.environ.get("SGLANG_PERF_TIMEOUT", "300")),
        )
        if rec is None:
            raise RuntimeError(f"missing perf record: {case.id}")
        from sglang.multimodal_gen.test.server.testcase_configs import (
            PerformanceSummary,
        )

        perf = PerformanceSummary.from_req_perf_record(
            rec, BASELINE_CONFIG.step_fractions
        )
        if case.server_args.modality == "video" and sp.num_frames and sp.num_frames > 0:
            if "per_frame_generation" not in perf.stage_metrics:
                perf.stage_metrics["per_frame_generation"] = perf.e2e_ms / sp.num_frames

        return {
            "stages_ms": {k: round(v, 2) for k, v in perf.stage_metrics.items()},
            "denoise_step_ms": {
                str(k): round(v, 2) for k, v in perf.all_denoise_steps.items()
            },
            "expected_e2e_ms": round(perf.e2e_ms, 2),
            "expected_avg_denoise_ms": round(perf.avg_denoise_ms, 2),
            "expected_median_denoise_ms": round(perf.median_denoise_ms, 2),
        }
    finally:
        ctx.cleanup()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--match", default="")
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--all-from-baseline", action="store_true")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    os.environ.setdefault("SGLANG_GEN_BASELINE", "1")
    os.environ["SGLANG_PERF_TIMEOUT"] = str(args.timeout)

    baseline_path = Path(args.baseline) if args.baseline else _baseline_path()
    out_path = Path(args.out) if args.out else baseline_path
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    scenarios = data.setdefault("scenarios", {})

    ids = set(args.case) if args.case else None
    pat = re.compile(args.match) if args.match else None
    if args.all_from_baseline:
        ids = set(scenarios.keys())
        pat = None

    all_cases = _all_cases()
    cases = []
    for c in all_cases:
        if ids and c.id not in ids:
            continue
        if pat and not pat.search(c.id):
            continue
        cases.append(c)

    if args.all_from_baseline and ids:
        case_ids = {c.id for c in all_cases}
        missing = sorted([i for i in ids if i not in case_ids])
        if missing:
            sys.stderr.write(f"missing cases in testcase_configs.py: {len(missing)}\n")

    if not cases:
        return 0

    for c in cases:
        prev = scenarios.get(c.id, {})
        note = prev.get("notes")
        baseline = _run_case(c)
        if note is not None:
            baseline["notes"] = note
        scenarios[c.id] = baseline
        sys.stdout.write(f"{c.id}\n")
        sys.stdout.flush()
        _torch_cleanup()

    out_path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
