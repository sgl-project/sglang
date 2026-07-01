"""User-facing auto-tuning entrypoint for SGLang kernels."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

DTYPE_CHOICES = ("auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8", "int4_w4a16")
DEFAULT_BATCH_SIZES = [
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
]
DEFAULT_OUTPUT_DIR = "sglang_moe_configs"


@dataclass(frozen=True)
class RuntimeInfo:
    device_name: str
    torch_version: str
    cuda_version: str
    triton_version: str


@dataclass(frozen=True)
class AutoTunePlan:
    model_path: str
    architecture: str
    tp_size: int
    ep_size: int
    dtype: str
    model_dtype: str
    batch_sizes: List[int]
    num_experts: int
    topk: int
    hidden_size: int
    shard_intermediate_size: int
    block_shape: Optional[List[int]]
    per_channel_quant: bool
    filename: str
    output_root: Path
    output_file: Path
    quick: bool
    search_space_size: Optional[int]
    runtime: RuntimeInfo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-tune SGLang kernels for a model. The MVP supports the "
            "Triton fused MoE kernel."
        )
    )
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        required=True,
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--tp-size",
        "--tp",
        dest="tp_size",
        type=int,
        default=1,
        help="Tensor parallel size.",
    )
    parser.add_argument(
        "--ep-size",
        dest="ep_size",
        type=int,
        default=1,
        help="Expert parallel size.",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_CHOICES,
        default="auto",
        help="Kernel dtype mode. 'auto' uses the model config dtype.",
    )
    parser.add_argument(
        "--batch-size",
        action="append",
        default=None,
        help=(
            "Batch size to tune. May be repeated or comma-separated, e.g. "
            "--batch-size 1,8 --batch-size 32."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Root directory for generated configs. The CLI writes under "
            "<output-dir>/configs/triton_<version>/."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced candidate search space for a faster tuning pass.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the model/kernel tuning plan without running tuning.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-channel-quant", action="store_true")
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    parser.add_argument(
        "--skip-load-validation",
        action="store_true",
        help=(
            "Skip the post-tuning check that the generated config is loadable "
            "through SGLANG_MOE_CONFIG_DIR."
        ),
    )
    return parser


def parse_batch_sizes(values: Optional[Iterable[str]]) -> Optional[List[int]]:
    if values is None:
        return None

    batch_sizes: List[int] = []
    seen = set()
    for value in values:
        for raw_part in str(value).split(","):
            part = raw_part.strip()
            if not part:
                raise ValueError("empty value in --batch-size")
            try:
                batch_size = int(part)
            except ValueError as exc:
                raise ValueError(f"invalid batch size: {part}") from exc
            if batch_size <= 0:
                raise ValueError("--batch-size values must be positive")
            if batch_size not in seen:
                batch_sizes.append(batch_size)
                seen.add(batch_size)

    return batch_sizes


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.tp_size <= 0:
        parser.error("--tp-size must be positive")
    if args.ep_size <= 0:
        parser.error("--ep-size must be positive")
    if args.tp_size % args.ep_size != 0:
        parser.error("--tp-size must be divisible by --ep-size")

    try:
        args.batch_sizes = parse_batch_sizes(args.batch_size)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def _ensure_source_root_on_path() -> None:
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "benchmark" / "kernels" / "fused_moe_triton").exists():
        source_root_str = str(source_root)
        if source_root_str not in sys.path:
            sys.path.insert(0, source_root_str)


def _load_common_utils():
    _ensure_source_root_on_path()
    try:
        return importlib.import_module(
            "benchmark.kernels.fused_moe_triton.common_utils"
        )
    except ModuleNotFoundError as exc:
        if exc.name == "benchmark":
            raise RuntimeError(
                "SGLang auto_tune currently reuses the fused MoE tuner under "
                "benchmark/kernels/fused_moe_triton. Run it from a source "
                "checkout or make those benchmark sources importable."
            ) from exc
        raise


def _load_tuner():
    _ensure_source_root_on_path()
    try:
        return importlib.import_module(
            "benchmark.kernels.fused_moe_triton.tuning_fused_moe_triton"
        )
    except ModuleNotFoundError as exc:
        if exc.name == "benchmark":
            raise RuntimeError(
                "SGLang auto_tune currently reuses the fused MoE tuner under "
                "benchmark/kernels/fused_moe_triton. Run it from a source "
                "checkout or make those benchmark sources importable."
            ) from exc
        if exc.name == "ray":
            raise RuntimeError(
                "SGLang auto_tune reuses the fused MoE tuner, which requires "
                "Ray. Install the optional Ray dependency, e.g. "
                "`pip install 'ray[default]>=2.54.0'`, and retry."
            ) from exc
        raise


def _load_runtime_config_loader():
    return importlib.import_module(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config"
    )


def _install_runtime_server_args(plan: AutoTunePlan):
    server_args_module = importlib.import_module("sglang.srt.server_args")
    previous_server_args = getattr(server_args_module, "_global_server_args", None)

    try:
        server_args_module.get_global_server_args()
    except ValueError as exc:
        if "Global server args is not set" not in str(exc):
            raise
        server_args_module.set_global_server_args_for_scheduler(
            server_args_module.ServerArgs(model_path=plan.model_path)
        )

    def restore() -> None:
        server_args_module._global_server_args = previous_server_args

    return restore


def _dtype_flags(dtype: str) -> dict[str, bool]:
    return {
        "use_fp8_w8a8": dtype == "fp8_w8a8",
        "use_int8_w8a8": dtype == "int8_w8a8",
        "use_int8_w8a16": dtype == "int8_w8a16",
        "use_int4_w4a16": dtype == "int4_w4a16",
    }


def _dtype_selector(dtype: Any, dtype_flags: dict[str, bool]) -> Optional[str]:
    if dtype_flags["use_fp8_w8a8"]:
        return "fp8_w8a8"
    if dtype_flags["use_int8_w8a8"]:
        return "int8_w8a8"
    if dtype_flags["use_int4_w4a16"]:
        return "int4_w4a16"
    if dtype_flags["use_int8_w8a16"]:
        return "int8_w8a16"
    if str(dtype) in {"torch.float32", "torch.float"}:
        return "float32"
    return None


def _build_config_filename(
    num_experts: int,
    shard_intermediate_size: int,
    dtype: Any,
    dtype_flags: dict[str, bool],
    block_shape: Optional[List[int]],
    per_channel_quant: bool,
    device_name: str,
) -> str:
    dtype_str = _dtype_selector(dtype, dtype_flags)
    dtype_part = "" if dtype_str is None else f",dtype={dtype_str}"
    block_shape_part = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    )
    per_channel_quant_part = ",per_channel_quant=True" if per_channel_quant else ""
    sanitized_device_name = device_name.replace(" ", "_")

    N = shard_intermediate_size // 2
    if dtype_flags["use_int4_w4a16"]:
        N = N // 2

    return (
        f"E={num_experts},N={N},device_name={sanitized_device_name}"
        f"{dtype_part}{block_shape_part}{per_channel_quant_part}.json"
    )


def _collect_runtime_info() -> RuntimeInfo:
    torch_version = "unavailable"
    cuda_version = "unavailable"
    triton_version = "unavailable"
    device_name = "unknown_device"

    try:
        torch = importlib.import_module("torch")
        torch_version = getattr(torch, "__version__", "unavailable")
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None) or "none"
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device_name = torch.xpu.get_device_name(0)
        elif hasattr(torch, "hpu") and torch.hpu.is_available():
            device_name = torch.hpu.get_device_name(0)
        elif hasattr(torch, "npu") and torch.npu.is_available():
            device_name = torch.npu.get_device_name(0)
    except Exception:
        pass

    try:
        triton = importlib.import_module("triton")
        triton_version = getattr(triton, "__version__", "unavailable")
    except Exception:
        pass

    return RuntimeInfo(
        device_name=device_name,
        torch_version=torch_version,
        cuda_version=cuda_version,
        triton_version=triton_version,
    )


def _triton_version_dir(triton_version: str) -> str:
    if not triton_version or triton_version == "unavailable":
        return "triton_unknown"
    return f"triton_{triton_version.replace('.', '_')}"


def resolve_output_path(
    output_dir: str | Path, filename: str, triton_version: str
) -> Path:
    return (
        Path(output_dir).expanduser().resolve()
        / "configs"
        / _triton_version_dir(triton_version)
        / filename
    )


def select_quick_search_space(
    search_space: List[dict[str, int]], max_configs: int = 128
) -> List[dict[str, int]]:
    if max_configs <= 0:
        raise ValueError("max_configs must be positive")

    preferred = [
        config
        for config in search_space
        if config.get("BLOCK_SIZE_M") in {16, 32, 64, 128}
        and config.get("BLOCK_SIZE_N") in {64, 128, 256}
        and config.get("BLOCK_SIZE_K") in {64, 128}
        and config.get("GROUP_SIZE_M") in {1, 16, 32}
        and config.get("num_warps") in {4, 8}
        and config.get("num_stages") in {2, 3, 4}
    ]
    candidates = preferred or list(search_space)

    if len(candidates) <= max_configs:
        return candidates
    if max_configs == 1:
        return [candidates[0]]

    step = (len(candidates) - 1) / (max_configs - 1)
    indices = sorted({round(i * step) for i in range(max_configs)})
    return [candidates[index] for index in indices]


def _filter_search_space_for_model(
    common_utils,
    search_space: List[dict[str, int]],
    block_shape: Optional[List[int]],
) -> List[dict[str, int]]:
    if hasattr(common_utils, "filter_search_space_for_model"):
        return common_utils.filter_search_space_for_model(search_space, block_shape)
    if block_shape is None:
        return search_space
    block_k = block_shape[1]
    return [config for config in search_space if block_k % config["BLOCK_SIZE_K"] == 0]


def resolve_tuning_plan(args: argparse.Namespace) -> AutoTunePlan:
    common_utils = _load_common_utils()
    runtime = _collect_runtime_info()
    batch_sizes = args.batch_sizes or DEFAULT_BATCH_SIZES

    model_config = common_utils.get_model_config(
        args.model_path,
        args.tp_size,
        args.ep_size,
        args.disable_shared_experts_fusion,
    )
    dtype_flags = _dtype_flags(args.dtype)

    try:
        filename = common_utils.get_config_filename(
            model_config["num_experts"],
            model_config["shard_intermediate_size"],
            model_config["hidden_size"],
            model_config["topk"],
            model_config["dtype"],
            dtype_flags["use_fp8_w8a8"],
            dtype_flags["use_int8_w8a8"],
            dtype_flags["use_int8_w8a16"],
            dtype_flags["use_int4_w4a16"],
            args.per_channel_quant,
            model_config["block_shape"],
        )
    except AttributeError as exc:
        if "replace" not in str(exc):
            raise
        filename = _build_config_filename(
            model_config["num_experts"],
            model_config["shard_intermediate_size"],
            model_config["dtype"],
            dtype_flags,
            model_config["block_shape"],
            args.per_channel_quant,
            runtime.device_name,
        )

    output_root = Path(args.output_dir).expanduser().resolve()
    output_file = resolve_output_path(output_root, filename, runtime.triton_version)

    search_space_size = None
    try:
        search_space = common_utils.get_configs_compute_bound()
        if args.quick:
            search_space = select_quick_search_space(search_space)
        search_space = _filter_search_space_for_model(
            common_utils, search_space, model_config["block_shape"]
        )
        search_space_size = len(search_space)
    except Exception:
        if not args.dry_run:
            raise

    return AutoTunePlan(
        model_path=args.model_path,
        architecture=model_config["architecture"],
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        dtype=args.dtype,
        model_dtype=str(model_config["dtype"]),
        batch_sizes=batch_sizes,
        num_experts=model_config["num_experts"],
        topk=model_config["topk"],
        hidden_size=model_config["hidden_size"],
        shard_intermediate_size=model_config["shard_intermediate_size"],
        block_shape=model_config["block_shape"],
        per_channel_quant=args.per_channel_quant,
        filename=filename,
        output_root=output_root,
        output_file=output_file,
        quick=args.quick,
        search_space_size=search_space_size,
        runtime=runtime,
    )


def _format_list(values: Iterable[Any]) -> str:
    return ", ".join(str(value) for value in values)


def _runtime_config_n(plan: AutoTunePlan) -> int:
    N = plan.shard_intermediate_size // 2
    if _dtype_flags(plan.dtype)["use_int4_w4a16"]:
        N = N // 2
    return N


def print_summary(
    plan: AutoTunePlan,
    dry_run: bool = False,
    load_validated: bool = False,
) -> None:
    status = "Dry run" if dry_run else "Completed"
    print(f"SGLang auto_tune: {status}")
    print(f"  model: {plan.model_path}")
    print(f"  architecture: {plan.architecture}")
    print(
        "  fused_moe_triton: "
        f"E={plan.num_experts}, topk={plan.topk}, "
        f"hidden={plan.hidden_size}, shard_intermediate={plan.shard_intermediate_size}"
    )
    print(
        f"  parallelism: tp={plan.tp_size}, ep={plan.ep_size}; "
        f"dtype={plan.dtype} (model dtype: {plan.model_dtype})"
    )
    print(f"  batch sizes: {_format_list(plan.batch_sizes)}")
    if plan.search_space_size is not None:
        mode = "quick" if plan.quick else "full"
        print(f"  search: {mode}, {plan.search_space_size} candidates")
    print(
        f"  device: {plan.runtime.device_name}; CUDA: {plan.runtime.cuda_version}; "
        f"Triton: {plan.runtime.triton_version}; torch: {plan.runtime.torch_version}"
    )
    print("  output config paths:")
    print(f"    {plan.output_file}")
    if dry_run:
        print("  runtime loader validation: skipped (dry run)")
    elif load_validated:
        print("  runtime loader validation: passed")
    print("  reuse with:")
    print(
        "    "
        f"SGLANG_MOE_CONFIG_DIR={shlex.quote(str(plan.output_root))} "
        f"python -m sglang.launch_server --model-path "
        f"{shlex.quote(plan.model_path)} --tp {plan.tp_size} "
        f"--ep-size {plan.ep_size}"
    )


def validate_output_config(plan: AutoTunePlan) -> List[int]:
    if not plan.output_file.exists():
        raise RuntimeError(
            f"tuning completed but output config was not found at {plan.output_file}"
        )

    with plan.output_file.open() as file:
        raw_config = json.load(file)

    expected_keys = sorted(int(key) for key in raw_config)
    runtime_loader = _load_runtime_config_loader()
    get_moe_configs = runtime_loader.get_moe_configs
    restore_server_args = _install_runtime_server_args(plan)
    if hasattr(get_moe_configs, "cache_clear"):
        get_moe_configs.cache_clear()

    dtype = _dtype_selector(plan.model_dtype, _dtype_flags(plan.dtype))
    block_n = plan.block_shape[0] if plan.block_shape else 0
    block_k = plan.block_shape[1] if plan.block_shape else 0
    previous_config_dir = os.environ.get("SGLANG_MOE_CONFIG_DIR")
    os.environ["SGLANG_MOE_CONFIG_DIR"] = str(plan.output_root)
    try:
        loaded_config = get_moe_configs(
            plan.num_experts,
            _runtime_config_n(plan),
            dtype,
            block_n=block_n,
            block_k=block_k,
            per_channel_quant=plan.per_channel_quant,
        )
    finally:
        if previous_config_dir is None:
            os.environ.pop("SGLANG_MOE_CONFIG_DIR", None)
        else:
            os.environ["SGLANG_MOE_CONFIG_DIR"] = previous_config_dir
        if hasattr(get_moe_configs, "cache_clear"):
            get_moe_configs.cache_clear()
        restore_server_args()

    if loaded_config is None:
        raise RuntimeError(
            "generated config was not loadable through SGLANG_MOE_CONFIG_DIR="
            f"{plan.output_root}"
        )

    loaded_keys = sorted(int(key) for key in loaded_config)
    if loaded_keys != expected_keys:
        raise RuntimeError(
            "generated config loaded with unexpected batch-size keys: "
            f"expected {expected_keys}, got {loaded_keys}"
        )

    return loaded_keys


def run_tuning(args: argparse.Namespace, plan: AutoTunePlan) -> dict[str, Any]:
    tuner = _load_tuner()
    search_space = None
    if args.quick:
        search_space = select_quick_search_space(tuner.get_configs_compute_bound())

    result = tuner.tune_fused_moe_triton(
        model=args.model_path,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        dtype=args.dtype,
        batch_sizes=plan.batch_sizes,
        seed=args.seed,
        per_channel_quant=args.per_channel_quant,
        disable_shared_experts_fusion=args.disable_shared_experts_fusion,
        tune=True,
        search_space=search_space,
        output_file=str(plan.output_file),
    )
    if not args.skip_load_validation:
        result["load_validation_keys"] = validate_output_config(plan)
        result["load_validated"] = True
    else:
        result["load_validated"] = False
    return result


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)
        plan = resolve_tuning_plan(args)
        if args.dry_run:
            print_summary(plan, dry_run=True)
            return 0

        result = run_tuning(args, plan)
        print_summary(
            plan,
            dry_run=False,
            load_validated=bool(result.get("load_validated")),
        )
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
