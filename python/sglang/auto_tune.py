from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

from sglang.tune.fused_moe_triton import FusedMoeTritonTuner
from sglang.tune.types import AutoTuneConfig, ComponentResult, ComponentTuner

LOGGER = logging.getLogger(__name__)


class AutoTuneRunner:
    def __init__(self, tuners: Sequence[ComponentTuner]) -> None:
        self._registry: Dict[str, ComponentTuner] = {
            tuner.name: tuner for tuner in tuners
        }

    def list_components(self) -> Tuple[str, ...]:
        return tuple(sorted(self._registry.keys()))

    def run(
        self, config: AutoTuneConfig, components: Sequence[str]
    ) -> List[ComponentResult]:
        results: List[ComponentResult] = []
        for component in components:
            tuner = self._registry.get(component)
            if tuner is None:
                LOGGER.error("No tuner registered for component '%s'.", component)
                results.append(
                    ComponentResult(
                        name=component,
                        status="unavailable",
                        details=f"No tuner registered for '{component}'.",
                        output_files=tuple(),
                        metadata={},
                    )
                )
                continue
            if not tuner.is_available():
                LOGGER.warning(
                    "Skipping component '%s' because its dependencies are unavailable.",
                    component,
                )
                results.append(
                    ComponentResult(
                        name=component,
                        status="skipped",
                        details="Required dependencies are not available.",
                        output_files=tuple(),
                        metadata={},
                    )
                )
                continue
            LOGGER.info("Starting auto-tuning for component '%s'.", component)
            try:
                result = tuner.run(config)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.exception("Auto-tuning failed for component '%s'.", component)
                result = ComponentResult(
                    name=component,
                    status="failed",
                    details=str(exc),
                    output_files=tuple(),
                    metadata={"exception_type": type(exc).__name__},
                )
            results.append(result)
        return results


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def parse_components(
    raw_components: Optional[Sequence[str]], default_components: Sequence[str]
) -> Tuple[str, ...]:
    if not raw_components:
        return tuple(default_components)
    parsed: List[str] = []
    for item in raw_components:
        parts = [part.strip() for part in item.split(",") if part.strip()]
        parsed.extend(parts)
    if "all" in {component.lower() for component in parsed}:
        return tuple(default_components)
    return tuple(dict.fromkeys(parsed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-tune sglang kernels and dispatch parameters for a specific model."
    )
    parser.add_argument(
        "--model-path", required=True, type=str, help="Model identifier or local path."
    )
    parser.add_argument(
        "--tp",
        "--tensor-parallel",
        dest="tensor_parallel",
        type=int,
        default=1,
        help="Tensor parallel degree.",
    )
    parser.add_argument(
        "--ep",
        "--expert-parallel",
        dest="expert_parallel",
        type=int,
        default=1,
        help="Expert parallel degree.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8"],
        default="auto",
        help="Quantization mode passed to underlying tuners.",
    )
    parser.add_argument(
        "--per-channel-quant",
        action="store_true",
        help="Enable per-channel quantization for supported kernels.",
    )
    parser.add_argument(
        "--disable-shared-experts-fusion",
        action="store_true",
        help="Disable shared experts fusion in fused MoE tuner.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override for kernel tuning.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "auto_tune_results",
        help="Directory used to store tuning outputs.",
    )
    parser.add_argument(
        "--components",
        action="append",
        help="Comma separated list of components to tune. Use --components all for every available component.",
    )
    parser.add_argument(
        "--list-components",
        action="store_true",
        help="List available tuning components and exit.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    runner = AutoTuneRunner(
        tuners=(
            FusedMoeTritonTuner(),  # TODO: add more tuners here, only fused_moe_triton is supported now
        )
    )

    if args.list_components:
        for component in runner.list_components():
            print(component)
        return 0

    components = parse_components(args.components, runner.list_components())
    results_dir = args.output_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    config = AutoTuneConfig(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel,
        expert_parallel_size=args.expert_parallel,
        dtype=args.dtype,
        per_channel_quant=args.per_channel_quant,
        disable_shared_experts_fusion=args.disable_shared_experts_fusion,
        batch_size=args.batch_size,
        seed=args.seed,
        results_dir=results_dir,
    )

    results = runner.run(config, components)

    summary: MutableMapping[str, Any] = {
        "_meta": {
            "model_path": config.model_path,
            "tensor_parallel_size": config.tensor_parallel_size,
            "expert_parallel_size": config.expert_parallel_size,
            "dtype": config.dtype,
            "components": list(components),
        }
    }

    for result in results:
        summary[result.name] = result.to_summary()

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOGGER.info("Wrote summary to %s.", summary_path)

    any_failure = any(result.status not in {"success", "skipped"} for result in results)
    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
