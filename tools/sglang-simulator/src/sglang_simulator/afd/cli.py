"""Run one source-bound homogeneous decode step on CPU, without a model load."""

import argparse
import json
from pathlib import Path

from .adapters import create_adapter
from .costs import template
from .native_config import select_native_config
from .runner import critical_path as critical_path
from .runner import simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--adapter",
        help="Explicit trusted Python module:factory; default: pinned SGLang adapter",
    )
    parser.add_argument(
        "--deployment",
        type=Path,
        help="Deployment JSON for template generation, required for custom adapters",
    )
    parser.add_argument("--costs", type=Path)
    parser.add_argument("--write-template", action="store_true")
    parser.add_argument("--mode", default="afd")
    parser.add_argument(
        "--routing",
        choices=("f-side", "direct"),
        default=None,
        help="F-side gather/reduce (default), or A-side routing with full-F Direct broadcast",
    )
    parser.add_argument("--ffn-ranks", type=int)
    parser.add_argument(
        "--afd-config", help="Native AFD JSON; select ffn_backend fixed/direct"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--effective-kv", type=int, default=8704)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native, routing, ranks = select_native_config(
        args.afd_config, routing=args.routing, ffn_ranks=args.ffn_ranks
    )
    if args.adapter and native:
        parser.error(
            "--afd-config selects a built-in adapter; use --deployment for custom adapters"
        )
    if args.write_template == (args.costs is not None):
        parser.error("choose exactly one of --costs or --write-template")
    if args.deployment and not args.write_template:
        parser.error("--deployment is only used with --write-template")
    if args.write_template:
        if args.adapter and not args.deployment:
            parser.error("custom adapters require --deployment for templates")
        config = json.loads(args.deployment.read_text()) if args.deployment else None
        adapter = create_adapter(
            args.source_root,
            config["mode"] if config else args.mode,
            factory=args.adapter,
            routing=routing,
        )
        if config is None:
            config = adapter.template_config(
                batch_size=args.batch_size,
                effective_kv=args.effective_kv,
                ffn_ranks=ranks,
            )
        if native:
            native.check(config)
        mapping = adapter.cost_mapping(config)
        adapter.verify()
        result = template(config, mapping)
    else:
        document = json.loads(args.costs.read_text())
        adapter = create_adapter(
            args.source_root,
            document["deployment"]["mode"],
            factory=args.adapter,
            routing=routing,
        )
        if native:
            native.check(document["deployment"])
        result = simulate(document, adapter=adapter)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
