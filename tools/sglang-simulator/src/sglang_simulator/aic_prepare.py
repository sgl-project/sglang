"""Prepare a typed, immutable AFD cost snapshot from explicit AIC requests."""

import argparse
import json
from pathlib import Path

from .afd.adapters import create_adapter
from .aic_costs import prepare_cost_document, write_snapshot
from .aic_providers import AICProviders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aic-config", type=Path, required=True)
    parser.add_argument("--aic-repository-root", type=Path)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--api", choices=("core", "legacy"), default="core")
    parser.add_argument("--allow-conditional", action="store_true")
    parser.add_argument("--allow-unknown", action="store_true")
    parser.add_argument("--allow-unknown-source", action="store_true")
    parser.add_argument("--allow-archive-control", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be new")
    config = json.loads(args.deployment.read_text())
    requests = json.loads(args.requests.read_text())
    if any("billing" not in request for request in requests.values()):
        parser.error(
            "Each prepared request must declare billing.group and billing.kind"
        )
    adapter = create_adapter(
        args.source_root,
        config["mode"],
        factory=args.adapter,
        routing=config.get("routing", "f-side"),
    )
    adapter.verify()
    policy = ["matched"]
    if args.allow_conditional:
        policy.append("conditional")
    if args.allow_unknown:
        policy.append("unknown")
    resolver = AICProviders(
        args.aic_config,
        api=args.api,
        allowed_qualifications=policy,
        repository_root=args.aic_repository_root,
        allow_unknown_source=args.allow_unknown_source,
        allow_archives=args.allow_archive_control,
    )
    # Discover the exact published evidence identities; no old numeric oracle.
    # prepare_cost_document re-queries against the same bound inputs.
    for request in requests.values():
        resolver(request)
    inputs = list(resolver.inputs) + [
        args.deployment,
        args.requests,
        Path(__file__).with_name("aic_providers.py"),
    ]
    document = prepare_cost_document(
        config,
        adapter.cost_mapping(config),
        resolver,
        requests=requests,
        input_paths=inputs,
        allowed_qualifications=policy,
        allow_unknown_source=args.allow_unknown_source,
    )
    write_snapshot(document, args.output)


if __name__ == "__main__":
    main()
