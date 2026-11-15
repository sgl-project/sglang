import argparse
import hashlib
import json
import re
import sys
from importlib import metadata
from pathlib import Path

BUILD_TARGETS = {"sglang", "sgl-kernel", "sglang-kernel"}
PINNED_ADDITIONS = {
    "sailify": "1.0.0",
    "sgl-eval": "0.1.2",
    "math-verify": "0.9.0",
    "latex2sympy2-extended": "1.11.0",
    "antlr4-python3-runtime": "4.9.3",
    "editdistance": "0.8.1",
}


def normalize_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def snapshot():
    packages = {}
    for dist in metadata.distributions():
        name = normalize_name(dist.metadata["Name"])
        if name not in packages:
            packages[name] = {
                "version": dist.version,
                "record": hashlib.sha256(
                    (dist.read_text("RECORD") or "").encode()
                ).hexdigest(),
            }
    return packages


def validate_evaluators(packages):
    for name, version in PINNED_ADDITIONS.items():
        if name == "sailify" or name not in packages:
            continue
        installed = packages[name]["version"]
        if installed != version:
            raise ValueError(
                f"Preinstalled {name}=={installed} does not match required "
                f"{name}=={version}; refusing to replace it."
            )


def missing_wheels(wheel_dir, packages):
    if not wheel_dir.is_dir():
        raise ValueError(f"Wheel directory does not exist: {wheel_dir}")
    available = {}
    for path in wheel_dir.resolve().glob("*.whl"):
        match = re.fullmatch(
            r"([^-]+)-([^-]+)-(?:[0-9][^-]*-)?[^-]+-[^-]+-[^-]+\.whl", path.name
        )
        if path.is_file() and match:
            key = (normalize_name(match[1]), match[2])
            available.setdefault(key, []).append(path)
    selected = []
    for name, version in PINNED_ADDITIONS.items():
        if name in packages:
            continue
        matches = available.get((name, version), [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one wheel for {name}=={version} in {wheel_dir}; "
                f"found {len(matches)}."
            )
        selected.append(str(matches[0]))
    return selected


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("snapshot")
    capture.add_argument("output", type=Path)
    check = commands.add_parser("check")
    check.add_argument("before", type=Path)
    check.add_argument("after", type=Path)
    commands.add_parser("requirements")
    wheels = commands.add_parser("missing-wheels")
    wheels.add_argument("wheel_dir", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        args.output.write_text(json.dumps(snapshot(), indent=2, sort_keys=True) + "\n")
        return 0
    if args.command == "requirements":
        for name, version in PINNED_ADDITIONS.items():
            if name != "sailify":
                print(f"{name}=={version}")
        return 0
    if args.command == "missing-wheels":
        try:
            packages = snapshot()
            validate_evaluators(packages)
            lines = missing_wheels(args.wheel_dir, packages)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        for line in lines:
            print(line)
        return 0

    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    changed = {
        name: {"before": before.get(name), "after": after.get(name)}
        for name in before.keys() | after.keys()
        if name not in BUILD_TARGETS
        and not (
            name not in before
            and name in PINNED_ADDITIONS
            and after[name]["version"] == PINNED_ADDITIONS[name]
        )
        and before.get(name) != after.get(name)
    }
    if changed:
        print("Unexpected dependency changes:", file=sys.stderr)
        print(json.dumps(changed, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print("All preinstalled dependencies other than build targets are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
