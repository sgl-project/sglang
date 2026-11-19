import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from importlib import metadata
from pathlib import Path

BUILD_TARGETS = {"sglang", "sgl-kernel", "sglang-kernel"}
RUNTIME_UPGRADES = {"transformers", "peft"}
PINNED_ADDITIONS = {
    "sailify": "1.0.0",
    "sgl-eval": "0.1.2",
    "math-verify": "0.9.0",
    "latex2sympy2-extended": "1.11.0",
    "antlr4-python3-runtime": "4.9.3",
    "editdistance": "0.8.1",
}
SAIL_ALIASES = {
    "deep-gemm-for-sail": "deep-gemm",
    "flash-attention-for-sail": "flash-attn",
    "flashinfer-for-sail": "flashinfer-python",
    "flashmla-for-sail": "flash-mla",
    "torch-for-sail": "torch",
    "triton-for-sail": "triton",
}
PLATFORM_PACKAGES = {
    "torch",
    "torchvision",
    "torchaudio",
    "torchao",
    "torch-memory-saver",
    "triton",
    "deep-gemm",
    "deep-ep",
    "flash-attn",
    "flash-attn-3",
    "flash-attn-4",
    "flash-mla",
    "flashinfer-python",
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


def is_platform_package(name):
    return (
        name in PLATFORM_PACKAGES
        or name in SAIL_ALIASES
        or name.startswith(("nvidia-", "cuda-"))
        or name.endswith("-for-sail")
    )


def read_wheel(path):
    from packaging.utils import parse_wheel_filename
    from packaging.version import Version

    name, version, _, _ = parse_wheel_filename(path.name)
    with zipfile.ZipFile(path) as archive:
        entries = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(entries) != 1:
            raise ValueError(f"Expected one METADATA entry in {path}")
        message = Parser().parsestr(archive.read(entries[0]).decode("utf-8"))
    if (
        normalize_name(message["Name"]) != name
        or Version(message["Version"]) != version
    ):
        raise ValueError(f"Wheel filename and metadata disagree: {path}")
    return message


def wheel_inventory(wheel_dir):
    wheels = {}
    for path in sorted(wheel_dir.glob("*.whl")):
        message = read_wheel(path)
        name = normalize_name(message["Name"])
        if name in wheels:
            raise ValueError(f"Multiple wheels for {name} in {wheel_dir}")
        wheels[name] = (path.resolve(), message)
    return wheels


def walk_dev_sail(root, load):
    from packaging.markers import default_environment
    from packaging.requirements import Requirement

    if "dev-sail" not in {
        normalize_name(extra) for extra in root.get_all("Provides-Extra", [])
    }:
        raise ValueError("The built SGLang wheel does not provide dev_sail")
    pending = [(Requirement("sglang[dev_sail]"), False)]
    pending.extend(
        (Requirement(f"{n}=={v}"), False) for n, v in PINNED_ADDITIONS.items()
    )
    visited = set()
    environment = default_environment()
    while pending:
        requirement, strict = pending.pop()
        name = normalize_name(requirement.name)
        strict = strict or name in RUNTIME_UPGRADES
        if requirement.url:
            raise ValueError(f"Direct URL dependency is not supported: {requirement}")
        message = root if name == "sglang" else load(name, requirement, strict)
        if message is None:
            continue
        extras = {"", *(normalize_name(extra) for extra in requirement.extras)}
        for extra in sorted(extras):
            if (name, extra, strict) in visited:
                continue
            visited.add((name, extra, strict))
            for value in message.get_all("Requires-Dist", []):
                child = Requirement(value)
                if child.marker is None or child.marker.evaluate(
                    {**environment, "extra": extra}
                ):
                    child.marker = None
                    pending.append((child, strict))


def dev_sail_wheels(wheel_dir):
    built = wheel_inventory(wheel_dir)
    if "sglang" not in built:
        raise ValueError("Missing built SGLang wheel")
    return built["sglang"][1], wheel_inventory(wheel_dir / "runtime-deps")


def collect_dev_sail(wheel_dir):
    from packaging.requirements import Requirement

    root, wheels = dev_sail_wheels(wheel_dir)
    seen = set()
    while True:
        state = tuple(sorted((name, m["Version"]) for name, (_, m) in wheels.items()))
        if state in seen:
            raise ValueError("Offline dependency selection did not converge")
        seen.add(state)
        constraints = {}
        needed = set()

        def load(name, requirement, strict):
            if is_platform_package(name):
                return None
            if name in BUILD_TARGETS:
                raise ValueError(
                    f"Unexpected dependency on built package: {requirement}"
                )
            constraints.setdefault(name, set()).add(str(requirement))
            if name not in wheels:
                needed.add(name)
                return None
            message = wheels[name][1]
            if not requirement.specifier.contains(message["Version"], prereleases=True):
                needed.add(name)
                return None
            return message

        walk_dev_sail(root, load)
        if not needed:
            return
        requirements = sorted({r for name in needed for r in constraints[name]})
        with tempfile.TemporaryDirectory(dir=wheel_dir / "runtime-deps") as directory:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--index-url",
                    "https://pypi.org/simple",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    directory,
                    *requirements,
                ],
                check=True,
            )
            candidates = wheel_inventory(Path(directory))
            if candidates.keys() != needed:
                raise ValueError(
                    f"Expected exactly one replacement wheel for {sorted(needed)}"
                )
            for name in needed:
                path, message = candidates[name]
                for value in constraints[name]:
                    if not Requirement(value).specifier.contains(
                        message["Version"], prereleases=True
                    ):
                        raise ValueError(
                            f"Offline wheel {path.name} does not satisfy {value}"
                        )
            for name in sorted(needed):
                path, _ = candidates[name]
                if name in wheels:
                    wheels[name][0].unlink()
                path.replace(wheel_dir / "runtime-deps" / path.name)
        _, wheels = dev_sail_wheels(wheel_dir)


def missing_dev_sail(wheel_dir, packages):
    from packaging.version import Version

    root, wheels = dev_sail_wheels(wheel_dir)
    distributions = {}
    for dist in metadata.distributions():
        distributions.setdefault(normalize_name(dist.metadata["Name"]), dist.metadata)
    selected = {}

    def load(name, requirement, strict):
        if is_platform_package(name):
            actual = SAIL_ALIASES.get(name, name)
            if actual not in packages:
                raise ValueError(
                    f"Runtime image must preinstall {actual} for {requirement}"
                )
            if strict and not requirement.specifier.contains(
                packages[actual]["version"], prereleases=True
            ):
                raise ValueError(
                    f"Preinstalled {actual}=={packages[actual]['version']} "
                    f"does not satisfy {requirement}; update the runtime image."
                )
            return None
        if name in BUILD_TARGETS:
            raise ValueError(f"Unexpected dependency on built package: {requirement}")
        if name in packages and name not in RUNTIME_UPGRADES:
            if name not in distributions:
                raise ValueError(f"Missing installed metadata for {name}")
            if strict and not requirement.specifier.contains(
                packages[name]["version"], prereleases=True
            ):
                raise ValueError(
                    f"Preinstalled {name}=={packages[name]['version']} "
                    f"does not satisfy {requirement}; update the runtime image."
                )
            return distributions[name]
        if name not in wheels:
            raise ValueError(f"Offline wheelhouse is missing {requirement}")
        path, message = wheels[name]
        if not requirement.specifier.contains(message["Version"], prereleases=True):
            raise ValueError(
                f"Offline wheel {path.name} does not satisfy {requirement}"
            )
        if name in packages and Version(packages[name]["version"]) == Version(
            message["Version"]
        ):
            if name not in distributions:
                raise ValueError(f"Missing installed metadata for {name}")
            return distributions[name]
        selected[name] = path, message["Version"]
        return message

    walk_dev_sail(root, load)
    return selected


def check_snapshots(before, after, additions):
    if any(is_platform_package(name) or name in BUILD_TARGETS for name in additions):
        raise ValueError(
            "Dependency additions must not include platform or built packages"
        )
    changed = {
        name: {"before": before.get(name), "after": after.get(name)}
        for name in before.keys() | after.keys()
        if name not in BUILD_TARGETS
        and not (
            name in additions
            and name in after
            and after[name]["version"] == additions[name]
            and (
                name not in before
                or (
                    name in RUNTIME_UPGRADES
                    and before[name]["version"] != additions[name]
                )
            )
        )
        and (
            before.get(name) != after.get(name)
            or (
                name in RUNTIME_UPGRADES
                and name in additions
                and after.get(name, {}).get("version") != additions[name]
            )
        )
    }
    if changed:
        print("Unexpected dependency changes:", file=sys.stderr)
        print(json.dumps(changed, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print("Only built packages, planned runtime upgrades and additions changed.")
    return 0


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("snapshot")
    capture.add_argument("output", type=Path)
    check = commands.add_parser("check")
    check.add_argument("before", type=Path)
    check.add_argument("after", type=Path)
    check.add_argument("--additions", type=Path)
    collect = commands.add_parser("collect-dev-sail")
    collect.add_argument("wheel_dir", type=Path)
    dev_sail = commands.add_parser("missing-dev-sail")
    dev_sail.add_argument("wheel_dir", type=Path)
    dev_sail.add_argument("additions", type=Path)
    commands.add_parser("requirements")
    wheels = commands.add_parser("missing-wheels")
    wheels.add_argument("wheel_dir", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        args.output.write_text(json.dumps(snapshot(), indent=2, sort_keys=True) + "\n")
        return 0
    if args.command in ("collect-dev-sail", "missing-dev-sail"):
        try:
            if args.command == "collect-dev-sail":
                collect_dev_sail(args.wheel_dir)
            else:
                packages = snapshot()
                selected = missing_dev_sail(args.wheel_dir, packages)
                args.additions.write_text(
                    json.dumps({n: v for n, (_, v) in selected.items()}, indent=2)
                    + "\n"
                )
                for path, _ in selected.values():
                    print(path)
        except (ValueError, KeyError, zipfile.BadZipFile) as error:
            print(error, file=sys.stderr)
            return 1
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
    additions = (
        json.loads(args.additions.read_text()) if args.additions else PINNED_ADDITIONS
    )
    try:
        return check_snapshots(before, after, additions)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
