import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sysconfig
import tempfile
import zipfile
from pathlib import Path


def visible_mount(*, host_root, container):
    directory = Path(host_root) / "aiter-wheels/.locks"
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix="probe-", dir=directory)
    token = os.urandom(16).hex()
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(token)
        result = subprocess.run(
            [
                "docker",
                "exec",
                container,
                "cat",
                f"/sgl-data/aiter-wheels/.locks/{Path(filename).name}",
            ],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and result.stdout.strip() == token
    finally:
        Path(filename).unlink(missing_ok=True)


def build_environment():
    versions = {}
    for name in (
        "torch",
        "triton",
        "pytorch-triton-rocm",
        "flydsl",
        "numpy",
        "pybind11",
        "setuptools",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return json.dumps(
        {"soabi": sysconfig.get_config_var("SOABI"), "versions": versions},
        sort_keys=True,
    )


def cache_key(*, commit, image, arch, namespace, inputs, abi):
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("AITER cache requires an immutable 40-character commit")
    if not image or not arch or not abi:
        raise ValueError(
            "AITER cache requires an image digest, GPU architecture and ABI"
        )
    payload = {
        "commit": commit,
        "image": image,
        "arch": arch,
        "namespace": namespace,
        "abi": abi,
        "inputs": [
            hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if archive.testzip() is not None:
            raise ValueError("Corrupt AITER wheel")
        for prefix in ("aiter/", "aiter_meta/csrc/", "aiter_meta/hsa/"):
            if not any(name.startswith(prefix) for name in names):
                raise ValueError(f"AITER wheel is missing {prefix}")


def cached_wheel(*, directory, key):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("key") != key or manifest.get("version") != 1:
        raise ValueError("AITER cache key mismatch")
    wheels = list(root.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("AITER cache must contain exactly one wheel")
    wheel = wheels[0]
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != manifest.get("sha256"):
        raise ValueError("AITER cache checksum mismatch")
    validate_wheel(wheel)
    return wheel


def publish(*, wheel, root, key):
    source = Path(wheel)
    validate_wheel(source)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".tmp-", dir=root))
    try:
        copied = temporary / source.name
        shutil.copyfile(source, copied)
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "key": key,
                    "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
                }
            )
        )
        destination = root / key
        if destination.exists():
            cached_wheel(directory=destination, key=key)
        else:
            os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("environment")
    mount_parser = commands.add_parser("check-mount")
    mount_parser.add_argument("--host-root", required=True)
    mount_parser.add_argument("--container", required=True)
    key_parser = commands.add_parser("key")
    for argument in ("commit", "image", "arch", "namespace", "abi"):
        key_parser.add_argument(f"--{argument}", required=True)
    key_parser.add_argument("--input", action="append", default=[])
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--directory", required=True)
    validate_parser.add_argument("--key", required=True)
    publish_parser = commands.add_parser("publish")
    publish_parser.add_argument("--wheel", required=True)
    publish_parser.add_argument("--root", required=True)
    publish_parser.add_argument("--key", required=True)
    args = parser.parse_args()
    if args.command == "check-mount":
        try:
            visible = visible_mount(host_root=args.host_root, container=args.container)
        except OSError:
            visible = False
        raise SystemExit(0 if visible else 1)
    elif args.command == "environment":
        print(build_environment())
    elif args.command == "key":
        print(
            cache_key(
                commit=args.commit,
                image=args.image,
                arch=args.arch,
                namespace=args.namespace,
                inputs=args.input,
                abi=args.abi,
            )
        )
    elif args.command == "validate":
        print(cached_wheel(directory=args.directory, key=args.key))
    else:
        publish(wheel=args.wheel, root=args.root, key=args.key)


if __name__ == "__main__":
    main()
