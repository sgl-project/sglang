#!/usr/bin/env bash
set -euo pipefail

fail() { printf 'PPU CI: %s\n' "$*" >&2; exit 2; }

for name in PPU_CI_RUNTIME_IMAGE PPU_CI_SOURCE_SHA PPU_CI_WHEEL_DIR PPU_CI_ARTIFACT_DIR \
    PPU_CI_TARGET_MODEL_PATH PPU_CI_EAGLE3_MODEL_PATH PPU_CI_DFLASH_MODEL_PATH PPU_CI_EVAL_CACHE; do
    [[ -n "${!name:-}" ]] || fail "Set $name explicitly"
done
device=${PPU_CI_DEVICE:-${NVIDIA_VISIBLE_DEVICES:-}}
[[ "$device" =~ ^[0-9]+$ ]] || fail 'Select exactly one numeric device with PPU_CI_DEVICE'
# Canonicalize without integer overflow; 01 and 1 must share the same lock.
device="${device#"${device%%[!0]*}"}"
device=${device:-0}
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
repo=$(realpath -e "$repo")
[[ "$(git -C "$repo" rev-parse HEAD)" == "$PPU_CI_SOURCE_SHA" ]] || fail 'HEAD must equal PPU_CI_SOURCE_SHA exactly'
status=$(git -C "$repo" status --porcelain=v1 --untracked-files=normal --ignored --ignore-submodules=none)
[[ -z "$status" ]] || fail 'Source must be pristine, including untracked and ignored files'

# Exact inventories prevent unchecked extra wheels from being installed.
paths=$(python3 - "$repo" <<'PY'
import hashlib
import os
from pathlib import Path
import re
import sys


def fail(message):
    sys.exit("PPU CI: " + message)


def below(path, parent):
    return os.path.commonpath([str(path), str(parent)]) == str(parent)


def validate_symlinks(root):
    for directory, dirs, files in os.walk(
        root, followlinks=False, onerror=lambda error: fail(str(error))
    ):
        for name in dirs + files:
            link = Path(directory) / name
            if not link.is_symlink():
                continue
            if Path(os.readlink(link)).is_absolute():
                fail("Absolute symlink in read-only input: " + str(link))
            try:
                destination = link.resolve(strict=True)
            except (OSError, RuntimeError) as error:
                fail("Invalid symlink " + str(link) + ": " + str(error))
            if not below(destination, root):
                fail("Escaping symlink in read-only input: " + str(link))


try:
    repo = Path(sys.argv[1])
    names = ["WHEEL_DIR", "ARTIFACT_DIR", "TARGET_MODEL_PATH", "EAGLE3_MODEL_PATH", "DFLASH_MODEL_PATH", "EVAL_CACHE"]
    paths = []
    for name in names:
        raw = os.environ["PPU_CI_" + name]
        if not Path(raw).is_absolute():
            fail(name + " must be absolute")
        path = Path(raw).resolve(strict=True)
        if not path.is_dir():
            fail(name + " must be a directory")
        paths.append(path)
    for path in [repo] + paths:
        if "," in str(path):
            fail("Docker mount paths must not contain commas")
        if "\n" in str(path) or "\r" in str(path):
            fail("Docker mount paths must not contain newlines")
    wheels, artifacts, target, eagle3, dflash, cache = paths
    if below(wheels, repo) or below(artifacts, repo):
        fail("Wheels and artifacts must be outside source")
    if any(below(artifacts, path) for path in [wheels, target, eagle3, dflash, cache]):
        fail("Artifacts must not be inside read-only inputs")
    for root in [target, eagle3, dflash, cache]:
        validate_symlinks(root)
    for model in [target, eagle3, dflash]:
        if not (model / "config.json").is_file():
            fail("Missing model config.json: " + str(model))
    for name in ["test.jsonl", "test.jsonl.sha256"]:
        if not (cache / "mmlu" / name).is_file():
            fail("Missing eval cache mmlu/" + name)
    if (wheels / "source-sha.txt").read_text() != os.environ["PPU_CI_SOURCE_SHA"] + "\n":
        fail("Wheel source-sha.txt must match PPU_CI_SOURCE_SHA")
    if (wheels / "runtime-deps").is_symlink():
        fail("Wheel inventory must not use symlink directories")
    inventory = {"source-sha.txt": wheels / "source-sha.txt"}
    for wheel in wheels.rglob("*.whl"):
        name = "./" + str(wheel.relative_to(wheels))
        if not re.fullmatch(r"\./(?:runtime-deps/)?[A-Za-z0-9][A-Za-z0-9_.+-]*\.whl", name):
            fail("Unexpected wheel inventory entry: " + name)
        inventory[name] = wheel
    if not list(wheels.glob("*.whl")) or not list((wheels / "runtime-deps").glob("*.whl")):
        fail("Wheel inventory needs main and runtime-deps wheels")
    checksums = {}
    for line in (wheels / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match[2] in checksums:
            fail("Invalid or duplicate checksum inventory entry")
        checksums[match[2]] = match[1]
    if checksums.keys() != inventory.keys():
        fail("Checksum inventory must list source-sha.txt and every wheel exactly")
    for name, path in inventory.items():
        if path.is_symlink() or not path.is_file():
            fail("Wheel inventory requires regular files: " + name)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != checksums[name]:
            fail("Wheel checksum mismatch: " + name)
    print("\n".join(map(str, paths)))
except (OSError, ValueError) as error:
    fail(str(error))
PY
) || exit 2
mapfile -t inputs <<< "$paths"
wheels=${inputs[0]}
artifact_parent=${inputs[1]}

# Never unlink the shared lock: waiters must keep using the same inode.
exec 9>>"/tmp/sglang-ppu-ci-device-$device.lock"
flock -n 9 || fail "PPU device $device is busy"
private=$(mktemp -d "$artifact_parent/runtime.XXXXXXXX")
uid=$(id -u)
gid=$(id -g)
prefix="sglang-ppu-ci-$uid-$$-${private##*.}"
container=""
docker_pid=""
image_id=""
# Invoked by the EXIT trap, including an interrupted wait.
# shellcheck disable=SC2329
cleanup() {
    code=$?
    trap - EXIT
    trap '' INT TERM
    set +e
    cleanup_code=0
    if [[ -n "$container" ]]; then
        # Leave time for removal and ownership recovery before runner SIGKILL.
        docker stop --time=2 "$container" 9>&- >/dev/null 2>&1
        docker rm --force "$container" 9>&- >/dev/null 2>&1 || cleanup_code=1
    fi
    if [[ -n "$docker_pid" ]]; then
        kill "$docker_pid" 2>/dev/null
        wait "$docker_pid" 2>/dev/null
    fi
    # Recover ownership only after the runtime can no longer write artifacts.
    if [[ -n "$image_id" ]]; then
        docker run --rm --name "$prefix-ownership" --runtime=runc --network=none \
            --env NVIDIA_VISIBLE_DEVICES=void --user 0:0 \
            --mount "type=bind,source=$private,target=/artifacts" \
            --entrypoint chown "$image_id" -hR -- "$uid:$gid" /artifacts \
            9>&- >>"$private/ownership.log" 2>&1 || cleanup_code=1
    fi
    if [[ "$cleanup_code" != 0 ]]; then
        printf 'PPU CI: container cleanup or artifact ownership recovery failed\n' >&2
        [[ "$code" != 0 ]] || code=1
    fi
    exec 9>&-
    exit "$code"
}
trap 'cleanup' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! image_id=$(docker image inspect "$PPU_CI_RUNTIME_IMAGE" --format '{{.Id}}' 9>&- 2>/dev/null); then
    # Raw Docker diagnostics must stay outside all uploaded or mounted paths.
    pull_log=$(python3 - "$repo" "${inputs[@]}" <<'PY'
import os
from pathlib import Path
import sys
import tempfile

try:
    home = Path(os.environ.get("HOME", ""))
    if not home.is_absolute():
        sys.exit("PPU CI: Private image logs require an absolute HOME directory")
    home = home.resolve(strict=True)
    if not home.is_dir() or "\n" in str(home) or "\r" in str(home):
        sys.exit("PPU CI: Private image logs require a valid HOME directory")
    excluded = sys.argv[1:] + [
        os.environ[name] for name in ("RUNNER_TEMP", "GITHUB_WORKSPACE")
        if os.environ.get(name)
    ]
    for raw in excluded:
        path = Path(raw).resolve()
        if os.path.commonpath([str(home), str(path)]) == str(path):
            sys.exit("PPU CI: Private image logs require HOME outside temporary, source, artifact and input directories")
    directory = tempfile.mkdtemp(prefix=".sglang-ppu-ci-pull.", dir=str(home))
    fd, log = tempfile.mkstemp(prefix="pull-", suffix=".log", dir=directory)
    os.close(fd)
    print(log)
except (OSError, ValueError, RuntimeError):
    sys.exit("PPU CI: Private image logs could not be created under HOME")
PY
    ) || exit 2
    printf 'PPU CI: Private image pull log: %s\n' "$pull_log"
    docker pull "$PPU_CI_RUNTIME_IMAGE" 9>&- >"$pull_log" 2>&1 &
    docker_pid=$!
    wait "$docker_pid" || {
        code=$?
        printf '%s\n' 'PPU CI: Failed to pull runtime image' >&2
        exit "$code"
    }
    docker_pid=""
    image_id=$(docker image inspect "$PPU_CI_RUNTIME_IMAGE" --format '{{.Id}}' 9>&- 2>>"$pull_log") || {
        code=$?
        printf '%s\n' 'PPU CI: Failed to inspect runtime image after pull' >&2
        exit "$code"
    }
fi
[[ "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]] || fail 'Docker did not resolve an immutable image ID'
printf '%s\n' "$image_id" > "$private/runtime-image-id.txt"
printf '%s\n' "$PPU_CI_SOURCE_SHA" > "$private/source-sha.txt"
printf 'PPU CI artifacts: %s\n' "$private"

# Synchronous creation prevents containers from appearing after cleanup.
container="$prefix-test"
docker create --name "$container" --runtime=runc --shm-size=16g \
    --device=/dev/alixpu --device=/dev/alixpu_ctl --device=/dev/alixpu_sep \
    --device="/dev/alixpu_ppu$device" \
    --mount "type=bind,source=$repo,target=/source,readonly" \
    --mount "type=bind,source=$wheels,target=/wheels,readonly" \
    --mount "type=bind,source=${inputs[2]},target=/models/target,readonly" \
    --mount "type=bind,source=${inputs[3]},target=/models/eagle3,readonly" \
    --mount "type=bind,source=${inputs[4]},target=/models/dflash,readonly" \
    --mount "type=bind,source=${inputs[5]},target=/eval-cache,readonly" \
    --mount "type=bind,source=$private,target=/artifacts" \
    --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES=0 \
    --env PPU_CI_SOURCE_SHA="$PPU_CI_SOURCE_SHA" \
    --env PPU_CI_WHEEL_DIR=/wheels --env PPU_CI_ARTIFACT_DIR=/artifacts \
    --env PPU_CI_TARGET_MODEL_PATH=/models/target \
    --env PPU_CI_EAGLE3_MODEL_PATH=/models/eagle3 \
    --env PPU_CI_DFLASH_MODEL_PATH=/models/dflash \
    --env PPU_CI_EVAL_CACHE=/eval-cache --env SGLANG_IS_IN_CI=1 \
    --entrypoint bash "$image_id" /source/scripts/ci/ppu/build_and_test.sh test \
    9>&- >"$private/container-id.txt"
docker start --attach "$container" 9>&- >"$private/docker.log" 2>&1 &
docker_pid=$!
code=0
wait "$docker_pid" || code=$?
docker_pid=""
exit "$code"
