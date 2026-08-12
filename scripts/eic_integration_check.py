"""Run a post-deployment check against SGLang's v0.5.17 EIC backend.

The script uses the same SDK calls and ``remote-eic.yaml`` fields as
``EICStorage``. It writes only uniquely prefixed temporary keys and removes
them on every normal or exceptional exit.

    python scripts/eic_integration_check.py
    python scripts/eic_integration_check.py --page-bytes $((512*1024)) --flood-gib 8
"""

import argparse
import os
import time
import traceback
import uuid

import eic
import torch
import yaml

FAILURES = []
PROBE = None


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)
    return ok


def _status_mask(outcome, count):
    codes = list(getattr(outcome, "status_codes", ()))
    if not codes:
        return [False] * count
    mask = [code == eic.StatusCode.SUCCESS for code in codes[:count]]
    return mask + [False] * (count - len(mask))


class Probe:
    def __init__(self, config, page_bytes, dtype=torch.bfloat16):
        numel = page_bytes // dtype.itemsize
        if numel <= 0:
            raise ValueError("--page-bytes is smaller than one tensor element")
        self.shape = (numel,)
        self.dtype = dtype
        self.page_bytes = numel * dtype.itemsize
        self.namespace = config.get("eic_namespace", "")
        self.run_id = uuid.uuid4().hex[:12]
        self.written = set()
        self.connection = self._connect(config)

    @staticmethod
    def _connect(config):
        remote_url = config.get("remote_url")
        if not isinstance(remote_url, str) or not remote_url.startswith("eic://"):
            raise ValueError("remote_url must be an eic:// URL")

        log_dir = config.get("eic_log_dir")
        if not log_dir:
            raise ValueError("eic_log_dir is required")
        os.makedirs(log_dir, exist_ok=True)

        init_option = eic.InitOption()
        init_option.log_dir = log_dir
        init_option.log_level = eic.LogLevel(config.get("eic_log_level", 2))
        init_option.transport_type = eic.TransportType(config.get("eic_trans_type", 3))
        init_option.flag_file = config.get("eic_flag_file")

        connection = eic.Client()
        ret = connection.init(
            config.get("eic_instance_id"), remote_url[len("eic://") :], init_option
        )
        if ret != 0:
            raise RuntimeError(f"EIC client initialization failed with code {ret}")
        return connection

    def key(self, index):
        return f"eic_integration_check/{self.run_id}/{index}"

    def page(self, seed):
        generator = torch.Generator().manual_seed(seed)
        return torch.randint(
            -128, 127, self.shape, generator=generator, dtype=torch.int16
        ).to(self.dtype)

    @staticmethod
    def _keys(keys):
        result = eic.StringVector()
        for key in keys:
            result.append(key)
        return result

    @staticmethod
    def _buffers(pages):
        result = eic.IOBuffers()
        for page in pages:
            result.append(page.data_ptr(), page.numel() * page.element_size(), False)
        return result

    def write(self, keys, pages):
        option = eic.SetOption()
        option.ns = self.namespace
        option.ttl_second = -1
        status, outcome = self.connection.mset(
            self._keys(keys), self._buffers(pages), option
        )
        mask = _status_mask(outcome, len(keys))
        if status == eic.StatusCode.SUCCESS and not mask:
            mask = [True] * len(keys)
        self.written.update(key for key, ok in zip(keys, mask) if ok)
        return mask

    def read(self, keys):
        pages = [torch.zeros(self.shape, dtype=self.dtype) for _ in keys]
        option = eic.GetOption()
        option.ns = self.namespace
        status, _, outcome = self.connection.mget(
            self._keys(keys), option, self._buffers(pages)
        )
        mask = _status_mask(outcome, len(keys))
        if status == eic.StatusCode.SUCCESS and not mask:
            mask = [True] * len(keys)
        return pages, mask

    def exists(self, keys):
        option = eic.ExistOption()
        option.ns = self.namespace
        status, outcome = self.connection.mexist(self._keys(keys), option)
        mask = _status_mask(outcome, len(keys))
        if status == eic.StatusCode.SUCCESS and not mask:
            mask = [True] * len(keys)
        return mask

    def delete(self, keys):
        option = eic.DelOption()
        option.ns = self.namespace
        status, outcome = self.connection.mdel(self._keys(keys), option)
        mask = _status_mask(outcome, len(keys))
        if status == eic.StatusCode.SUCCESS and not mask:
            mask = [True] * len(keys)
        for key, ok in zip(keys, mask):
            if ok:
                self.written.discard(key)
        return mask

    def cleanup(self):
        keys = sorted(self.written)
        for start in range(0, len(keys), 256):
            self.delete(keys[start : start + 256])
        print(f"cleaned up {len(keys)} temporary keys")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "REMOTE_EIC_YAML", "/sgl-workspace/config/remote-eic.yaml"
        ),
        help="EIC YAML config; defaults to REMOTE_EIC_YAML or SGLang's standard path",
    )
    parser.add_argument("--page-bytes", type=int, default=1 << 20)
    parser.add_argument("--pages", type=int, default=32, help="pages per probe batch")
    parser.add_argument(
        "--flood-gib",
        type=float,
        default=0,
        help="write this much temporary data, then check exists/get consistency",
    )
    args = parser.parse_args()
    if args.pages < 2:
        parser.error("--pages must be at least 2")
    if args.flood_gib < 0:
        parser.error("--flood-gib must be non-negative")
    return args


def main():
    args = parse_args()
    config_path = os.path.abspath(args.config)
    if not check("config file present", os.path.isfile(config_path), config_path):
        return 1
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    global PROBE
    start = time.perf_counter()
    probe = PROBE = Probe(config, args.page_bytes)
    check(
        "client init",
        True,
        f"ns={probe.namespace or '<default>'} page={probe.page_bytes}B "
        f"{time.perf_counter() - start:.1f}s",
    )

    count = args.pages
    keys = [probe.key(index) for index in range(count)]
    check("absent key reports miss", not any(probe.exists(keys)))

    pages = [probe.page(index) for index in range(count)]
    start = time.perf_counter()
    write_mask = probe.write(keys, pages)
    elapsed = time.perf_counter() - start
    check(
        "write",
        all(write_mask),
        f"{count} pages {count * probe.page_bytes / elapsed / 2**30:.2f} GiB/s",
    )
    check("exists after write", all(probe.exists(keys)))

    start = time.perf_counter()
    values, read_mask = probe.read(keys)
    elapsed = time.perf_counter() - start
    check(
        "read",
        all(read_mask),
        f"{count} pages {count * probe.page_bytes / elapsed / 2**30:.2f} GiB/s",
    )
    mismatches = [
        index
        for index, ok in enumerate(read_mask)
        if ok and not torch.equal(values[index], pages[index])
    ]
    check("read-back is byte-identical", not mismatches, f"mismatched={mismatches[:4]}")

    half = count // 2
    ghosts = [probe.key(f"ghost{index}") for index in range(half)]
    _, mixed = probe.read(keys[:half] + ghosts)
    check(
        "mixed batch reports per-key hits",
        mixed[:half] == [True] * half and mixed[half:] == [False] * len(ghosts),
        f"hits={sum(mixed)}/{len(mixed)}",
    )

    replacement = probe.page(10**6)
    probe.write(keys[:1], [replacement])
    values, mask = probe.read(keys[:1])
    check("overwrite wins", mask[0] and torch.equal(values[0], replacement))

    victims = keys[:4]
    delete_mask = probe.delete(victims)
    _, mask = probe.read(victims)
    check("delete removes the value", all(delete_mask) and not any(mask))

    if args.flood_gib > 0:
        flood_pages = int(args.flood_gib * 2**30 / probe.page_bytes)
        start = time.perf_counter()
        for offset in range(0, flood_pages, count):
            batch = min(count, flood_pages - offset)
            flood_keys = [probe.key(f"flood{offset + index}") for index in range(batch)]
            flood_values = [
                probe.page(10**7 + offset + index) for index in range(batch)
            ]
            probe.write(flood_keys, flood_values)
        elapsed = time.perf_counter() - start
        print(
            f"flooded {flood_pages} pages ({args.flood_gib} GiB) in {elapsed:.1f}s "
            f"{flood_pages * probe.page_bytes / elapsed / 2**30:.2f} GiB/s"
        )

        survivors = keys[4:]
        exists = probe.exists(survivors)
        _, mask = probe.read(survivors)
        phantoms = [
            key
            for key, present, read in zip(survivors, exists, mask)
            if present and not read
        ]
        check(
            "eviction keeps exists and get consistent",
            not phantoms,
            f"survived={sum(mask)}/{len(survivors)} phantom={len(phantoms)}",
        )

    return 1 if FAILURES else 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        if PROBE is not None:
            PROBE.cleanup()
        print(
            "\nall checks passed"
            if exit_code == 0
            else f"\n{len(FAILURES) or 'aborted'} failed"
        )
    raise SystemExit(exit_code)
