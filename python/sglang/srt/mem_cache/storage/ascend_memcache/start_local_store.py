import argparse
import json
import logging
import os
import signal
import sys
import time
from typing import Any

from memcache_hybrid import DistributedObjectStore, LocalConfig

logger = logging.getLogger("ascend_memcache.start_local_store")

# Keys consumed by this launcher, not forwarded to LocalConfig.
_CTRL_KEYS = frozenset({"device_id", "init_bm"})


def _load_json_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return data


def _apply_local_config(config: LocalConfig, data: dict[str, Any]) -> list[str]:
    unknown: list[str] = []
    for key, value in data.items():
        if key in _CTRL_KEYS:
            continue
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            unknown.append(key)
    return unknown


def launch_local_store(config_path: str, block: bool = True) -> int:
    try:
        config_data = _load_json_config(config_path)
    except Exception as e:
        logger.error("Failed to load local store config from %s: %s", config_path, e)
        return 1

    local_cfg = LocalConfig()
    unknown = _apply_local_config(local_cfg, config_data)
    if unknown:
        logger.warning("Ignoring unknown LocalConfig keys: %s", unknown)

    device_id = int(config_data.get("device_id", 0))
    init_bm = bool(config_data.get("init_bm", True))

    store = DistributedObjectStore()
    ret = store.setup(local_cfg)
    if ret != 0:
        logger.error("DistributedObjectStore.setup failed, ret=%s", ret)
        return ret if isinstance(ret, int) else 2

    ret = store.init(device_id, init_bm)
    if ret != 0:
        logger.error(
            "DistributedObjectStore.init failed, device_id=%s init_bm=%s ret=%s",
            device_id,
            init_bm,
            ret,
        )
        return ret if isinstance(ret, int) else 3

    logger.info(
        "Local store started successfully: config=%s device_id=%s init_bm=%s",
        config_path,
        device_id,
        init_bm,
    )

    if not block:
        return 0

    stop = {"value": False}

    def _handle_sig(_signum, _frame):
        stop["value"] = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    logger.info("Local store is running. Press Ctrl+C to stop.")
    try:
        while not stop["value"]:
            time.sleep(1)
    finally:
        try:
            store.close()
        except Exception as e:
            logger.warning("DistributedObjectStore.close failed: %s", e)
    logger.info("Local store stopped.")
    return 0


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_dir, "localservice_config.json")

    parser = argparse.ArgumentParser(
        description="Launch Ascend MemCache local DistributedObjectStore via JSON."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=default_path,
        help=f"Path to local config JSON (default: {default_path})",
    )
    parser.add_argument(
        "--no-block",
        action="store_true",
        help="Initialize store and exit without keeping process alive.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return launch_local_store(args.config_path, block=not args.no_block)


if __name__ == "__main__":
    sys.exit(main())