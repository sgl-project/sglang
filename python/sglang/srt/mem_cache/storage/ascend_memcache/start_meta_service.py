import argparse
import json
import logging
import os
import sys
from typing import Any

from memcache_hybrid import MetaConfig, MetaService

logger = logging.getLogger("ascend_memcache.start_meta_service")


def _load_json_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return data


def _apply_meta_config(config: MetaConfig, data: dict[str, Any]) -> list[str]:
    unknown: list[str] = []
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            unknown.append(key)
    return unknown


def launch_meta_service(config_path: str) -> int:
    try:
        config_data = _load_json_config(config_path)
    except Exception as e:
        logger.error("Failed to load meta service config from %s: %s", config_path, e)
        return 1

    meta_cfg = MetaConfig()
    unknown = _apply_meta_config(meta_cfg, config_data)
    if unknown:
        logger.warning("Ignoring unknown MetaConfig keys: %s", unknown)

    try:
        setup_ret = MetaService.setup(meta_cfg)
        if isinstance(setup_ret, int) and setup_ret != 0:
            logger.error("MetaService.setup failed, ret=%s", setup_ret)
            return setup_ret
        logger.info("MetaService setup succeeded with config=%s", config_path)
        MetaService.main()
        return 0
    except KeyboardInterrupt:
        logger.info("MetaService interrupted by user.")
        return 0
    except Exception as e:
        logger.error("MetaService failed to run: %s", e)
        return 2


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_dir, "metaservice_config.json")

    parser = argparse.ArgumentParser(
        description="Launch Ascend MemCache MetaService via JSON."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=default_path,
        help=f"Path to meta service JSON config (default: {default_path})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return launch_meta_service(args.config_path)


if __name__ == "__main__":
    sys.exit(main())
