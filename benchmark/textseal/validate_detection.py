#!/usr/bin/env python3
import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import types
from pathlib import Path

from transformers import AutoTokenizer

PINNED_TEXTSEAL_COMMIT = "c60d0d1da2e59f09a698438e218a07ee779b4616"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_official_detector(checkout: Path):
    package = checkout / "textseal" / "watermarking"
    textseal_package = types.ModuleType("textseal")
    textseal_package.__path__ = [str(checkout / "textseal")]
    watermarking_package = types.ModuleType("textseal.watermarking")
    watermarking_package.__path__ = [str(package)]
    sys.modules["textseal"] = textseal_package
    sys.modules["textseal.watermarking"] = watermarking_package

    config_module = _load_module("textseal.watermarking.config", package / "config.py")
    _load_module("textseal.watermarking.core", package / "core.py")
    detector_module = _load_module(
        "textseal.watermarking.detector", package / "detector.py"
    )
    return config_module.WatermarkConfig, detector_module.TextSealDetector


def _load_config(path: Path):
    data = json.loads(path.read_text())
    return data["providers"]["textseal"]


def _load_texts(path: Path, watermarked: bool):
    texts = []
    with path.open() as source:
        for line in source:
            record = json.loads(line)
            if record["watermarked"] is watermarked:
                texts.append(record["text"])
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--watermark-config", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--textseal-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    commit = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={args.textseal_checkout.resolve()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=args.textseal_checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_TEXTSEAL_COMMIT:
        raise RuntimeError(f"expected TextSeal {PINNED_TEXTSEAL_COMMIT}, got {commit}")

    WatermarkConfig, TextSealDetector = _load_official_detector(args.textseal_checkout)
    watermark_config = _load_config(args.watermark_config)
    config = WatermarkConfig(
        watermark_type="textseal",
        secret_key=int(watermark_config["key_a"]),
        secret_key_b=int(watermark_config["key_b"]),
        ngram=watermark_config.get("ngram", 2),
        mixing_alpha=watermark_config.get("mixing_probability", 0.5),
        scoring_method="v2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    detector = TextSealDetector(tokenizer, config, scoring_method="v2")

    positive = detector.detect_batch(_load_texts(args.samples, True))
    negative = detector.detect_batch(_load_texts(args.samples, False))
    report = {
        "textseal_commit": commit,
        "model": args.model,
        "scoring_method": "v2",
        "threshold": 0.01,
        "positive_samples": len(positive),
        "negative_samples": len(negative),
        "detection_rate": statistics.mean(item["detected"] for item in positive),
        "false_positive_rate": statistics.mean(item["detected"] for item in negative),
        "positive_median_p_value": statistics.median(
            item["p_value"] for item in positive
        ),
        "negative_median_p_value": statistics.median(
            item["p_value"] for item in negative
        ),
        "positive_scored_tokens": [item["n_tokens"] for item in positive],
        "negative_scored_tokens": [item["n_tokens"] for item in negative],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
