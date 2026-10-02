# SPDX-License-Identifier: Apache-2.0
"""Measure the SM120 Kitchen INT8 default against the explicit 8192-row policy."""

import argparse
import json
import statistics

import torch

from sglang.multimodal_gen.runtime.layers.quantization import kitchen_int8
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config import (
    KitchenInt8Config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32700)
    parser.add_argument("--input-features", type=int, default=5376)
    parser.add_argument("--output-features", type=int, default=16128)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if min(args.rows, args.input_features, args.output_features, args.repeats) <= 0:
        parser.error("All dimensions and repeats must be positive")
    torch.manual_seed(12)
    method = kitchen_int8.KitchenInt8LinearMethod(
        KitchenInt8Config(), group_size=256, is_checkpoint_serialized=True
    )
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randint(
            -127,
            128,
            (args.output_features, args.input_features),
            device="cuda",
            dtype=torch.int8,
        ),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.rand(args.output_features, 1, device="cuda") * 0.002, requires_grad=False
    )
    inputs = torch.randn(
        args.rows, args.input_features, device="cuda", dtype=torch.bfloat16
    )
    bias = torch.randn(args.output_features, device="cuda", dtype=torch.bfloat16)
    records = []
    reference = None
    with torch.inference_mode():
        for override in (True, False, False, True):
            kitchen_int8._ROW_SPLIT_OVERRIDDEN = override
            kitchen_int8._MAX_ROWS_PER_CALL = 8192
            kitchen_int8._MIN_SPLIT_OUTPUT = 8192
            for _ in range(5):
                output = method.apply(layer, inputs, bias)
            if reference is None:
                reference = output.clone()
            assert torch.equal(output, reference)
            samples = []
            for _ in range(args.repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = method.apply(layer, inputs, bias)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            records.append(
                {
                    "arm": "8192" if override else "default",
                    "ms": samples,
                    "median_ms": statistics.median(samples),
                }
            )
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "args": vars(args),
                "exact": True,
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
