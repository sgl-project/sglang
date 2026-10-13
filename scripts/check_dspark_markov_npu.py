#!/usr/bin/env python3
"""Small operator check for DSpark draft token IDs; no model weights loaded.

Run in the serving container with its source directory selected:
  PYTHONPATH=/home/l00993641/sglang/python:$PYTHONPATH python3 check_dspark_markov_npu.py
This checks the installed torch reference path, not a captured NPU graph.
"""

import argparse
import importlib
import inspect
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='npu:0')
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401; register the NPU device
    module = importlib.import_module('sglang.kernels.ops.speculative.dspark.markov_sample')
    sampler = module.MarkovSampleStep
    device = torch.device(args.device)
    print('MARKOV_SOURCE:', inspect.getfile(sampler), flush=True)
    print('TORCH_VERSION:', torch.__version__, 'DEVICE:', device, flush=True)
    failures = 0

    # Token IDs are transported as float32 bit patterns by the torch sampler.
    # Numerical float conversion or flushing subnormals can corrupt this path.
    for label, dev in [('CPU', torch.device('cpu')), ('NPU', device)]:
        source = torch.tensor([0, 1, 2, 15, 2619, 64469, 128799], dtype=torch.int32, device=dev)
        packed = torch.stack([torch.ones_like(source, dtype=torch.float32), source.view(torch.float32)], dim=1)
        recovered = packed[:, 1].contiguous().view(torch.int32)
        got = recovered.cpu().tolist()
        expected = source.cpu().tolist()
        passed = got == expected
        print('BITCAST', label, 'PASS' if passed else 'FAIL', 'expected=', expected, 'got=', got, flush=True)
        failures += not passed

    vocab, hot_id = 128800, 64469
    for temperature in (0.0, 1.0):
        for step_dtype in (torch.bfloat16, torch.float32):
            outputs = {}
            for label, dev in [('CPU', torch.device('cpu')), ('NPU', device)]:
                base = torch.full((1, vocab), -40.0, dtype=torch.float32, device=dev)
                base[0, hot_id] = 40.0
                bias = torch.zeros_like(base)
                values = []
                for offset in range(4):
                    result = sampler.torch(
                        base_logits=base,
                        bias=bias,
                        temperatures=torch.full((1,), temperature, dtype=torch.float32, device=dev),
                        key=3407,
                        offset=torch.tensor([offset], dtype=torch.int64, device=dev),
                        step=0,
                        step_dtype=step_dtype,
                        has_sampled_rows=True,
                    )
                    values.append(int(result.cpu().item()))
                outputs[label] = values
                passed = values == [hot_id] * 4
                print('DOMINANT_TOKEN', label, 'temperature=', temperature, 'dtype=', str(step_dtype),
                      'PASS' if passed else 'FAIL', 'expected=', hot_id, 'got=', values, flush=True)
                failures += not passed
                del base, bias
            print('CPU_NPU_AGREE:', outputs['CPU'] == outputs['NPU'], flush=True)
    print('RESULT:', 'PASS' if failures == 0 else 'FAIL', 'failed_checks=', failures, flush=True)
    print('A PASS excludes only these eager operator cases; graph replay and real-logit sampling still need checking.')
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
