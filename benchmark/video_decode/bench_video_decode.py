# Benchmarks sglang's VideoDecoderWrapper on a clip: CPU torchcodec with the
# default parallel decoders versus the accelerator path (CUDA NVDEC or Intel
# XPU media engine through torchcodec-xpu). Reports wall and process CPU time.
#
# == Usage Examples ==
#
# python3 bench_video_decode.py --clip clip.mp4 --device xpu --num-frames 60
# python3 bench_video_decode.py --clip clip.mp4 --device cuda --dense

import argparse
import time

import numpy as np
import torch

from sglang.srt.utils.video_decoder import VideoDecoderWrapper


def _sync(device: str) -> None:
    if device != "cpu":
        torch.get_device_module(device.split(":")[0]).synchronize()


def _run(clip: str, device: str, indices: list[int], iters: int):
    with VideoDecoderWrapper(clip, device=device) as w:
        w.get_frames_as_tensor(indices[:1])  # warm up
        _sync(device)
        wall, cpu = time.perf_counter(), time.process_time()
        for _ in range(iters):
            out = w.get_frames_as_tensor(indices)
            _sync(device)
        wall = (time.perf_counter() - wall) / iters
        cpu = (time.process_time() - cpu) / iters
        return wall, cpu, str(out.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--device", default="xpu", help="cuda, xpu or xpu:N")
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--dense", action="store_true", help="decode every frame")
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    with VideoDecoderWrapper(args.clip, device="cpu") as probe:
        total, fps, (h, w) = len(probe), probe.avg_fps, probe.frame_shape
    if args.dense:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, args.num_frames, dtype=int).tolist()
    print(
        f"clip: {total} frames @ {fps:.1f} fps, {w}x{h}; decoding {len(indices)} frames"
    )
    print(
        f"{'requested -> decoded on':<28}{'wall ms':>10}{'CPU ms':>10}{'frames/s':>10}"
    )
    for device in ("cpu", args.device):
        wall, cpu, out_device = _run(args.clip, device, indices, args.iters)
        print(
            f"{device + ' -> ' + out_device:<28}"
            f"{wall * 1e3:>10.0f}{cpu * 1e3:>10.0f}{len(indices) / wall:>10.0f}"
        )


if __name__ == "__main__":
    main()
