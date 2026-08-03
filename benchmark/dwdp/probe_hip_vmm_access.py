import torch

from sglang.srt.layers.moe.dwdp import hip_vmm


def probe(shareable: bool):
    device = torch.cuda.current_device()
    granularity = hip_vmm.get_allocation_granularity(
        device,
        shareable=shareable,
        recommended=True,
    )
    results = []
    for multiplier in (1, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384):
        size = granularity * multiplier
        handle = (
            hip_vmm.create_shareable_handle(size, device)
            if shareable
            else hip_vmm.create_local_handle(size, device)
        )
        mapping = None
        try:
            mapping = hip_vmm.map_handles(
                [handle],
                [size],
                device,
                alignment=granularity,
            )
        except Exception as error:
            results.append((size, False, repr(error)))
        else:
            results.append((size, True, ""))
        finally:
            if mapping is not None:
                mapping.close()
            hip_vmm.release_handle(handle)
    return granularity, results


def main():
    print({"local": probe(False), "shareable": probe(True)})


if __name__ == "__main__":
    main()
