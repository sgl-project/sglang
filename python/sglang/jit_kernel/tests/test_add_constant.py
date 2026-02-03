import torch

from sglang.jit_kernel.add_constant import add_constant


def main():
    c = 1024
    src = torch.arange(0, 1024 + 1, dtype=torch.int32).cuda()
    dst = add_constant(src, c)
    assert torch.all(dst == src + c)


if __name__ == "__main__":
    main()
