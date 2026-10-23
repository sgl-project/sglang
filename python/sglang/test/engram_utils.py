"""Small real Engram modules for host-table component tests and benchmarks."""

from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.engram import Engram, EngramHasher, EngramLayout
from sglang.srt.model_loader.utils import set_default_torch_dtype

# Production host registrations live for the process lifetime.
_HOST_TABLES = []


def make_engram_layers(
    layout="per_rank", rows=1024, dim=32, layer_ids=(0, 2, 4), hidden_size=64
):
    config = SimpleNamespace(hidden_size=hidden_size, hc_mult=2, rms_norm_eps=1e-6)
    sizes = (rows // 4,) * 4
    spec = EngramLayout(
        3,
        layer_ids,
        (rows,) * len(layer_ids),
        ((sizes[:2], sizes[2:]),) * len(layer_ids),
        2,
        dim,
    )
    layers = []
    with (
        envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(True),
        envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.override(layout),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        for layer_id in layer_ids:
            layer = Engram(config, layer_id, spec)
            _HOST_TABLES.append(layer.embed.host_table)
            generator = torch.Generator(device="cpu").manual_seed(73 + layer_id)
            weight = torch.randn(rows, dim, generator=generator, device="cpu").to(
                torch.float8_e4m3fn
            )
            scale = torch.randint(
                123,
                130,
                (rows, dim // 32),
                generator=generator,
                dtype=torch.uint8,
                device="cpu",
            )
            if rows:
                scale[0] = 0
            layer.embed._load_rows(layer.embed.weight, weight)
            layer.embed._load_rows(layer.embed.scale, scale.view(torch.float8_e8m0fnu))
            layer.wkv.weight.data.normal_(std=0.01)
            layers.append(layer)
    return layers


def make_engram_hasher():
    """Two-token n-grams, three layers and four heads, without a tokenizer."""
    hasher = EngramHasher.__new__(EngramHasher)
    torch.nn.Module.__init__(hasher)
    hasher.max_ngram_size, hasher.pad_id = 2, 0
    hasher.image_token_id = None
    hasher.token_map = torch.arange(16, device="cuda")
    hasher.multipliers = torch.tensor([[3, 5]] * 3, device="cuda")
    hasher.primes = torch.tensor([[[251, 257, 263, 253]]] * 3, device="cuda")
    hasher.offsets = torch.tensor([[0, 251, 508, 771]] * 3, device="cuda")
    hasher.init_history(2, "cuda")
    return hasher
