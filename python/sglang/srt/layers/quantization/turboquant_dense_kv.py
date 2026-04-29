from __future__ import annotations

import math
from dataclasses import dataclass

import torch


TURBOQUANT_DENSE_KV_PRESETS = {
    "latent_k8": {"bits": 8, "norm_correction": False},
    "latent_4bit_nc": {"bits": 4, "norm_correction": True},
    "latent_k3_nc": {"bits": 3, "norm_correction": True},
    "latent_2p5bit_nc": {"bits": 2.5, "norm_correction": True},
}

TURBOQUANT_2P5_GROUP_SIZE = 128
TURBOQUANT_2P5_HIGH_CHANNELS = 32


def _is_2p5_bits(bits: float) -> bool:
    return math.isclose(float(bits), 2.5)


def _check_2p5_dim(dim: int) -> None:
    if dim % TURBOQUANT_2P5_GROUP_SIZE != 0:
        raise ValueError(
            "2.5-bit TurboQuant requires the latent dimension to be a "
            f"multiple of {TURBOQUANT_2P5_GROUP_SIZE}; got {dim}."
        )


def _packed_2p5_bytes(dim: int) -> int:
    _check_2p5_dim(dim)
    group_bytes = (
        TURBOQUANT_2P5_HIGH_CHANNELS * 3
        + (TURBOQUANT_2P5_GROUP_SIZE - TURBOQUANT_2P5_HIGH_CHANNELS) * 2
    ) // 8
    return dim // TURBOQUANT_2P5_GROUP_SIZE * group_bytes


@dataclass(frozen=True)
class TurboQuantDenseKVConfig:
    latent_dim: int
    rope_dim: int
    preset: str = "latent_4bit_nc"
    seed: int = 42
    rope_dtype: torch.dtype = torch.bfloat16

    @property
    def bits(self) -> float:
        return float(TURBOQUANT_DENSE_KV_PRESETS[self.preset]["bits"])

    @property
    def norm_correction(self) -> bool:
        return bool(TURBOQUANT_DENSE_KV_PRESETS[self.preset]["norm_correction"])

    @property
    def latent_bytes(self) -> int:
        if self.bits == 8:
            return self.latent_dim
        if _is_2p5_bits(self.bits):
            return _packed_2p5_bytes(self.latent_dim) + 2
        return math.ceil(self.latent_dim * int(self.bits) / 8) + 2

    @property
    def rope_bytes(self) -> int:
        return self.rope_dim * torch.tensor([], dtype=self.rope_dtype).element_size()

    @property
    def slot_bytes(self) -> int:
        return self.latent_bytes + self.rope_bytes


def validate_turboquant_dense_kv_preset(preset: str) -> None:
    if preset not in TURBOQUANT_DENSE_KV_PRESETS:
        valid = ", ".join(sorted(TURBOQUANT_DENSE_KV_PRESETS))
        raise ValueError(
            f"Unknown TurboQuant dense KV preset {preset!r}; choices: {valid}."
        )


def pack_indices(indices: torch.Tensor, bits: float) -> torch.Tensor:
    indices = indices.to(torch.uint8)
    if _is_2p5_bits(bits):
        dim = indices.shape[-1]
        _check_2p5_dim(dim)
        groups = dim // TURBOQUANT_2P5_GROUP_SIZE
        x = indices.reshape(*indices.shape[:-1], groups, TURBOQUANT_2P5_GROUP_SIZE)
        high = pack_indices(x[..., :TURBOQUANT_2P5_HIGH_CHANNELS], 3)
        low = pack_indices(x[..., TURBOQUANT_2P5_HIGH_CHANNELS :], 2)
        return torch.cat((high, low), dim=-1).flatten(-2).contiguous()
    if bits == 8:
        return indices.contiguous()
    if bits == 2:
        if indices.shape[-1] % 4:
            indices = torch.nn.functional.pad(indices, (0, (-indices.shape[-1]) % 4))
        x = indices.reshape(*indices.shape[:-1], -1, 4)
        out = x[..., 0] | (x[..., 1] << 2) | (x[..., 2] << 4) | (x[..., 3] << 6)
        return out.contiguous()
    if bits == 4:
        if indices.shape[-1] % 2:
            indices = torch.nn.functional.pad(indices, (0, 1))
        lo = indices[..., 0::2]
        hi = indices[..., 1::2] << 4
        return (lo | hi).contiguous()
    if bits == 3:
        pad = (-indices.shape[-1]) % 8
        if pad:
            indices = torch.nn.functional.pad(indices, (0, pad))
        x = indices.reshape(*indices.shape[:-1], -1, 8)
        b0 = x[..., 0] | (x[..., 1] << 3) | ((x[..., 2] & 0x03) << 6)
        b1 = (
            (x[..., 2] >> 2)
            | (x[..., 3] << 1)
            | (x[..., 4] << 4)
            | ((x[..., 5] & 0x01) << 7)
        )
        b2 = (x[..., 5] >> 1) | (x[..., 6] << 2) | (x[..., 7] << 5)
        return torch.stack((b0, b1, b2), dim=-1).flatten(-2).contiguous()
    raise ValueError(f"Unsupported TurboQuant bit width: {bits}.")


def unpack_indices(packed: torch.Tensor, dim: int, bits: float) -> torch.Tensor:
    packed = packed.to(torch.uint8)
    if _is_2p5_bits(bits):
        _check_2p5_dim(dim)
        groups = dim // TURBOQUANT_2P5_GROUP_SIZE
        group_bytes = _packed_2p5_bytes(TURBOQUANT_2P5_GROUP_SIZE)
        x = packed[..., : groups * group_bytes].reshape(
            *packed.shape[:-1],
            groups,
            group_bytes,
        )
        high_bytes = TURBOQUANT_2P5_HIGH_CHANNELS * 3 // 8
        high = unpack_indices(x[..., :high_bytes], TURBOQUANT_2P5_HIGH_CHANNELS, 3)
        low = unpack_indices(
            x[..., high_bytes:],
            TURBOQUANT_2P5_GROUP_SIZE - TURBOQUANT_2P5_HIGH_CHANNELS,
            2,
        )
        out = torch.empty(
            *packed.shape[:-1],
            groups,
            TURBOQUANT_2P5_GROUP_SIZE,
            dtype=torch.uint8,
            device=packed.device,
        )
        out[..., :TURBOQUANT_2P5_HIGH_CHANNELS] = high
        out[..., TURBOQUANT_2P5_HIGH_CHANNELS :] = low
        return out.reshape(*packed.shape[:-1], dim).contiguous()
    if bits == 8:
        return packed[..., :dim].contiguous()
    if bits == 2:
        groups = math.ceil(dim / 4)
        x = packed[..., :groups]
        out = torch.empty(
            *packed.shape[:-1],
            groups,
            4,
            dtype=torch.uint8,
            device=packed.device,
        )
        out[..., 0] = x & 0x03
        out[..., 1] = (x >> 2) & 0x03
        out[..., 2] = (x >> 4) & 0x03
        out[..., 3] = (x >> 6) & 0x03
        return out.flatten(-2)[..., :dim].contiguous()
    if bits == 4:
        out_dim = math.ceil(dim / 2) * 2
        out = torch.empty(
            *packed.shape[:-1],
            out_dim,
            dtype=torch.uint8,
            device=packed.device,
        )
        out[..., 0::2] = packed & 0x0F
        out[..., 1::2] = packed >> 4
        return out[..., :dim].contiguous()
    if bits == 3:
        groups = math.ceil(dim / 8)
        bytes_needed = groups * 3
        x = packed[..., :bytes_needed].reshape(*packed.shape[:-1], groups, 3)
        out = torch.empty(
            *packed.shape[:-1],
            groups,
            8,
            dtype=torch.uint8,
            device=packed.device,
        )
        out[..., 0] = x[..., 0] & 0x07
        out[..., 1] = (x[..., 0] >> 3) & 0x07
        out[..., 2] = ((x[..., 0] >> 6) | ((x[..., 1] & 0x01) << 2)) & 0x07
        out[..., 3] = (x[..., 1] >> 1) & 0x07
        out[..., 4] = (x[..., 1] >> 4) & 0x07
        out[..., 5] = ((x[..., 1] >> 7) | ((x[..., 2] & 0x03) << 1)) & 0x07
        out[..., 6] = (x[..., 2] >> 2) & 0x07
        out[..., 7] = (x[..., 2] >> 5) & 0x07
        return out.flatten(-2)[..., :dim].contiguous()
    raise ValueError(f"Unsupported TurboQuant bit width: {bits}.")


def _fwht(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    y = x.contiguous()
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :].clone()
        b = y[:, :, 1, :].clone()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        h *= 2
    return y.reshape(*x.shape) / math.sqrt(n)


def _lloyd_max_normal(
    bits: int,
    dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = 1 << bits
    sigma = 1.0 / math.sqrt(dim)
    dtype = torch.float32
    q = torch.arange(1, n, dtype=dtype, device=device) / n
    boundaries = math.sqrt(2.0) * sigma * torch.erfinv(2.0 * q - 1.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    for _ in range(100):
        lo = torch.cat(
            (
                torch.tensor([-torch.inf], dtype=dtype, device=device),
                boundaries,
            )
        )
        hi = torch.cat(
            (
                boundaries,
                torch.tensor([torch.inf], dtype=dtype, device=device),
            )
        )
        lo_z = lo / sigma
        hi_z = hi / sigma
        lo_pdf = torch.where(
            torch.isfinite(lo_z),
            torch.exp(-0.5 * lo_z * lo_z) * inv_sqrt_2pi,
            0,
        )
        hi_pdf = torch.where(
            torch.isfinite(hi_z),
            torch.exp(-0.5 * hi_z * hi_z) * inv_sqrt_2pi,
            0,
        )
        lo_cdf = torch.where(
            torch.isfinite(lo_z),
            0.5 * (1.0 + torch.erf(lo_z / math.sqrt(2.0))),
            0,
        )
        hi_cdf = torch.where(
            torch.isfinite(hi_z),
            0.5 * (1.0 + torch.erf(hi_z / math.sqrt(2.0))),
            1,
        )
        centroids = (
            sigma * (lo_pdf - hi_pdf) / (hi_cdf - lo_cdf).clamp_min(1e-12)
        )
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])

    return centroids.contiguous(), boundaries.contiguous()


class TurboQuantDenseKVCodec:
    def __init__(self, config: TurboQuantDenseKVConfig, device: torch.device):
        validate_turboquant_dense_kv_preset(config.preset)
        self.config = config
        self.device = device
        self.bits = config.bits
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        signs = torch.randint(
            0,
            2,
            (config.latent_dim,),
            generator=generator,
            dtype=torch.int8,
        )
        self.signs1 = (
            signs.to(device=device, dtype=torch.float32) * 2 - 1
        ).contiguous()
        signs = torch.randint(
            0,
            2,
            (config.latent_dim,),
            generator=generator,
            dtype=torch.int8,
        )
        self.signs2 = (
            signs.to(device=device, dtype=torch.float32) * 2 - 1
        ).contiguous()
        if _is_2p5_bits(self.bits):
            self.centroids_high, self.boundaries_high = _lloyd_max_normal(
                3,
                config.latent_dim,
                device,
            )
            self.centroids_low, self.boundaries_low = _lloyd_max_normal(
                2,
                config.latent_dim,
                device,
            )
            self.centroids = None
            self.boundaries = None
        elif self.bits < 8:
            self.centroids, self.boundaries = _lloyd_max_normal(
                int(self.bits),
                config.latent_dim,
                device,
            )
            self.centroids_high = None
            self.boundaries_high = None
            self.centroids_low = None
            self.boundaries_low = None
        else:
            self.centroids = None
            self.boundaries = None
            self.centroids_high = None
            self.boundaries_high = None
            self.centroids_low = None
            self.boundaries_low = None

    @property
    def slot_bytes(self) -> int:
        return self.config.slot_bytes

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return _fwht(x * self.signs1) * self.signs2

    def inverse_rotate(self, y: torch.Tensor) -> torch.Tensor:
        return _fwht(y * self.signs2) * self.signs1

    def _quantize_rotated(self, rotated: torch.Tensor) -> torch.Tensor:
        if not _is_2p5_bits(self.bits):
            return torch.searchsorted(
                self.boundaries,
                rotated.contiguous(),
            ).to(torch.uint8)

        dim = rotated.shape[-1]
        _check_2p5_dim(dim)
        groups = dim // TURBOQUANT_2P5_GROUP_SIZE
        x = rotated.reshape(*rotated.shape[:-1], groups, TURBOQUANT_2P5_GROUP_SIZE)
        high = torch.searchsorted(
            self.boundaries_high,
            x[..., :TURBOQUANT_2P5_HIGH_CHANNELS].contiguous(),
        ).to(torch.uint8)
        low = torch.searchsorted(
            self.boundaries_low,
            x[..., TURBOQUANT_2P5_HIGH_CHANNELS :].contiguous(),
        ).to(torch.uint8)
        indices = torch.empty(
            *rotated.shape[:-1],
            groups,
            TURBOQUANT_2P5_GROUP_SIZE,
            dtype=torch.uint8,
            device=rotated.device,
        )
        indices[..., :TURBOQUANT_2P5_HIGH_CHANNELS] = high
        indices[..., TURBOQUANT_2P5_HIGH_CHANNELS :] = low
        return indices.reshape(*rotated.shape).contiguous()

    def _centroids_for_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if not _is_2p5_bits(self.bits):
            return self.centroids[indices.long()]

        dim = indices.shape[-1]
        _check_2p5_dim(dim)
        groups = dim // TURBOQUANT_2P5_GROUP_SIZE
        x = indices.reshape(*indices.shape[:-1], groups, TURBOQUANT_2P5_GROUP_SIZE)
        out = torch.empty(
            *indices.shape[:-1],
            groups,
            TURBOQUANT_2P5_GROUP_SIZE,
            dtype=torch.float32,
            device=indices.device,
        )
        out[..., :TURBOQUANT_2P5_HIGH_CHANNELS] = self.centroids_high[
            x[..., :TURBOQUANT_2P5_HIGH_CHANNELS].long()
        ]
        out[..., TURBOQUANT_2P5_HIGH_CHANNELS :] = self.centroids_low[
            x[..., TURBOQUANT_2P5_HIGH_CHANNELS :].long()
        ]
        return out.reshape(*indices.shape)

    def compress(self, latent: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        latent = latent.reshape(latent.shape[0], self.config.latent_dim)
        rope = rope.reshape(rope.shape[0], self.config.rope_dim)
        if self.bits == 8:
            latent_bytes = (
                latent.to(torch.float8_e4m3fn).contiguous().view(torch.uint8)
            )
        else:
            latent_f = latent.to(torch.float32)
            norms = torch.linalg.vector_norm(latent_f, dim=-1).clamp_min(1e-8)
            unit = latent_f / norms[:, None]
            indices = self._quantize_rotated(self.rotate(unit))
            if self.config.norm_correction:
                recon = self.inverse_rotate(self._centroids_for_indices(indices))
                norms = norms / torch.linalg.vector_norm(recon, dim=-1).clamp_min(
                    1e-8
                )
            latent_bytes = torch.cat(
                (
                    pack_indices(indices, self.bits),
                    norms.to(torch.float16)
                    .contiguous()
                    .view(torch.uint8)
                    .reshape(latent.shape[0], 2),
                ),
                dim=-1,
            )

        rope_bytes = rope.to(self.config.rope_dtype).contiguous().view(torch.uint8)
        return torch.cat((latent_bytes, rope_bytes), dim=-1).reshape(
            latent.shape[0],
            1,
            self.slot_bytes,
        )

    def decompress(
        self,
        compressed: torch.Tensor,
        dst_dtype: torch.dtype,
    ) -> torch.Tensor:
        compressed = compressed.reshape(compressed.shape[0], self.slot_bytes)
        if self.bits == 8:
            latent_end = self.config.latent_dim
            latent = (
                compressed[:, :latent_end]
                .contiguous()
                .view(torch.float8_e4m3fn)
                .to(dst_dtype)
            )
        else:
            packed_bytes = self.config.latent_bytes - 2
            packed = compressed[:, :packed_bytes]
            norm_bytes = compressed[:, packed_bytes : packed_bytes + 2]
            indices = unpack_indices(
                packed,
                self.config.latent_dim,
                self.bits,
            ).long()
            norms = (
                norm_bytes.contiguous()
                .view(torch.float16)
                .reshape(compressed.shape[0])
                .to(torch.float32)
            )
            latent = self.inverse_rotate(self._centroids_for_indices(indices)) * norms[
                :, None
            ]
            latent = latent.to(dst_dtype)

        rope_start = self.config.latent_bytes
        rope = (
            compressed[:, rope_start:]
            .contiguous()
            .view(self.config.rope_dtype)
            .reshape(compressed.shape[0], self.config.rope_dim)
            .to(dst_dtype)
        )
        return torch.cat((latent, rope), dim=-1).reshape(
            compressed.shape[0],
            1,
            self.config.latent_dim + self.config.rope_dim,
        )
