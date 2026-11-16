"""Golden pin of the unified L3 key scheme without speculative decoding.

Every namespace digest, object key, chunk span and stored byte below was
recorded from the scheme as it first shipped. A deployment without speculative
decoding must keep producing exactly these, whatever is added to the scheme
later: objects already in a shared store are addressed by them.

The literals are data, not expectations to maintain. If a case fails, the
change under test moved a key or a byte of an existing deployment; fix the
change, do not regenerate. The file is self-contained on purpose (its own
fakes, no imports from other tests), so edits elsewhere cannot move what it
pins.
"""

import ctypes
import hashlib
import os
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_key_scheme import namespace_digest, plan_unified_kv
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MODEL_ID = "golden/model"

# name -> (plan_unified_kv arguments, (page_size, head_dim, itemsize))
_MHA = dict(rank_replicated=False, attn_cp_size=1, is_final_stage=True)
_MLA = dict(rank_replicated=True, local_kv_heads=0, attn_cp_size=1, is_final_stage=True)
PLAN_CASES = {
    "mha_tp1_no_partition": (
        dict(_MHA, dtype="bfloat16", local_kv_heads=8, attn_tp_rank=0, attn_tp_size=1)
        | dict(start_layer=0, end_layer=12),
        (16, 64, 2),
    ),
    "mha_tp1_partitioned": (
        dict(_MHA, dtype="bfloat16", local_kv_heads=8, attn_tp_rank=0, attn_tp_size=1)
        | dict(start_layer=0, end_layer=12, head_group=2, layer_partition=4),
        (16, 64, 2),
    ),
    "mha_tp2_rank1_partitioned": (
        dict(_MHA, dtype="bfloat16", local_kv_heads=4, attn_tp_rank=1, attn_tp_size=2)
        | dict(start_layer=0, end_layer=12, head_group=2, layer_partition=4),
        (16, 64, 2),
    ),
    "mha_tp2_rank1_no_partition": (
        dict(_MHA, dtype="bfloat16", local_kv_heads=4, attn_tp_rank=1, attn_tp_size=2)
        | dict(start_layer=0, end_layer=12),
        (16, 64, 2),
    ),
    "mha_tp4_rank3_partitioned": (
        dict(_MHA, dtype="bfloat16", local_kv_heads=2, attn_tp_rank=3, attn_tp_size=4)
        | dict(start_layer=0, end_layer=12, head_group=2, layer_partition=4),
        (16, 64, 2),
    ),
    "mha_tp2_rank1_cp2": (
        dict(_MHA, dtype="bfloat16", local_kv_heads=4, attn_tp_rank=1, attn_tp_size=2)
        | dict(start_layer=0, end_layer=12, head_group=2, layer_partition=4)
        | dict(attn_cp_size=2),
        (16, 64, 2),
    ),
    "mha_pp2_stage0": (
        dict(_MHA, dtype="bfloat16", local_kv_heads=4, attn_tp_rank=1, attn_tp_size=2)
        | dict(start_layer=0, end_layer=8, head_group=2, layer_partition=4)
        | dict(is_final_stage=False),
        (16, 64, 2),
    ),
    "mha_pp2_stage1_short_tail": (
        dict(_MHA, dtype="bfloat16", local_kv_heads=4, attn_tp_rank=1, attn_tp_size=2)
        | dict(start_layer=8, end_layer=13, head_group=2, layer_partition=4),
        (16, 64, 2),
    ),
    "mha_pp2_stage1_no_layer_partition": (
        dict(_MHA, dtype="bfloat16", local_kv_heads=4, attn_tp_rank=1, attn_tp_size=2)
        | dict(start_layer=8, end_layer=13, head_group=2),
        (16, 64, 2),
    ),
    "mha_fp8_e4m3": (
        dict(_MHA, dtype="float8_e4m3fn", local_kv_heads=4, attn_tp_rank=0)
        | dict(attn_tp_size=2, start_layer=0, end_layer=12, head_group=2)
        | dict(layer_partition=4),
        (16, 64, 1),
    ),
    "mha_fp8_e5m2": (
        dict(_MHA, dtype="float8_e5m2", local_kv_heads=4, attn_tp_rank=0)
        | dict(attn_tp_size=2, start_layer=0, end_layer=12, head_group=2)
        | dict(layer_partition=4),
        (16, 64, 1),
    ),
    "mla_tp1_no_partition": (
        dict(_MLA, dtype="bfloat16", attn_tp_rank=0, attn_tp_size=1)
        | dict(start_layer=0, end_layer=27),
        (64, 576, 2),
    ),
    "mla_tp2_rank1_partitioned": (
        dict(_MLA, dtype="bfloat16", attn_tp_rank=1, attn_tp_size=2)
        | dict(start_layer=0, end_layer=27, layer_partition=8),
        (64, 576, 2),
    ),
    "mla_pp2_stage1": (
        dict(_MLA, dtype="bfloat16", attn_tp_rank=0, attn_tp_size=1)
        | dict(start_layer=16, end_layer=27, layer_partition=8),
        (64, 576, 2),
    ),
}

GOLDEN_PLANS = {
    "mha_tp1_no_partition": {
        "digest": "ukv1-b737592f1c6f4054",
        "suffixes": ["ukv1-b737592f1c6f4054_L0-12_H0"],
        "layer_ranges": [(0, 12)],
        "head_ranges": [(0, 8)],
        "spans": [(0, 393216)],
        "bytes_per_page": 393216,
    },
    "mha_tp1_partitioned": {
        "digest": "ukv1-c6f83f3d35128b45",
        "suffixes": [
            "ukv1-c6f83f3d35128b45_L0-4_H0",
            "ukv1-c6f83f3d35128b45_L0-4_H1",
            "ukv1-c6f83f3d35128b45_L0-4_H2",
            "ukv1-c6f83f3d35128b45_L0-4_H3",
            "ukv1-c6f83f3d35128b45_L4-8_H0",
            "ukv1-c6f83f3d35128b45_L4-8_H1",
            "ukv1-c6f83f3d35128b45_L4-8_H2",
            "ukv1-c6f83f3d35128b45_L4-8_H3",
            "ukv1-c6f83f3d35128b45_L8-12_H0",
            "ukv1-c6f83f3d35128b45_L8-12_H1",
            "ukv1-c6f83f3d35128b45_L8-12_H2",
            "ukv1-c6f83f3d35128b45_L8-12_H3",
        ],
        "layer_ranges": [(0, 4), (4, 8), (8, 12)],
        "head_ranges": [(0, 2), (2, 4), (4, 6), (6, 8)],
        "spans": [
            (0, 32768),
            (98304, 32768),
            (196608, 32768),
            (294912, 32768),
            (32768, 32768),
            (131072, 32768),
            (229376, 32768),
            (327680, 32768),
            (65536, 32768),
            (163840, 32768),
            (262144, 32768),
            (360448, 32768),
        ],
        "bytes_per_page": 393216,
    },
    "mha_tp2_rank1_partitioned": {
        "digest": "ukv1-c6f83f3d35128b45",
        "suffixes": [
            "ukv1-c6f83f3d35128b45_L0-4_H2",
            "ukv1-c6f83f3d35128b45_L0-4_H3",
            "ukv1-c6f83f3d35128b45_L4-8_H2",
            "ukv1-c6f83f3d35128b45_L4-8_H3",
            "ukv1-c6f83f3d35128b45_L8-12_H2",
            "ukv1-c6f83f3d35128b45_L8-12_H3",
        ],
        "layer_ranges": [(0, 4), (4, 8), (8, 12)],
        "head_ranges": [(0, 2), (2, 4)],
        "spans": [
            (0, 32768),
            (98304, 32768),
            (32768, 32768),
            (131072, 32768),
            (65536, 32768),
            (163840, 32768),
        ],
        "bytes_per_page": 196608,
    },
    "mha_tp2_rank1_no_partition": {
        "digest": "ukv1-136d8b42aacae3bf",
        "suffixes": ["ukv1-136d8b42aacae3bf_L0-12_H1"],
        "layer_ranges": [(0, 12)],
        "head_ranges": [(0, 4)],
        "spans": [(0, 196608)],
        "bytes_per_page": 196608,
    },
    "mha_tp4_rank3_partitioned": {
        "digest": "ukv1-c6f83f3d35128b45",
        "suffixes": [
            "ukv1-c6f83f3d35128b45_L0-4_H3",
            "ukv1-c6f83f3d35128b45_L4-8_H3",
            "ukv1-c6f83f3d35128b45_L8-12_H3",
        ],
        "layer_ranges": [(0, 4), (4, 8), (8, 12)],
        "head_ranges": [(0, 2)],
        "spans": [(0, 32768), (32768, 32768), (65536, 32768)],
        "bytes_per_page": 98304,
    },
    "mha_tp2_rank1_cp2": {
        "digest": "ukv1-c6f83f3d35128b45",
        "suffixes": [
            "ukv1-c6f83f3d35128b45_L0-4_H2",
            "ukv1-c6f83f3d35128b45_L0-4_H3",
            "ukv1-c6f83f3d35128b45_L4-8_H2",
            "ukv1-c6f83f3d35128b45_L4-8_H3",
            "ukv1-c6f83f3d35128b45_L8-12_H2",
            "ukv1-c6f83f3d35128b45_L8-12_H3",
        ],
        "layer_ranges": [(0, 4), (4, 8), (8, 12)],
        "head_ranges": [(0, 2), (2, 4)],
        "spans": [
            (0, 32768),
            (98304, 32768),
            (32768, 32768),
            (131072, 32768),
            (65536, 32768),
            (163840, 32768),
        ],
        "bytes_per_page": 196608,
    },
    "mha_pp2_stage0": {
        "digest": "ukv1-c6f83f3d35128b45",
        "suffixes": [
            "ukv1-c6f83f3d35128b45_L0-4_H2",
            "ukv1-c6f83f3d35128b45_L0-4_H3",
            "ukv1-c6f83f3d35128b45_L4-8_H2",
            "ukv1-c6f83f3d35128b45_L4-8_H3",
        ],
        "layer_ranges": [(0, 4), (4, 8)],
        "head_ranges": [(0, 2), (2, 4)],
        "spans": [(0, 32768), (65536, 32768), (32768, 32768), (98304, 32768)],
        "bytes_per_page": 131072,
    },
    "mha_pp2_stage1_short_tail": {
        "digest": "ukv1-c6f83f3d35128b45",
        "suffixes": [
            "ukv1-c6f83f3d35128b45_L8-12_H2",
            "ukv1-c6f83f3d35128b45_L8-12_H3",
            "ukv1-c6f83f3d35128b45_L12-13_H2",
            "ukv1-c6f83f3d35128b45_L12-13_H3",
        ],
        "layer_ranges": [(0, 4), (4, 5)],
        "head_ranges": [(0, 2), (2, 4)],
        "spans": [(0, 32768), (40960, 32768), (32768, 8192), (73728, 8192)],
        "bytes_per_page": 81920,
    },
    "mha_pp2_stage1_no_layer_partition": {
        "digest": "ukv1-a9dc40a96fd88944",
        "suffixes": [
            "ukv1-a9dc40a96fd88944_L8-13_H2",
            "ukv1-a9dc40a96fd88944_L8-13_H3",
        ],
        "layer_ranges": [(0, 5)],
        "head_ranges": [(0, 2), (2, 4)],
        "spans": [(0, 40960), (40960, 40960)],
        "bytes_per_page": 81920,
    },
    "mha_fp8_e4m3": {
        "digest": "ukv1-6b729a2cefe2f8d8",
        "suffixes": [
            "ukv1-6b729a2cefe2f8d8_L0-4_H0",
            "ukv1-6b729a2cefe2f8d8_L0-4_H1",
            "ukv1-6b729a2cefe2f8d8_L4-8_H0",
            "ukv1-6b729a2cefe2f8d8_L4-8_H1",
            "ukv1-6b729a2cefe2f8d8_L8-12_H0",
            "ukv1-6b729a2cefe2f8d8_L8-12_H1",
        ],
        "layer_ranges": [(0, 4), (4, 8), (8, 12)],
        "head_ranges": [(0, 2), (2, 4)],
        "spans": [
            (0, 16384),
            (49152, 16384),
            (16384, 16384),
            (65536, 16384),
            (32768, 16384),
            (81920, 16384),
        ],
        "bytes_per_page": 98304,
    },
    "mha_fp8_e5m2": {
        "digest": "ukv1-245a27fa410f6292",
        "suffixes": [
            "ukv1-245a27fa410f6292_L0-4_H0",
            "ukv1-245a27fa410f6292_L0-4_H1",
            "ukv1-245a27fa410f6292_L4-8_H0",
            "ukv1-245a27fa410f6292_L4-8_H1",
            "ukv1-245a27fa410f6292_L8-12_H0",
            "ukv1-245a27fa410f6292_L8-12_H1",
        ],
        "layer_ranges": [(0, 4), (4, 8), (8, 12)],
        "head_ranges": [(0, 2), (2, 4)],
        "spans": [
            (0, 16384),
            (49152, 16384),
            (16384, 16384),
            (65536, 16384),
            (32768, 16384),
            (81920, 16384),
        ],
        "bytes_per_page": 98304,
    },
    "mla_tp1_no_partition": {
        "digest": "ukv1-3247e0081701b5bf",
        "suffixes": ["ukv1-3247e0081701b5bf_L0-27"],
        "layer_ranges": [(0, 27)],
        "head_ranges": [(0, 1)],
        "spans": [(0, 1990656)],
        "bytes_per_page": 1990656,
    },
    "mla_tp2_rank1_partitioned": {
        "digest": "ukv1-64940e04b4e857d2",
        "suffixes": [
            "ukv1-64940e04b4e857d2_L0-8",
            "ukv1-64940e04b4e857d2_L8-16",
            "ukv1-64940e04b4e857d2_L16-24",
            "ukv1-64940e04b4e857d2_L24-27",
        ],
        "layer_ranges": [(0, 8), (8, 16), (16, 24), (24, 27)],
        "head_ranges": [(0, 1)],
        "spans": [(0, 589824), (589824, 589824), (1179648, 589824), (1769472, 221184)],
        "bytes_per_page": 1990656,
    },
    "mla_pp2_stage1": {
        "digest": "ukv1-64940e04b4e857d2",
        "suffixes": ["ukv1-64940e04b4e857d2_L16-24", "ukv1-64940e04b4e857d2_L24-27"],
        "layer_ranges": [(0, 8), (8, 11)],
        "head_ranges": [(0, 1)],
        "spans": [(0, 589824), (589824, 221184)],
        "bytes_per_page": 811008,
    },
}

GOLDEN_CONFIGS = {
    "mha_tp2_rank1_pp2_stage1": {
        "suffixes": [
            "ukv1-8f579cfe2149c0bb_L4-6_H2",
            "ukv1-8f579cfe2149c0bb_L4-6_H3",
            "ukv1-8f579cfe2149c0bb_L6-7_H2",
            "ukv1-8f579cfe2149c0bb_L6-7_H3",
        ],
        "spans": [(0, 16384), (24576, 16384), (16384, 8192), (40960, 8192)],
        "is_storage_writer": True,
        "is_rank_replicated": False,
        "model_name": "golden/model",
    },
    "mha_tp2_rank0_standalone": {
        "suffixes": [
            "ukv1-8f579cfe2149c0bb_L4-6_H0",
            "ukv1-8f579cfe2149c0bb_L4-6_H1",
            "ukv1-8f579cfe2149c0bb_L6-7_H0",
            "ukv1-8f579cfe2149c0bb_L6-7_H1",
        ],
        "spans": [(0, 16384), (24576, 16384), (16384, 8192), (40960, 8192)],
        "is_storage_writer": True,
        "is_rank_replicated": False,
        "model_name": "golden/model",
    },
    "mha_tp2_rank1_cp_rank1": {
        "suffixes": [
            "ukv1-8f579cfe2149c0bb_L4-6_H2",
            "ukv1-8f579cfe2149c0bb_L4-6_H3",
            "ukv1-8f579cfe2149c0bb_L6-7_H2",
            "ukv1-8f579cfe2149c0bb_L6-7_H3",
        ],
        "spans": [(0, 16384), (24576, 16384), (16384, 8192), (40960, 8192)],
        "is_storage_writer": False,
        "is_rank_replicated": False,
        "model_name": "golden/model",
    },
    "mha_tp1_no_partition": {
        "suffixes": ["ukv1-43953e0c86f8d618_L4-7_H0"],
        "spans": [(0, 98304)],
        "is_storage_writer": True,
        "is_rank_replicated": False,
        "model_name": "golden/model",
    },
    "mla_tp2_rank0": {
        "suffixes": ["ukv1-94f0d71a234fc4ae_L4-6", "ukv1-94f0d71a234fc4ae_L6-7"],
        "spans": [(0, 4096), (4096, 2048)],
        "is_storage_writer": True,
        "is_rank_replicated": True,
        "model_name": "golden/model",
    },
    "mla_tp2_rank1": {
        "suffixes": ["ukv1-94f0d71a234fc4ae_L4-6", "ukv1-94f0d71a234fc4ae_L6-7"],
        "spans": [(0, 4096), (4096, 2048)],
        "is_storage_writer": False,
        "is_rank_replicated": True,
        "model_name": "golden/model",
    },
}

GOLDEN_PAGE_KEYS = [
    "9875a2a003e2979c437da49d086018662cb9c5a23b47f3ec3804db119c81353d",
    "21c05bea24929e9c5e1d5a8fca136afc12069a0b22c64a2221cd6149f4399214",
]

GOLDEN_FILES = {
    "mha": {
        "21c05bea24929e9c5e1d5a8fca136afc12069a0b22c64a2221cd6149f4399214_ukv1-542472633bab7949_L0-2_H2.bin": (
            256,
            "537ecb657142dde007a965782c88a476472cc16647f91079ec642b4f35f1aedb",
        ),
        "21c05bea24929e9c5e1d5a8fca136afc12069a0b22c64a2221cd6149f4399214_ukv1-542472633bab7949_L0-2_H3.bin": (
            256,
            "5feef6b4fed0c250d7e76a86fabe99cb697dab859d97556d1e0228bdc38cdd58",
        ),
        "21c05bea24929e9c5e1d5a8fca136afc12069a0b22c64a2221cd6149f4399214_ukv1-542472633bab7949_L2-4_H2.bin": (
            256,
            "b7d2dfb101091e65ca507ba0392d3a74b66464169a2cd0cd17d3b02ade83052e",
        ),
        "21c05bea24929e9c5e1d5a8fca136afc12069a0b22c64a2221cd6149f4399214_ukv1-542472633bab7949_L2-4_H3.bin": (
            256,
            "350292c36e0cfbf24846acf9a33fc80f1001d84126dfc196d522c848c61338a3",
        ),
        "9875a2a003e2979c437da49d086018662cb9c5a23b47f3ec3804db119c81353d_ukv1-542472633bab7949_L0-2_H2.bin": (
            256,
            "49b3e8e02da97716d4d5f5da8e92e7ab9532186015ca874c84cf7b386e418a2f",
        ),
        "9875a2a003e2979c437da49d086018662cb9c5a23b47f3ec3804db119c81353d_ukv1-542472633bab7949_L0-2_H3.bin": (
            256,
            "436b43090dd46628913721678c411fb39a318e237d8c8172e20d4af9e7d1aa4f",
        ),
        "9875a2a003e2979c437da49d086018662cb9c5a23b47f3ec3804db119c81353d_ukv1-542472633bab7949_L2-4_H2.bin": (
            256,
            "0884424cc940d1ff3ba90a7e8110e4e2e40e45c3f2fae4dc6fd5ce65961e3f6f",
        ),
        "9875a2a003e2979c437da49d086018662cb9c5a23b47f3ec3804db119c81353d_ukv1-542472633bab7949_L2-4_H3.bin": (
            256,
            "96fd62d546401c123f08130292aea63dc454915f4b054fc346562b3de8518a64",
        ),
    },
    "mla": {
        "21c05bea24929e9c5e1d5a8fca136afc12069a0b22c64a2221cd6149f4399214_ukv1-20639290a26e80f6_L0-2.bin": (
            128,
            "78be6166ff3790a50cb1e3ea9357c3fe8fb36caaa29d058b1961976506622861",
        ),
        "21c05bea24929e9c5e1d5a8fca136afc12069a0b22c64a2221cd6149f4399214_ukv1-20639290a26e80f6_L2-4.bin": (
            128,
            "e1d2579c667f6a40ee5b385a55d7095ebd77b80799fe73cf67583a2517504788",
        ),
        "9875a2a003e2979c437da49d086018662cb9c5a23b47f3ec3804db119c81353d_ukv1-20639290a26e80f6_L0-2.bin": (
            128,
            "d7e35d44a21df6aedb1dde3425d95884df446995820ccec65182c2d821d21727",
        ),
        "9875a2a003e2979c437da49d086018662cb9c5a23b47f3ec3804db119c81353d_ukv1-20639290a26e80f6_L2-4.bin": (
            128,
            "7b7739c3f580f0cb11bea8bce192185dcb86a008d63fc2102c92a0a5de37a094",
        ),
    },
}

GOLDEN_MOONCAKE = {
    "mha": {
        "golden-model_21c05bea24929e9c5e1d5a8fca136afc12069a0b22c64a2221cd6149f4399214_ukv1-542472633bab7949_L0-2_H2": (
            256,
            "537ecb657142dde007a965782c88a476472cc16647f91079ec642b4f35f1aedb",
        ),
        "golden-model_21c05bea24929e9c5e1d5a8fca136afc12069a0b22c64a2221cd6149f4399214_ukv1-542472633bab7949_L0-2_H3": (
            256,
            "5feef6b4fed0c250d7e76a86fabe99cb697dab859d97556d1e0228bdc38cdd58",
        ),
        "golden-model_21c05bea24929e9c5e1d5a8fca136afc12069a0b22c64a2221cd6149f4399214_ukv1-542472633bab7949_L2-4_H2": (
            256,
            "b7d2dfb101091e65ca507ba0392d3a74b66464169a2cd0cd17d3b02ade83052e",
        ),
        "golden-model_21c05bea24929e9c5e1d5a8fca136afc12069a0b22c64a2221cd6149f4399214_ukv1-542472633bab7949_L2-4_H3": (
            256,
            "350292c36e0cfbf24846acf9a33fc80f1001d84126dfc196d522c848c61338a3",
        ),
        "golden-model_9875a2a003e2979c437da49d086018662cb9c5a23b47f3ec3804db119c81353d_ukv1-542472633bab7949_L0-2_H2": (
            256,
            "49b3e8e02da97716d4d5f5da8e92e7ab9532186015ca874c84cf7b386e418a2f",
        ),
        "golden-model_9875a2a003e2979c437da49d086018662cb9c5a23b47f3ec3804db119c81353d_ukv1-542472633bab7949_L0-2_H3": (
            256,
            "436b43090dd46628913721678c411fb39a318e237d8c8172e20d4af9e7d1aa4f",
        ),
        "golden-model_9875a2a003e2979c437da49d086018662cb9c5a23b47f3ec3804db119c81353d_ukv1-542472633bab7949_L2-4_H2": (
            256,
            "0884424cc940d1ff3ba90a7e8110e4e2e40e45c3f2fae4dc6fd5ce65961e3f6f",
        ),
        "golden-model_9875a2a003e2979c437da49d086018662cb9c5a23b47f3ec3804db119c81353d_ukv1-542472633bab7949_L2-4_H3": (
            256,
            "96fd62d546401c123f08130292aea63dc454915f4b054fc346562b3de8518a64",
        ),
    },
    "mla": {
        "golden-model_21c05bea24929e9c5e1d5a8fca136afc12069a0b22c64a2221cd6149f4399214_ukv1-20639290a26e80f6_L0-2": (
            128,
            "78be6166ff3790a50cb1e3ea9357c3fe8fb36caaa29d058b1961976506622861",
        ),
        "golden-model_21c05bea24929e9c5e1d5a8fca136afc12069a0b22c64a2221cd6149f4399214_ukv1-20639290a26e80f6_L2-4": (
            128,
            "e1d2579c667f6a40ee5b385a55d7095ebd77b80799fe73cf67583a2517504788",
        ),
        "golden-model_9875a2a003e2979c437da49d086018662cb9c5a23b47f3ec3804db119c81353d_ukv1-20639290a26e80f6_L0-2": (
            128,
            "d7e35d44a21df6aedb1dde3425d95884df446995820ccec65182c2d821d21727",
        ),
        "golden-model_9875a2a003e2979c437da49d086018662cb9c5a23b47f3ec3804db119c81353d_ukv1-20639290a26e80f6_L2-4": (
            128,
            "7b7739c3f580f0cb11bea8bce192185dcb86a008d63fc2102c92a0a5de37a094",
        ),
    },
}


def _layout(plan, kwargs, geometry):
    page_size, head_dim, itemsize = geometry
    mla = kwargs["rank_replicated"]
    return PageUnifiedLayout(
        page_size=page_size,
        layer_num=kwargs["end_layer"] - kwargs["start_layer"],
        head_num=1 if mla else kwargs["local_kv_heads"],
        head_group_num=plan.head_group_num,
        head_dim=head_dim,
        itemsize=itemsize,
        is_mla=mla,
    )


def _plan_record(name):
    kwargs, geometry = PLAN_CASES[name]
    page_size = geometry[0]
    plan = plan_unified_kv(model_id=MODEL_ID, page_size=page_size, **kwargs)
    layout = _layout(plan, kwargs, geometry)
    return {
        "digest": namespace_digest(plan.namespace),
        "suffixes": list(plan.suffixes),
        "layer_ranges": [tuple(r) for r in plan.layer_ranges],
        "head_ranges": [tuple(r) for r in plan.head_ranges],
        "spans": [tuple(s) for s in plan.chunk_spans(layout)],
        "bytes_per_page": layout.bytes_per_page,
    }


# name -> (mla, grouped, attn tp rank, attn tp size, pp rank, pp size,
#          (cp rank, cp size), head partition, layer partition)
CONFIG_CASES = {
    "mha_tp2_rank1_pp2_stage1": (False, True, 1, 2, 1, 2, (0, 1), 2, 2),
    "mha_tp2_rank0_standalone": (False, False, 0, 2, 1, 2, (0, 1), 2, 2),
    "mha_tp2_rank1_cp_rank1": (False, True, 1, 2, 1, 2, (1, 2), 2, 2),
    "mha_tp1_no_partition": (False, True, 0, 1, 1, 2, (0, 1), None, None),
    "mla_tp2_rank0": (True, True, 0, 2, 1, 2, (0, 1), None, 2),
    "mla_tp2_rank1": (True, True, 1, 2, 1, 2, (0, 1), None, 2),
}


def _config_record(name):
    mla, grouped, tp_rank, tp_size, pp_rank, pp_size, cp, head_part, layer_part = (
        CONFIG_CASES[name]
    )
    local_heads = 8 // tp_size
    groups = 1 if mla or head_part is None else local_heads // head_part
    device_pool = object.__new__(MLATokenToKVPool) if mla else SimpleNamespace()
    # FP8 variants share a uint8 storage view; the logical dtype is identity.
    device_pool.dtype = torch.float8_e4m3fn
    device_pool.store_dtype = torch.uint8
    host_pool = SimpleNamespace(
        dtype=torch.uint8,
        layout="page_unified",
        storage_format_tag=None,
        head_num=0 if mla else local_heads,
        head_group_num=groups,
        start_layer=4,
        end_layer=7,
        layer_num=3,
        page_unified_layout=PageUnifiedLayout(
            page_size=16,
            layer_num=3,
            head_num=1 if mla else local_heads,
            head_group_num=groups,
            head_dim=128,
            itemsize=1,
            is_mla=mla,
        ),
    )
    controller = object.__new__(HiCacheController)
    controller.page_size = 16
    controller.storage_host_pool = host_pool
    controller.enable_storage_metrics = False
    if grouped:
        anchor = SimpleNamespace(host_pool=host_pool, device_pool=device_pool)
        controller.mem_pool_host = SimpleNamespace(
            entries=[anchor], anchor_entry=anchor, layout="page_unified"
        )
        controller.mem_pool_device = (
            device_pool if mla else SimpleNamespace(dtype=torch.float16)
        )
    else:
        controller.mem_pool_host = host_pool
        controller.mem_pool_device = device_pool
    memory = SimpleNamespace(
        hicache_storage_key_scheme="unified",
        hicache_storage_head_partition=head_part,
        hicache_storage_layer_partition=layer_part,
    )
    parallel = SimpleNamespace(
        tp_rank=tp_rank,
        tp_size=tp_size,
        attn_tp_rank=tp_rank,
        attn_tp_size=tp_size,
        attn_dcp_size=1,
        attn_dcp_rank=0,
        pp_rank=pp_rank,
        pp_size=pp_size,
    )
    module = "sglang.srt.managers.cache_controller"
    with (
        patch(f"{module}.get_memory", return_value=memory),
        patch(f"{module}.get_parallel", return_value=parallel),
        patch(f"{module}.is_dp_attention_enabled", return_value=False),
        patch.object(controller, "get_attn_cp_rank_and_size", return_value=cp),
    ):
        config = controller._generate_storage_config("golden/model")
    return {
        "suffixes": list(config.unified_suffixes),
        "spans": [tuple(s) for s in config.unified_chunk_spans],
        "is_storage_writer": config.is_storage_writer,
        "is_rank_replicated": config.is_rank_replicated,
        "model_name": config.model_name,
    }


# Stored objects: one MHA rank (TP2 rank 1, two head groups) and one MLA rank.
STORE_PAGE_SIZE = 4
STORE_LAYERS = 4
STORE_HEAD_DIM = 8
STORE_TOKENS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]


def _store_rank(mla):
    layout = PageUnifiedLayout(
        page_size=STORE_PAGE_SIZE,
        layer_num=STORE_LAYERS,
        head_num=1 if mla else 2,
        head_group_num=1 if mla else 2,
        head_dim=STORE_HEAD_DIM,
        itemsize=2,
        is_mla=mla,
    )
    plan = plan_unified_kv(
        model_id=MODEL_ID,
        dtype="bfloat16",
        page_size=STORE_PAGE_SIZE,
        rank_replicated=mla,
        local_kv_heads=0 if mla else 2,
        attn_tp_rank=0 if mla else 1,
        attn_tp_size=1 if mla else 2,
        attn_cp_size=1,
        start_layer=0,
        end_layer=STORE_LAYERS,
        is_final_stage=True,
        head_group=None if mla else 1,
        layer_partition=2,
    )
    return layout, plan


def _store_config(mla, plan, layout, extra_config):
    return HiCacheStorageConfig(
        tp_rank=0 if mla else 1,
        tp_size=1 if mla else 2,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=mla,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name=MODEL_ID,
        unified_suffixes=plan.suffixes,
        unified_chunk_spans=plan.chunk_spans(layout),
        extra_config=extra_config,
    )


def _store_pages(layout, count):
    """Deterministic page blocks: byte i of page p is a fixed function of (p, i)."""
    pages = torch.empty(layout.page_dims(count), dtype=torch.bfloat16)
    raw = pages.view(torch.uint8).reshape(count, -1)
    index = torch.arange(raw.shape[1], dtype=torch.int64)
    for page in range(count):
        raw[page] = ((index * 131 + page * 17 + 7) % 251).to(torch.uint8)
    return pages


def _page_keys():
    keys = get_storage_hash_str(STORE_TOKENS, page_size=STORE_PAGE_SIZE)
    return list(keys)


class _FileHostPool:
    def __init__(self, layout, pages):
        self.page_unified_layout = layout
        self.page_size = layout.page_size
        self.pages = pages

    def get_data_page(self, offset, flat=False):
        page = self.pages[offset // self.page_size]
        return page.flatten() if flat else page

    def get_dummy_flat_data_page(self):
        return torch.empty_like(self.pages[0]).flatten()

    def set_from_flat_data_page(self, offset, page):
        target = self.pages[offset // self.page_size]
        target.copy_(page.reshape_as(target))


def _file_record(mla):
    layout, plan = _store_rank(mla)
    keys = _page_keys()
    pages = _store_pages(layout, len(keys))
    extra_config = {
        "enable_metadata_cache": True,
        "metadata_ttl": -1,
        "max_size": 0,
        "min_free_space": 0,
    }
    with (
        tempfile.TemporaryDirectory() as directory,
        envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(directory),
    ):
        backend = HiCacheFile(_store_config(mla, plan, layout, extra_config))
        pool = _FileHostPool(layout, pages)
        backend.register_mem_pool_host(pool)
        backend.register_mem_host_pool_v2(pool, PoolName.KV)
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=keys,
            host_indices=torch.arange(len(keys) * layout.page_size),
        )
        results = backend.batch_set_v2([transfer])
        assert results == {PoolName.KV: [True] * len(keys)}, results
        record = {}
        for name in sorted(os.listdir(directory)):
            with open(os.path.join(directory, name), "rb") as f:
                data = f.read()
            record[name] = (len(data), hashlib.sha256(data).hexdigest())
        return record


class _MooncakeCluster:
    def __init__(self):
        self.objects = {}


def _mooncake_modules(cluster):
    class FakeMooncakeDistributedStore:
        def setup(self, *args, **kwargs):
            return 0

        def register_buffer(self, ptr, size):
            return 0

        def put(self, key, value, *args):
            cluster.objects[key] = bytes(value)
            return 0

        def is_exist(self, key):
            return 1 if key in cluster.objects else 0

        def get(self, key):
            return cluster.objects.get(key)

        def batch_is_exist(self, keys):
            return [1 if key in cluster.objects else 0 for key in keys]

        def batch_put_from(self, keys, ptrs, sizes, *config):
            for key, ptr, size in zip(keys, ptrs, sizes):
                cluster.objects.setdefault(key, ctypes.string_at(ptr, size))
            return [0] * len(keys)

    class ReplicateConfig:
        def __init__(self):
            self.group_ids = None

    mooncake = types.ModuleType("mooncake")
    store = types.ModuleType("mooncake.store")
    store.MooncakeDistributedStore = FakeMooncakeDistributedStore
    store.ReplicateConfig = ReplicateConfig
    return {"mooncake": mooncake, "mooncake.store": store}


class _MooncakeHostPool:
    _page_ids = HostKVCache._page_ids
    get_page_buffer_meta = HostKVCache._page_unified_buffer_meta

    def __init__(self, layout, pages):
        self.page_unified_layout = layout
        self.page_size = layout.page_size
        self.layout = "page_unified"
        self.kv_buffer = pages

    def get_ksize_per_token(self):
        layout = self.page_unified_layout
        return layout.bytes_per_page // layout.components // self.page_size


def _mooncake_record(mla):
    layout, plan = _store_rank(mla)
    keys = _page_keys()
    pages = _store_pages(layout, len(keys))
    cluster = _MooncakeCluster()
    extra_config = {
        "master_server_address": "127.0.0.1:50051",
        "check_server": False,
        "global_segment_size": 0,
    }
    with patch.dict("sys.modules", _mooncake_modules(cluster)):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        store = MooncakeStore(_store_config(mla, plan, layout, extra_config))
    store.register_mem_pool_host(_MooncakeHostPool(layout, pages))
    indices = torch.arange(len(keys) * layout.page_size)
    results = store.batch_set_v1(keys, indices)
    assert results == [True] * len(keys), results
    return {
        key: (len(data), hashlib.sha256(data).hexdigest())
        for key, data in sorted(cluster.objects.items())
        # Every store instance also puts one warmup object at construction.
        if "warmup" not in key
    }


class TestUnifiedKeySchemeGolden(CustomTestCase):
    maxDiff = None

    def test_plans(self):
        self.assertEqual(sorted(GOLDEN_PLANS), sorted(PLAN_CASES))
        for name in PLAN_CASES:
            with self.subTest(case=name):
                self.assertEqual(_plan_record(name), GOLDEN_PLANS[name])

    def test_fp8_variants_stay_in_distinct_keyspaces(self):
        self.assertNotEqual(
            GOLDEN_PLANS["mha_fp8_e4m3"]["digest"],
            GOLDEN_PLANS["mha_fp8_e5m2"]["digest"],
        )

    def test_controller_storage_configs(self):
        self.assertEqual(sorted(GOLDEN_CONFIGS), sorted(CONFIG_CASES))
        for name in CONFIG_CASES:
            with self.subTest(case=name):
                self.assertEqual(_config_record(name), GOLDEN_CONFIGS[name])

    def test_page_keys(self):
        self.assertEqual(_page_keys(), GOLDEN_PAGE_KEYS)

    def test_file_objects(self):
        for mla in (False, True):
            with self.subTest(mla=mla):
                self.assertEqual(
                    _file_record(mla), GOLDEN_FILES["mla" if mla else "mha"]
                )

    def test_mooncake_objects(self):
        for mla in (False, True):
            with self.subTest(mla=mla):
                self.assertEqual(
                    _mooncake_record(mla), GOLDEN_MOONCAKE["mla" if mla else "mha"]
                )


if __name__ == "__main__":
    unittest.main()
