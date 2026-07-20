import asyncio
import base64
import io
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from bench_parity import make_photo_like

from sglang.srt.managers.mm_utils import data_hash, hash_feature
from sglang.srt.multimodal.inkling import InklingProcessor
from sglang.srt.multimodal.processors import inkling as prc


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def wav_bytes(seconds=1.0, sr=16000):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    buf = io.BytesIO()
    sf.write(
        buf, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr, format="WAV"
    )
    return buf.getvalue()


def make_proc():
    proc = prc.InklingMultimodalProcessor.__new__(prc.InklingMultimodalProcessor)
    proc.IMAGE_TOKEN_ID = 100
    proc.AUDIO_TOKEN_ID = 101
    proc.AUDIO_END_TOKEN_ID = 102
    proc.inkling_processor = InklingProcessor()
    return proc


proc = make_proc()
img = png_bytes(make_photo_like(200, 320, seed=7))
aud = wav_bytes()

out = proc.assemble([1, 100, 2, 101, 3], [img], [aud])
img_item = next(i for i in out.mm_items if i.modality.name == "IMAGE")
aud_item = next(i for i in out.mm_items if i.modality.name == "AUDIO")
assert img_item.hash == data_hash(img), "image hash != data_hash(raw bytes)"
assert aud_item.hash == data_hash(aud), "audio hash != data_hash(raw bytes)"
print(f"  assemble: image hash={img_item.hash:#x} audio hash={aud_item.hash:#x} OK")

h0 = img_item.hash
img_item.set_pad_value()
assert img_item.hash == h0 and img_item.pad_value is not None
print(f"  set_pad_value: hash preserved, pad_value={img_item.pad_value} OK")

out2 = proc.assemble([1, 100, 2], [img], [])
assert out2.mm_items[0].hash == h0
print("  determinism: same bytes -> same hash OK")

data_url = "data:image/png;base64," + base64.b64encode(img).decode()
req = SimpleNamespace(input_ids=[1, 100, 100, 2])
out3 = asyncio.run(
    proc.process_mm_data_async(
        image_data=[data_url, data_url], audio_data=None, request_obj=req
    )
)
assert all(i.hash == h0 for i in out3.mm_items), "data: URL roundtrip hash mismatch"
print("  process_mm_data_async: concurrent resolve + hash OK")

orig = prc._resolve_media_item
prc._resolve_media_item = lambda it: (time.sleep(0.3), orig(it))[1]
t0 = time.perf_counter()
asyncio.run(prc._resolve_media_items([data_url] * 8))
elapsed = time.perf_counter() - t0
prc._resolve_media_item = orig
assert elapsed < 1.2, f"8x 0.3s resolves took {elapsed:.2f}s; expected ~0.3s"
print(f"  concurrency: 8 x 0.3s resolves in {elapsed:.2f}s OK")

imgs_5 = [png_bytes(make_photo_like(1080, 1920, seed=s)) for s in range(5)]
feats = [
    torch.randn(1323, 1, 40, 40, 3, dtype=torch.bfloat16).expand(1323, 2, 40, 40, 3)
    for _ in range(5)
]
t0 = time.perf_counter()
for b in imgs_5:
    data_hash(b)
t_bytes = (time.perf_counter() - t0) * 1e3
t0 = time.perf_counter()
for f in feats:
    hash_feature(f)
t_feat = (time.perf_counter() - t0) * 1e3
print(
    f"  hash cost 5 imgs: raw bytes {t_bytes:.1f}ms vs feature tensor {t_feat:.1f}ms "
    f"({t_feat / t_bytes:.0f}x)"
)

print("HASH_FETCH_OK")
