# A01_B20: Distribution metadata evidence — `diffusion` extra exists, but omits `accelerate` / `ftfy`

## Why this note exists

We needed a **non-speculative** proof of what `uv pip install "sglang[diffusion]" --prerelease=allow` actually means in the shipped package.

This note records the **authoritative** dependency list from the installed `sglang` distribution metadata (PEP 621 / wheel `METADATA`), showing:

- The installed distribution **does provide** a `diffusion` extra (`Provides-Extra: diffusion`).
- The `diffusion` extra’s `Requires-Dist` list **does not include** `accelerate` (and also does not include `ftfy`).
- Therefore, installing `sglang[diffusion]` is **not sufficient** to satisfy diffusers `device_map` runtime requirements for DiT / FLUX / Wan models.

---

## Evidence (terminal log @docker 172–182)

### Command

```bash
python -c "import importlib.metadata as m; d=m.distribution('sglang'); print([k for k in d.metadata.get_all('Provides-Extra',[]) or []]); print([x for x in (d.metadata.get_all('Requires-Dist',[]) or []) if 'extra == \"diffusion\"' in x])"
```

### Output

```text
['checkpoint-engine', 'diffusion', 'tracing', 'test', 'dev', 'all']
['PyYAML==6.0.1; extra == \"diffusion\"', 'cloudpickle; extra == \"diffusion\"', 'diffusers==0.36.0; extra == \"diffusion\"', 'imageio==2.36.0; extra == \"diffusion\"', 'imageio-ffmpeg==0.5.1; extra == \"diffusion\"', 'moviepy>=2.0.0; extra == \"diffusion\"', 'opencv-python-headless==4.10.0.84; extra == \"diffusion\"', 'remote-pdb; extra == \"diffusion\"', 'st_attn==0.0.7; (platform_machine != \"aarch64\" and platform_machine != \"arm64\") and extra == \"diffusion\"', 'vsa==0.0.4; (platform_machine != \"aarch64\" and platform_machine != \"arm64\") and extra == \"diffusion\"', 'runai_model_streamer; extra == \"diffusion\"', 'cache-dit==1.2.0; extra == \"diffusion\"']
```

---

## Interpretation

- `diffusion` **is** an officially declared extra in the shipped distribution.
- But the declared dependency set is missing at least:
  - `accelerate` (required by diffusers when using `device_map="cuda"/"auto"`)
  - `ftfy` (required by some prompt text-cleaning paths)

So the correct diagnosis is not “`diffusion` extra doesn’t exist”, but rather:

> The shipped `diffusion` extra exists, but its dependency list is incomplete for real-world diffusion/DiT usage.

This matches the runtime traceback we observed:

- `NotImplementedError: Using device_map requires the accelerate library. Please install it using: pip install accelerate.`

