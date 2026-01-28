# A01_B18: Draft English reply for Issue #17671

Below is a concise English reply you can post directly on Issue `#17671`, based on all tests in B07–B17.

---

Hi @kevin85421 and SGLang maintainers,

I spent some time debugging this on my local RTX 4090 with the latest dev image and wanted to share concrete findings from the diffusion side.

### 1. What I tested

- **Environment**
  - GPU: RTX 4090
  - Docker image: `lmsysorg/sglang:dev`
  - Image ID: `sha256:461c0f6164173501fda97ac037e06806748eea7fd01686d2cb75b73a6cc62fd8`
  - Created: 2026‑01‑27

- **Sanity checks for “does diffusion exist at all?”**
  - `import diffusers` → OK (`diffusers ok 0.36.0`)
  - `import sglang.multimodal_gen` → OK
  - `from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator` → OK

- **End‑to‑end runs with SD‑style models**
  - `runwayml/stable-diffusion-v1-5` via
    ```bash
    sglang generate --model-path runwayml/stable-diffusion-v1-5 \
      --backend diffusers --prompt "test" --save-output
    ```
    → server starts, diffusers backend selected, model download begins; failure is `No space left on device` (Docker disk quota), not a missing module.
  - `segmind/tiny-sd` → server starts, connects to HF Hub, downloads `model_index.json`, but fails because the HF repo snapshot itself is incomplete (missing required components).

So for classic SD models, the current dev image does have a working diffusion *code path* – the issues I hit there were disk space and specific HF model repos, not a total lack of diffusion code.

### 2. Where things really break for FLUX / Wan / Cache‑DiT

Once I moved to more advanced DiT‑style models (Wan 2.1, Flux‑like, Cache‑DiT), the picture changed:

1. **Missing `accelerate` in the image**
   - When loading DiT models, the logs show:
     ```
     Loading diffusers pipeline with dtype=torch.bfloat16, device_map=cuda
     NotImplementedError: Using `device_map` requires the `accelerate` library.
     ```
   - SGLang’s diffusion pipeline calls `DiffusionPipeline.from_pretrained(..., device_map="cuda"/"auto")`, which is correct for diffusers, but the dev image does **not** ship `accelerate`.
   - Result: the scheduler worker dies, and the diffusion server never comes up, even though the GPU and drivers are fine.

   - Additionally, I verified the installed `sglang` distribution metadata: it does provide a `diffusion` extra, but that extra does **not** declare `accelerate` (and also does not declare `ftfy`). So even `uv/pip install "sglang[diffusion]"` is currently insufficient for DiT/FLUX/Wan runtime requirements.

2. **Missing `ftfy` for text cleaning**
   - In more complete text‑normalization paths, prompt cleaning calls into code that expects `ftfy` to be installed.
   - Without `ftfy`, these paths crash before the prompt even reaches the model.

3. **Refactor leftovers in the loader utilities**
   - There is at least one function rename that wasn’t propagated everywhere:
     - `set_default_dtype` was renamed to `set_default_torch_dtype`,
     - but some downstream code (e.g. comfyui / qwen image pipelines) still imports `set_default_dtype`, which leads to `AttributeError` at load time.

In other words:

- The **LLM path** (Qwen/Llama etc.) in this image is production‑ready.
- The **Diffusion path for SD‑style models** is “barely usable but debuggable”.
- The **Diffusion path for DiT / FLUX / Wan** is effectively in a **half‑finished state**: the code assumes `diffusers + accelerate + ftfy`, but the image only provides part of that stack.

This matches the user experience in this issue: *“following the official doc to launch FLUX simply doesn’t work”*.

### 3. Why this is not a “Windows only” problem

I initially tested under Windows + Docker Desktop + WSL2, but these specific failures are **image‑wide**, not OS‑specific:

- The missing `accelerate` / `ftfy` are properties of the container filesystem.
  - Any host (Windows or Linux) pulling `lmsysorg/sglang:dev` with this ID will get the same Python environment.
- The refactor bug (`set_default_dtype` vs `set_default_torch_dtype`) is in the Python sources, so it will raise on Linux exactly the same way.

Windows vs. Linux only changes **how** OOM manifests (`-9` from WSL2 vs. Linux OOM‑killer), not the root cause that the image itself is missing required diffusion dependencies.

### 4. Suggested fixes (image + code)

I think this issue is pointing at a real gap in the dev image for diffusion workloads, especially DiT / FLUX / Wan. Concretely, I’d suggest:

1. **Update the dev image (or provide a diffusion‑ready variant)**  
   In the Dockerfile / build scripts, after installing `sglang`, add at minimum:
   ```dockerfile
   RUN uv pip install accelerate ftfy
   ```
   Or expose a dedicated tag like `sglang:dev-diffusion` / extras such as `sglang[diffusion]` that guarantee these libs are present.

2. **Clean up the loader refactor leftovers**
   - Ensure all callers use `set_default_torch_dtype` (or provide a small alias `set_default_dtype = set_default_torch_dtype` for backwards compatibility).

3. **Document resource and env requirements for FLUX / Wan / DiT models**
   - These models are significantly heavier than plain SD‑Turbo: on a 24 GB 4090 under WSL2, I frequently hit system OOM (exit code `-9`) even after fixing Python deps.
   - It would help a lot to have a short “DiT diffusion checklist” in the docs: required RAM/VRAM, recommended flags, and any known‑bad combinations (e.g. certain offload + Cache‑DiT combos).

### 5. Happy to help, but may not be able to fully fix it alone

I’ve captured more detailed logs and analysis in my own notes (including the exact tracebacks for the `accelerate` and `ftfy` issues, and the Cache‑DiT OOM behavior). I’m happy to:

- Share more traces if needed,
- Help validate a new image that includes `accelerate` / `ftfy`,
- Or review a small PR that wires these dependencies into the build.

I can try to open a PR myself to add the missing dependencies, but given how central the Docker image is to the project, it might be safer if someone from the core team takes the lead on wiring this into your official build pipeline.

Thanks again for the great work on SGLang – the LLM path is already very solid; this bug mostly shows that the diffusion/DiT side just needs its “infrastructure” finished so that users don’t have to manually `pip install accelerate ftfy` inside the container.

---

**Last updated:** 2026‑01‑28

