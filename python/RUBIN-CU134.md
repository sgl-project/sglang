# Rubin CUDA 13.4 source environment

Use the pinned ARM64 image
`lmsysorg/sglang@sha256:b1b9981b6feab6e0c569d6c05693fd8757c0a0389c263eb30e59b2752eea3941`.
It provides the versions recorded in `requirements-rubin-cu134.txt`. The normal
`pyproject.toml` pins remain unchanged for the GB300/cu130 path; do not run a
dependency resolver against that file in the Rubin image. The source tree is
mounted through `PYTHONPATH`, and Rust extensions must be built with the
image's active Python/Torch environment (`python setup.py build_rust`).

The 0a79 rebase branch and the Rubin image are separate build inputs. Rebuild
Rust extensions and all architecture-specific AOT artifacts against the
Rubin image; record the image digest, extension/cubin SHA256, SASS architecture,
register count, and shared-memory use. No SM103a artifact is a Rubin fallback.

For Q38 AOT bundles, legacy flat manifests are accepted only on SM103a. New
multi-architecture bundle roots may contain this dispatcher shape:

```json
{
  "architectures": {
    "sm103a": "sm103a",
    "sm107a": "sm107a"
  }
}
```

Each referenced subdirectory contains the existing per-bundle `manifest.json`
and its artifacts; that manifest should declare the matching `arch` value.
Alternatively, an environment variable can point directly at a target bundle
whose manifest declares `"arch": "sm107a"`.

No Rubin GPU was available for this work. **Not tested on Rubin hardware.**
