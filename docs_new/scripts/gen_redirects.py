#!/usr/bin/env python3
"""Generate Mintlify docs.json redirects from old Sphinx paths to new Mintlify paths."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OLD_DOCS = REPO / "docs"
NEW_DOCS = REPO / "docs_new" / "docs"

# Directory-level renames (old → new, under /docs/ prefix)
SECTION_RENAMES = {
    "get_started": "get-started",
    "platforms": "hardware-platforms",
    "supported_models": "supported-models",
    "diffusion": "sglang-diffusion",
}

# Explicit file-level mappings. Keys are old URL paths (no .html, with leading /).
# Values are new URL paths (with /docs/ prefix, no extension).
EXPLICIT = {
    # get_started → get-started
    "/get_started/install": "/docs/get-started/installation",
    # developer_guide rename
    "/developer_guide/development_jit_kernel_guide": "/docs/developer_guide/JIT_kernels",
    # platforms → hardware-platforms (with file renames)
    "/platforms/amd_gpu": "/docs/hardware-platforms/amd-gpus",
    "/platforms/cpu_server": "/docs/hardware-platforms/cpu-server",
    "/platforms/tpu": "/docs/hardware-platforms/tpu",
    "/platforms/xpu": "/docs/hardware-platforms/xpu",
    # platforms/ascend → hardware-platforms/ascend-npus (flattened, renamed)
    "/platforms/ascend/ascend_npu": "/docs/hardware-platforms/ascend-npus/SGLang-installation-with-NPUs-support",
    "/platforms/ascend/ascend_npu_best_practice": "/docs/hardware-platforms/ascend-npus/Best-Practice-on-Ascend-NPU",
    "/platforms/ascend/ascend_npu_deepseek_example": "/docs/hardware-platforms/ascend-npus/DeepSeek-Examples",
    "/platforms/ascend/ascend_npu_glm5_examples": "/docs/hardware-platforms/ascend-npus/GLM-5",
    "/platforms/ascend/ascend_npu_qwen3_examples": "/docs/hardware-platforms/ascend-npus/Qwen3-Examples",
    "/platforms/ascend/ascend_npu_qwen3_5_examples": "/docs/hardware-platforms/ascend-npus/Qwen3.5",
    "/platforms/ascend/ascend_npu_support_features": "/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU",
    "/platforms/ascend/ascend_npu_support_models": "/docs/hardware-platforms/ascend-npus/Support-Models-on-Ascend-NPU",
    # Old pages dropped — redirect to section overview
    "/platforms/ascend/ascend_contribution_guide": "/docs/hardware-platforms/overview",
    "/platforms/ascend/ascend_npu_environment_variables": "/docs/hardware-platforms/overview",
    "/platforms/ascend/ascend_npu_quantization": "/docs/hardware-platforms/overview",
    "/platforms/ascend/ascend_npu_support": "/docs/hardware-platforms/overview",
    "/platforms/ascend/mindspore_backend": "/docs/hardware-platforms/overview",
    "/platforms/ascend_npu_ring_sp_performance": "/docs/hardware-platforms/overview",
    "/platforms/apple_metal": "/docs/hardware-platforms/overview",
    "/platforms/mthreads_gpu": "/docs/hardware-platforms/overview",
    "/platforms/nvidia_jetson": "/docs/hardware-platforms/overview",
    "/platforms/plugin": "/docs/hardware-platforms/overview",
    # supported_models → supported-models (flattened, renamed)
    "/supported_models": "/docs/supported-models",
    "/supported_models/index": "/docs/supported-models",
    "/supported_models/extending/mindspore_models": "/docs/supported-models/mindspore-models",
    "/supported_models/extending/modelscope": "/docs/supported-models/modelscope",
    "/supported_models/extending/support_new_models": "/docs/supported-models/new-model-support",
    "/supported_models/extending/transformers_fallback": "/docs/supported-models/transformers-fallback",
    "/supported_models/extending/index": "/docs/supported-models",
    "/supported_models/retrieval_ranking/classify_models": "/docs/supported-models/classification-models",
    "/supported_models/retrieval_ranking/embedding_models": "/docs/supported-models/embedding-models",
    "/supported_models/retrieval_ranking/rerank_models": "/docs/supported-models/rerank-models",
    "/supported_models/retrieval_ranking/index": "/docs/supported-models",
    "/supported_models/specialized/reward_models": "/docs/supported-models/reward-models",
    "/supported_models/specialized/index": "/docs/supported-models",
    "/supported_models/text_generation/generative_models": "/docs/supported-models/large-language-models",
    "/supported_models/text_generation/multimodal_language_models": "/docs/supported-models/vision-language-models",
    "/supported_models/text_generation/diffusion_language_models": "/docs/supported-models/diffusion-language-models",
    "/supported_models/text_generation/index": "/docs/supported-models",
    # diffusion → sglang-diffusion (file renames snake_case → kebab-case)
    "/diffusion": "/docs/sglang-diffusion/installation",
    "/diffusion/index": "/docs/sglang-diffusion/installation",
    "/diffusion/installation": "/docs/sglang-diffusion/installation",
    "/diffusion/environment_variables": "/docs/sglang-diffusion/environment-variables",
    "/diffusion/ci_perf": "/docs/sglang-diffusion/ci-performance",
    "/diffusion/api/cli": "/docs/sglang-diffusion/api/cli",
    "/diffusion/api/openai_api": "/docs/sglang-diffusion/api/openai-api",
    "/diffusion/performance/attention_backends": "/docs/sglang-diffusion/attention-backends",
    "/diffusion/performance/cache/cache_dit": "/docs/sglang-diffusion/cache-dit",
    "/diffusion/performance/cache/index": "/docs/sglang-diffusion/caching-acceleration",
    "/diffusion/performance/cache/teacache": "/docs/sglang-diffusion/tea-cache",
    "/diffusion/performance/index": "/docs/sglang-diffusion/performance-optimization",
    "/diffusion/performance/profiling": "/docs/sglang-diffusion/profiling",
    # Diffusion pages dropped
    "/diffusion/api/post_processing": "/docs/sglang-diffusion/installation",
    "/diffusion/compatibility_matrix": "/docs/sglang-diffusion/installation",
    "/diffusion/contributing": "/docs/sglang-diffusion/installation",
    "/diffusion/development": "/docs/sglang-diffusion/installation",
    "/diffusion/disaggregation": "/docs/sglang-diffusion/installation",
    "/diffusion/performance/ring_sp_performance": "/docs/sglang-diffusion/performance-optimization",
    "/diffusion/quantization": "/docs/sglang-diffusion/installation",
    "/diffusion/reference": "/docs/sglang-diffusion/installation",
    "/diffusion/support_new_models": "/docs/sglang-diffusion/installation",
    "/diffusion/usage": "/docs/sglang-diffusion/installation",
    # basic_usage dropped pages
    "/basic_usage/deepseek_ocr": "/docs/basic_usage/overview",
    "/basic_usage/qwen3_5": "/docs/basic_usage/qwen3",
    # advanced_features dropped pages
    "/advanced_features/adaptive_speculative_decoding": "/docs/advanced_features/speculative_decoding",
    "/advanced_features/hisparse_guide": "/docs/advanced_features/overview",
    # references dropped
    "/references/learn_more": "/",
    "/references/release_lookup": "/docs/references/overview",
    # Root index
    "/index": "/",
    "/": "/",
}


def old_url_from_path(rel: Path) -> str | None:
    """Convert old docs/<rel> to its Sphinx URL path (no .html, leading /)."""
    parts = list(rel.parts)
    stem = rel.stem
    # Skip README, release_lookup/README, top-level non-doc files
    if stem == "README":
        return None
    # Drop the extension → URL path
    new_parts = parts[:-1] + [stem]
    return "/" + "/".join(new_parts)


def new_url_for(old_url: str, new_files_set: set[str]) -> str | None:
    """Compute new URL from old URL using section rename + explicit overrides."""
    if old_url in EXPLICIT:
        return EXPLICIT[old_url]
    # Default rule: `/section/path` → `/docs/section/path`, applying section renames
    parts = old_url.strip("/").split("/")
    if not parts or not parts[0]:
        return None
    section = parts[0]
    section = SECTION_RENAMES.get(section, section)
    new_url = "/docs/" + "/".join([section] + parts[1:])
    # Verify destination exists in new file tree
    if new_url in new_files_set:
        return new_url
    return None  # unmapped


def list_new_urls() -> set[str]:
    urls = set()
    for p in NEW_DOCS.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in (".mdx", ".ipynb", ".md"):
            continue
        rel = p.relative_to(NEW_DOCS)
        # Mintlify routes .mdx / .ipynb as `/docs/<path-without-ext>`
        url = "/docs/" + str(rel.with_suffix("")).replace(os.sep, "/")
        urls.add(url)
    return urls


def main():
    new_urls = list_new_urls()
    redirects: list[dict] = []
    seen_sources: set[str] = set()
    unmapped: list[str] = []

    # Iterate all old files
    old_files = []
    for p in sorted(OLD_DOCS.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix not in (".md", ".rst", ".ipynb"):
            continue
        rel = p.relative_to(OLD_DOCS)
        # Skip non-doc dirs
        if rel.parts and rel.parts[0] in (
            "_static",
            "performance_dashboard",
            "release_lookup",
        ):
            continue
        old_files.append(rel)

    for rel in old_files:
        old_url = old_url_from_path(rel)
        if old_url is None:
            continue
        # Old Sphinx URLs end in .html
        source = old_url + ".html"
        if source in seen_sources:
            continue
        new_url = new_url_for(old_url, new_urls)
        if new_url is None:
            unmapped.append(source)
            continue
        redirects.append({"source": source, "destination": new_url})
        seen_sources.add(source)

    # Also add explicit entries whose source key wasn't derived from a file (e.g. index variants)
    for old_key, new_val in EXPLICIT.items():
        source = old_key + ".html"
        if source in seen_sources:
            continue
        # Only add if old_key corresponds to an actual old page pattern we care about
        # Skip bare "/" and "/index" (handled by Mintlify default)
        if old_key in ("/", "/index"):
            continue
        redirects.append({"source": source, "destination": new_val})
        seen_sources.add(source)

    # Output
    print(f"# Total redirects: {len(redirects)}")
    print(f"# Unmapped old URLs: {len(unmapped)}")
    if unmapped:
        print("# --- UNMAPPED ---")
        for u in unmapped:
            print(f"#   {u}")
    print(json.dumps(redirects, indent=2))


if __name__ == "__main__":
    main()
