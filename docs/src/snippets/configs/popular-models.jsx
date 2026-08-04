// Single `export const popularModels` literal — no spreads/calls/IIFE (Mintlify
// re-evals at hydration).
//
// Rotated by <PopularModels> (/src/snippets/_popular_models.jsx) on the docs home
// (`variant="hero"`, uses each entry's `hero` block) and the Cookbook home
// (compact strip, uses `name` / `badge` / `tags`). Both walk this list in order,
// so an entry added here becomes a slide on both; an entry with no `hero` block
// still rotates on the home page, just without a blurb.
//
// Keep the copy to claims that hold on the linked page: platform counts from that
// model's config `supportedHardware`, precisions from its `quantizations`, blurbs
// paraphrasing that page's own opening.

export const popularModels = [
  {
    name: "Kimi-K3",
    vendor: "Moonshot AI",
    href: "/cookbook/autoregressive/Moonshotai/Kimi-K3",
    logo: "/cards/logos/moonshotai.png",
    badge: "New",
    tags: ["8 platforms", "PD disagg", "DSPARK"],
    hero: {
      eyebrow: "Featured model · New",
      headline: "Meet Kimi-K3 on SGLang",
      blurb:
        "SGLang natively implements and deeply optimizes K3's new architecture with fused KDA decode kernels, DP attention, MTP, PD disaggregation, and KDA-aware prefix caching. Kimi-K3 is supported on both NVIDIA and AMD GPUs.",
      tags: ["2.8T parameters", "Fused KDA decode", "NVIDIA + AMD"],
      cta: "Open the Kimi-K3 cookbook",
      caption: "Kimi-K3 deployment guide",
    },
  },
  {
    name: "Inkling",
    vendor: "Thinking Machines",
    href: "/cookbook/autoregressive/ThinkingMachines/Inkling",
    logo: "/cards/logos/thinkingmachines.png",
    badge: "New",
    tags: ["7 platforms", "NVFP4 / BF16", "MTP + DSpark"],
    hero: {
      eyebrow: "Featured model · New",
      headline: "Meet Inkling on SGLang",
      blurb:
        "Thinking Machines' open-weights Mixture-of-Experts model — 975B parameters, 41B active per token, a 1M-token context window, and native text, image, and audio input. The cookbook covers its MTP speculative-decoding path and long-context prefix caching on NVIDIA and AMD.",
      tags: ["975B · 41B active", "1M context", "Text + image + audio"],
      cta: "Open the Inkling cookbook",
      caption: "Inkling deployment guide",
    },
  },
  {
    name: "GLM-5.2",
    vendor: "Z.ai",
    href: "/cookbook/autoregressive/GLM/GLM-5.2",
    logo: "/cards/logos/glm.png",
    badge: "New",
    tags: ["7 platforms", "DSA attention", "FP8 / NVFP4"],
    hero: {
      eyebrow: "Featured model · New",
      headline: "Meet GLM-5.2 on SGLang",
      blurb:
        "Z.ai's DeepSeek-Sparse-Attention Mixture-of-Experts model, with MTP speculative decoding and a 1M-token context window. Recipes cover FP8, BF16, and NVFP4 across H200, B200, B300, GB300, and AMD MI300X / MI325X / MI355X.",
      tags: ["DSA attention", "1M context", "FP8 / BF16 / NVFP4"],
      cta: "Open the GLM-5.2 cookbook",
      caption: "GLM-5.2 deployment guide",
    },
  },
];
