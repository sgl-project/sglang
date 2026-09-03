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
    name: "Qwen3.8-Flash-Next",
    vendor: "Qwen",
    href: "/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next",
    logo: "/cards/logos/qwen.png",
    badge: "New",
    tags: ["6 platforms", "GDN + QSA hybrid", "BF16 / FP8 / NVFP4"],
    hero: {
      eyebrow: "Featured model \u00b7 New",
      headline: "Meet Qwen3.8-Flash-Next on SGLang",
      blurb:
        "Qwen's early preview of the Qwen4 architecture \u2014 176B total parameters with 6B active, three of every four layers Gated DeltaNet and the fourth global attention running Qwen Sparse Attention, over an ultra-sparse MoE with an in-checkpoint MTP head. The cookbook covers single-node TP4 serving on H200 / B200 / B300 / GB300 and TP8 on MI350X / MI355X.",
      tags: ["176B / 6B active", "262K context", "Single-node"],
      cta: "Open the Qwen3.8-Flash-Next cookbook",
      caption: "Qwen3.8-Flash-Next deployment guide",
    },
  },
  {
    name: "GLM-5.3-Flash",
    vendor: "Z.ai",
    href: "/cookbook/autoregressive/GLM/GLM-5.3-Flash",
    logo: "/cards/logos/glm.png",
    badge: "New",
    tags: ["9 platforms", "MLA + DSA + KDA hybrid", "Multimodal"],
    hero: {
      eyebrow: "Featured model · New",
      headline: "Meet GLM-5.3-Flash on SGLang",
      blurb:
        "Z.ai's natively multimodal Mixture-of-Experts model — 320B total parameters with 18B active, 45 text layers combining MLA, DSA sparse, and KDA linear attention, a 24-layer vision encoder for image and video input, and a native MTP draft layer for speculative decoding. Recipes cover H100 / H200 / B200 / B300 / GB200 / GB300 and AMD MI300X / MI325X / MI355X.",
      tags: ["320B / 18B active", "1M context", "Text + image + video"],
      cta: "Open the GLM-5.3-Flash cookbook",
      caption: "GLM-5.3-Flash deployment guide",
    },
  },
  {
    name: "MiniMax-H3",
    vendor: "MiniMax",
    href: "/cookbook/diffusion/MiniMax/MiniMax-H3",
    logo: "/cards/logos/minimax.png",
    badge: "New",
    tags: ["7 platforms", "Video + audio", "BF16 / FP8"],
    hero: {
      eyebrow: "Featured model · New",
      headline: "Meet MiniMax-H3 on SGLang",
      blurb:
        "MiniMax's video-and-audio diffusion model — one request returns an MP4 carrying 24 fps video and a synchronized stereo audio track. SGLang Diffusion serves all three task profiles — text, first/last frame, and image / video / audio reference conditioning — with Ulysses × Ring sequence parallelism and recipes across B200, B300, H200, H100, AMD MI300X / MI355X, and 2× RTX 5090.",
      tags: ["Video + synced audio", "4–15 s at 24 fps", "8× B200 → 2× RTX 5090"],
      cta: "Open the MiniMax-H3 cookbook",
      caption: "MiniMax-H3 deployment guide",
    },
  },
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
];
