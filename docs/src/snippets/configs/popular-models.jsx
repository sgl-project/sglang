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
    name: "Qwen3.8-27B",
    vendor: "Qwen",
    href: "/cookbook/autoregressive/Qwen/Qwen3.8-27B",
    logo: "/cards/logos/qwen.png",
    badge: "New",
    tags: ["4 platforms", "Hybrid GDN", "BF16 / FP8 / NVFP4"],
    hero: {
      eyebrow: "Featured model \u00b7 New",
      headline: "Meet Qwen3.8-27B on SGLang",
      blurb:
        "A dense hybrid Gated Delta Networks model \u2014 48 GDN linear-attention layers interleaved with 16 full-attention, an in-checkpoint MTP head, and a native 262,144-token context. The cookbook covers single-GPU serving on H200 and RTX PRO 6000 / 5090.",
      tags: ["Dense 27B", "262K context", "Single-GPU"],
      cta: "Open the Qwen3.8-27B cookbook",
      caption: "Qwen3.8-27B deployment guide",
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
];
