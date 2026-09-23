// ComfyUI support statement for a diffusion cookbook page.
//
// The per-model facts live here rather than in each page so they stay in sync
// with the plugin. `integrated` must match a key in `executor_class_dict`
// (python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/core/generator.py);
// a model absent from that dict has no integrated mode.

const PLUGIN_PATH = "python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion";
const PLUGIN_URL =
  "https://github.com/sgl-project/sglang/tree/main/" + PLUGIN_PATH;

const MODELS = {
  flux: {
    serverNode: "SGLDiffusion Generate Image",
    integratedKey: "flux",
    workflow: "flux_sgld_sp.json",
  },
  "z-image": {
    serverNode: "SGLDiffusion Generate Image",
    integratedKey: "lumina2",
    workflow: "z-image_sgld.json",
  },
  "qwen-image": {
    serverNode: "SGLDiffusion Generate Image",
    integratedKey: "qwen_image",
    workflow: "qwen_image_sgld.json",
  },
  "qwen-image-edit": {
    serverNode: "SGLDiffusion Generate Image",
    integratedKey: "qwen_image_edit",
    note: "Image editing through the integrated path is experimental.",
  },
  "minimax-h3": {
    serverNode: "SGLDiffusion Generate H3",
    verified: true,
    integratedBlockedBecause:
      "H3 denoises a packed video-and-audio sequence in one pass and routes " +
      "conditioning by task, while ComfyUI's KSampler drives a single latent " +
      "tensor and has no audio branch",
  },
  image: { serverNode: "SGLDiffusion Generate Image" },
  video: { serverNode: "SGLDiffusion Generate Video" },
};

export const ComfyUISupport = ({ model = "video", note }) => {
  const spec = MODELS[model] || MODELS.video;
  const extraNote = note || spec.note;

  return (
    <div>
      <p>
        Run this model from ComfyUI with the{" "}
        <a href={PLUGIN_URL}>SGLDiffusion plugin</a>, which ships in the SGLang
        repository at <code>{PLUGIN_PATH}</code>.
      </p>

      <p>
        <strong>Server mode</strong> — SGLang runs the pipeline and ComfyUI
        sends the request. Start a server as shown above, point the{" "}
        <code>SGLDiffusion Server Model</code> node at it, then generate with{" "}
        <code>{spec.serverNode}</code>.
        {spec.verified
          ? " This path has been run end to end against a live server."
          : ""}
      </p>

      {spec.integratedKey ? (
        <p>
          <strong>Integrated mode</strong> — ComfyUI's own sampler, CLIP, and
          VAE drive the loop while SGLang replaces the model forward. Load the
          checkpoint with <code>SGLDiffusion UNET Loader</code> and set{" "}
          <code>model_type</code> to <code>{spec.integratedKey}</code> on the{" "}
          <code>SGLDiffusion Options</code> node.
          {spec.workflow ? (
            <>
              {" "}
              A reference workflow is included at{" "}
              <code>{`${PLUGIN_PATH}/workflows/${spec.workflow}`}</code>.
            </>
          ) : null}
        </p>
      ) : (
        <p>
          <strong>Integrated mode</strong> — not available for this model
          {spec.integratedBlockedBecause
            ? `: ${spec.integratedBlockedBecause}`
            : ", which has no executor in the plugin"}
          . Use server mode.
        </p>
      )}

      {extraNote ? <p>{extraNote}</p> : null}
    </div>
  );
};
