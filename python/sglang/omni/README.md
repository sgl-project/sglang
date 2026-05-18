# SGLang Omni

`sglang.omni` is the orchestration layer for interleaved multimodal generation
models.

SGLang now supports models that mix:

- multimodal understanding, for example text-image input
- autoregressive text generation
- multimodal content generation, for example image output
- interleaved conversations where the model can generate text, then media, then
  continue the conversation

The first supported model family is SenseNova U1. This class of model is similar
to transfusion / MoT-style systems: an AR model reads the conversation, emits
text tokens, and uses special media boundary tokens such as an image marker to
handoff generation to a multimodal generation engine.

## How It Works

`sglang.omni` is intentionally thin. It does not replace SRT or `multimodal_gen`.

- `sglang.srt` owns AR text generation, tokenizer state, sessions, and KV cache.
- `sglang.multimodal_gen` owns image / audio / video generation pipelines.
- `sglang.omni` coordinates the handoff between them.

The core loop is:

1. ask the AR backend to prefill and decode until the next modality boundary
2. return normal text boundaries directly to the user
3. call a multimodal generation backend when the boundary is image / audio /
   video
4. commit the generated media back into the AR session when the model requires
   multi-turn continuation
5. continue decoding until EOS or the request limit is reached

## Directory Layout

- `core/`: model-agnostic request / response structs and the coordinator loop
- `runtime/`: integration with SRT scheduler state and request transport
- `backends/`: AR and multimodal generation execution backends
- `model_adapters/`: model-specific token grammar, prompts, and commit rules
- `configs/`: model wiring and registry
- `entrypoints/`: HTTP and streaming adapters
- `webui/`: lightweight debugging UI for omni models

## SRT Session Boundary

For colocated AR-backed models, the SRT path is split into three parts:

- `SRTBackedOmniSessionAdapter`: model-specific subclasses translate omni request options into SRT session operations.
- `OmniSessionRuntime`: owns the generic SRT session. It creates SRT requests, tracks session handles, updates context counts, and calls the scheduler executor.
- `OmniSessionModelHooks`: defines model-specific hooks called by the runtime: prepare user-turn inputs, prepare appended media inputs, account prefill, decode the next text/media boundary, decode VLM text, account generated-media commits, and clean model-local session state.

## Adding A New Interleaved Model

Use the existing SenseNova U1 path as the reference.

1. Add the model's AR support in SRT.
   The SRT side should be able to prefill messages, decode until the model's
   media boundary token, and continue from an existing session.

2. Add the model's media generation path in `multimodal_gen`.
   The generation path should return a `GeneratedSegment` with the visible media
   output and, when needed, a model-native commit payload.

3. Add a model adapter under `model_adapters/<model>/`.
   This is where model-specific behavior belongs: prompt formatting, boundary
   token rules, thinking text handling, condition paths, and media commit rules.
   Put request-level mode/option logic in the session adapter, and put token
   grammar / prepared-input / decode-boundary logic in `OmniSessionModelHooks`.

4. Add model wiring under `configs/<model>.py`.
   Build an `OmniCoordinator` with an AR backend and a multimodal generation
   backend. Register the model key in `configs/registry.py`.

5. Choose a multimodal generation backend.
   Use `LazyDirectPipelineForwardBackend` for a colocated, direct pipeline call.
   Use `PipelineExecutorBackend` when the multimodal generation runtime should own stage scheduling and executor behavior.
   Only same-process backends that borrow live SRT model / KV state should
   depend on `ColocatedContextOps`; standalone backends should stay on the
   generic `ContextOps` surface.

6. Expose the model through `sglang serve --model-type omni`.
   The normal HTTP / streaming entrypoints should call into the generic omni coordinator instead of adding model-specific HTTP paths.

Keep model-specific logic out of `core/` and `runtime/`.
