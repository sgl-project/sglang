# Media processing stages

This document describes the stages that turn one image in a request into model
input, and how the stages depend on each other. It describes the model-independent
structure only. The rules of each model are in the code of that model.

## Stages

| Stage | Input | Output | Needs pixels |
|---|---|---|---|
| Fetch | URL, file path, `data:` URL, or base64 | Encoded bytes | No |
| Probe | Encoded bytes | Height, width, orientation | No |
| Decode | Encoded bytes | `uint8` pixels | Produces them |
| Color / orientation | Pixels and file metadata | RGB pixels in the layout the model expects | Yes |
| Plan | Probe result, model parameters, request options | Target size, patch grid, expansion | No |
| Transform | Pixels and the target size from the plan | Resized, padded, or tiled pixels | Yes |
| Rescale + normalize | Transformed pixels, mean, std | Float pixels | Yes |
| Patchify | Float pixels, patch size | Tensor for the vision tower | Yes |
| Expand placeholders | Prompt token ids and the expansion from the plan | Expanded token ids | No |
| Digest | Encoded bytes | `blake3:<hex>` | No |
| Identity | Digest, processor fingerprint, effective options, plan | Cache identity | No |

Terms:

- **Resize** changes the resolution. **Rescale** multiplies by 1/255. They are
  different stages.
- The **expansion** of an image is the list of token ids that replaces the one
  placeholder token of that image in the prompt.
- The **grid** is `(t, h, w)` in patches.

## Dependencies

```
                     +- Decode - Color/orientation -+
                     |                              v
source - Fetch - bytes                          Transform - Rescale+normalize - Patchify -> vision tower
                     |                              ^
                     +- Probe ------ Plan ----------+
                     |                 |
                     |                 +- Expand placeholders -> token ids
                     |                 |
                     +- Digest --------+- Identity
```

There are two chains. Both start at Fetch.

- The pixel chain is Decode, Color / orientation, Transform, Rescale + normalize,
  Patchify.
- The token chain is Probe, Plan, Expand placeholders, plus Digest and Identity.

The token chain does not read pixels. It can run where there is no GPU and no
decoder, for example in a router.

The two chains meet at one point: Transform takes its target size from Plan. If
the two disagree, the number of embeddings from the vision tower does not match
the number of placeholder tokens in the prompt.

Plan does not read pixels, but it must use the height and width that the pixel
chain produces. If Color / orientation applies the EXIF orientation, Plan must
apply it too.

## Rules

- **Plan is defined by code.** Two callers must get the same token count for the
  same image, so each rounding step and each floating-point operation is part of
  the contract. Each model has its own plan function. A model reuses the plan
  function of another model only if the algorithm is the same in every detail.
- **The order of the pixel stages is a property of the model.** Most models use
  the order in the table. Some models normalize before they resize, and some
  composite the alpha channel after they resize.
- **Request options are explicit arguments.** A plan function takes the model
  parameters and the options of the request as separate arguments. It does not
  read environment variables.
- **Text in an expansion goes through a callback.** If a model inserts text
  around its image tokens, the plan function receives an `encode` callback. This
  crate does not depend on a tokenizer.
- **Errors have a class.** `MmError::InvalidRequest` means that the request is at
  fault. `MmError::Internal` means a fault in our own code or parameters.

## Where the stages are in this crate

| Stage | Location | State |
|---|---|---|
| Fetch | `common/fetch.rs` | Present |
| Probe | | Not in this crate yet |
| Decode | `common::decode_rgb` | Present |
| Color / orientation | `common::decode_rgb` | Conversion to RGB only |
| Plan | `plan/<model>.rs` | Present for the models in `plan/` |
| Transform | `common/resize.rs` | Resize kernels are present |
| Rescale + normalize, Patchify | `inkling/`, `dsv41/`, `qwen_vl/` | Present as CPU kernels that also resize |
| Expand placeholders | `common/expansion.rs` | Present |
| Digest | `common::content_digest` | Present |
| Identity | | Not in this crate yet |
