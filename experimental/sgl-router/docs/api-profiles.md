# API profiles

An **API profile** is the client-facing contract the router enforces for the
model it serves:

- which parameter values are accepted, and the defaults when a request
  omits them;
- how large an output may be;
- request size limits;
- model-name aliases;
- the error `type` clients see.

Routing, discovery and timeouts are not part of a profile.

One profile covers every protocol. `/v1/responses` and `/v1/messages` are
converted to chat requests first, and the profile is applied to that chat
request. A rule on `max_tokens` therefore also governs Messages' `max_tokens`
and Responses' `max_output_tokens`. Rejections happen before admission, so a
rejected request costs no queue slot and no engine call.

## Choosing a profile

The first source that is set wins:

| # | Source | Example |
|---|---|---|
| 1 | `--api-profile-file <path>` | a mounted ConfigMap |
| 2 | `--api-profile <name>` | `--api-profile stepfun-step5` |
| 3 | `SGLANG_ROUTER_API_PROFILE`: a preset name, or a path (contains `/` or ends in `.yaml`) | `SGLANG_ROUTER_API_PROFILE=moonshot-kimi` |
| 4 | `/etc/sgl-router/profile.yaml`, if it exists | baked into a vendor-specific image |
| 5 | built-in `openai-compatible` | — |

`openai-compatible` has no rules, so a router started without a profile
behaves exactly as it did before profiles existed. Keep the generic image on
it, and select vendor rules per deployment.

### Built-in presets

Presets are compiled into the binary from `profiles/*.yaml`. The image also
ships readable copies under `/usr/share/sgl-router/profiles/`.

| Preset | Contract |
|---|---|
| `openai-compatible` | Nothing enforced, nothing injected |
| `moonshot-kimi` | Moonshot-style immutable sampling: `top_p`, `top_k`, penalties and `n` pinned; `temperature` in [0, 1] |
| `stepfun-step5` | Ranges with defaults; `top_k: 0` means no limit; `max_tokens` clamped to 64000; 128 MiB body; at most 60 images; `reasoning_format`; `request_params_invalid` error type |

### Checking what is in effect

```bash
sgl-router --model-id m --worker-urls http://x:1 --api-profile-file profile.yaml --print-profile
```

`--print-profile` prints the merged profile as YAML and exits. It still
needs the router's required flags. At startup the router logs
`api profile loaded` with the profile name and where it came from.

## File format

The file is YAML. Every key is optional. An unknown key is a startup error,
so a typo cannot silently do nothing.

```yaml
name: my-vendor              # shown in logs
extends: stepfun-step5       # optional: a preset name or a file path (relative to this file)

models:
  aliases: [step-5-preview]  # accepted besides --model-id, rewritten to it before routing

limits:
  max_body_bytes: 128MiB     # B, KiB, MiB, GiB or a plain integer; 413 above it (default 100MiB)
  max_images: 60             # image_url parts per request; 400 above it

output:
  max_tokens:
    cap: 64000
    on_exceed: clamp         # reject (default): 400 above cap | clamp: lower it to cap
    default: 8192            # injected when the request sets no budget (cap if unset)

params:                      # one rule per top-level request field
  temperature: { default: 1.0, min: 0, max: 2 }
  reasoning_format: { values: [general, deepseek-style] }

errors:
  bad_request_type: request_params_invalid

protocols:                   # each enables its routes; a disabled route answers 404
  chat: true
  messages: true
  responses: true

messages:
  thinking_blocks: always    # always (default) | on_request
```

### `params`

Keys are chat request field names: `temperature`, `top_p`, `top_k`,
`frequency_penalty`, `presence_penalty`, `n`, `min_p`, `reasoning_format`,
and so on. Any top-level field can have a rule.

| Field | Meaning |
|---|---|
| `default` | Injected when the request omits the field (an explicit `null` counts as omitted) |
| `pin` | The only accepted value, also injected when omitted. Cannot be combined with `default`, `min`, `max` or `values` |
| `min`, `max` | Inclusive bounds |
| `exclusive_min`, `exclusive_max` | `true` makes that bound exclusive (for example `top_p` in (0, 1]) |
| `values` | Accepted values |
| `type` | `int` rejects non-integers such as `n: 1.5`; `number` is the default |
| `normalize` | `[[from, to], …]`: rewrites an accepted value before forwarding |

A request value goes through these steps in order:

1. **Check**: `pin`, then `values`, then `type`, then `min`/`max`. A failure
   returns 400.
2. **Normalize**: rewrite the value with `normalize`.
3. **Inject**: if the request omitted the field, add `pin` or `default`,
   normalized the same way.

Numbers are compared the way the engine reads them, so `1`, `1.0` and `"1"`
are equal. A value that is not a number at all (`"hot"`, `true`) skips the
numeric checks and is left to the engine to reject.

Because `normalize` runs after the check, rules are written in the client's
vocabulary. StepFun's `top_k: 0` means "no limit": it passes `min: 0` and is
then forwarded as the engine's `-1`:

```yaml
params:
  top_k: { min: 0, type: int, normalize: [[0, -1]] }
```

`reasoning_format` has built-in behaviour. When a chat request sends
`general`, the reply's `reasoning_content` is renamed to `reasoning`, in both
buffered and streaming replies. It only takes effect if `reasoning_format`
has a rule here.

### `output.max_tokens`

The budget is read with the engine's precedence: `max_completion_tokens`,
unless it is `0`, otherwise `max_tokens`. A value above `cap` is rejected or
clamped according to `on_exceed`. A request that sets no budget gets
`default`, or `cap` when there is no `default`.

### `errors.bad_request_type`

This replaces the `type` of every 400 on the inference routes: the router's
own rejections and the engine's. The envelope stays the protocol's own: the
OpenAI `{"error": {...}}` form for chat and Responses, the Anthropic
`{"type": "error", "error": {...}}` form for Messages, and sglang's flat
`{"type", "message", …}` for engine chat errors.

### `messages.thinking_blocks`

When `/v1/messages` replies include `thinking` blocks.

| Value | Behaviour |
|---|---|
| `always` (default) | Whenever the model reasons. Suits always-reasoning models such as Step-5, and clients that time the reasoning phase |
| `on_request` | Only when the request sets `thinking.type` to `enabled` or `adaptive`, which is Anthropic's default. Otherwise the router drops the blocks from buffered and streamed replies. The model still reasons and those tokens are still billed |

### sglang-native `/generate`

`/generate` is always served (it is not a `protocols` entry) and follows the
same profile, read where `GenerateReqInput` keeps each field:

- `params` rules for sampling fields (`temperature`, `top_p`, `top_k`,
  `frequency_penalty`, `presence_penalty`, `n`) apply to
  `sampling_params.<name>`. Other rules, such as `reasoning_format`, have no
  `/generate` spelling and are not applied. Neither is `limits.max_images`.
- `output.max_tokens` applies to `sampling_params.max_new_tokens`. It is
  enforced but never injected: an absent value takes the engine's default
  (128), already under any sane cap. An explicit `null` asks the engine for
  unbounded output, so it counts as over the cap.
- `models.aliases`, `limits.max_body_bytes` and `errors.bad_request_type`
  apply as on the other routes.

## `extends` and merging

A child file is merged onto its parent:

- **Maps merge key by key.** This applies inside `params` too: a child
  `temperature: {max: 1.5}` keeps the parent's `default` and `min`.
- **Scalars and lists replace** the parent's value.
- **`null` deletes**: `params: {n: null}` removes an inherited rule.

Chains are allowed. A cycle is a startup error.

```yaml
extends: stepfun-step5
params:
  temperature: { max: 1.5 }
  n: null
```

## Legacy flags

`--max-output-tokens` and `--override-sampling-params` /
`--sampling-param-conflict` still work. They are applied on top of the
profile:

| Flag | Becomes |
|---|---|
| `--max-output-tokens N` | `output.max_tokens.cap` (the profile's `on_exceed` and `default` are kept) |
| `--override-sampling-params` exact value, `reject` | `pin` |
| `--override-sampling-params` exact value, `allow` | `default` |
| `--override-sampling-params` `{"min", "max"}` band | `min` / `max` |

A legacy flag replaces the whole rule for each parameter it names. Prefer
profiles for new deployments.

## Startup validation

The router refuses to start, naming the source and the key, when the profile:

- has an unknown key, or a value of the wrong type;
- has `min > max`, or a `default` or `pin` its own rule would reject;
- combines `pin` with `default`, `min`, `max` or `values`;
- has a `normalize` `from` value the rule never accepts;
- has a zero `cap`, size or image limit, or an `output.max_tokens.default`
  above `cap`;
- has an `extends` that does not resolve, or forms a cycle.

## Error responses

A rejected request gets a 400 that names the chat field and the accepted
range:

```json
{"error": {"type": "invalid_request_error", "code": "bad_request",
           "message": "bad request: temperature must be between 0 and 2, got 2.1"}}
```

The same request on `/v1/messages`:

```json
{"type": "error", "error": {"type": "invalid_request_error",
                            "message": "bad request: temperature must be between 0 and 2, got 2.1"}}
```

## Examples

### Per-deployment override through a ConfigMap

```yaml
# ConfigMap key profile.yaml
extends: stepfun-step5
models:
  aliases: [step-5-preview]
output:
  max_tokens: { cap: 32000 }
```

```yaml
# Router container
args: ["--api-profile-file", "/etc/router/profile.yaml", ...]
volumeMounts:
  - { name: router-profile, mountPath: /etc/router }
```

### A new vendor

Add `profiles/<vendor>.yaml` and a line to `PRESETS` in
`src/profile/load.rs`. The `every_preset_loads_and_validates` test covers it
automatically.
