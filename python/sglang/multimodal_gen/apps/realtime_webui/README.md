# SGLang Diffusion Realtime WebUI

Standalone browser demo for `/v1/realtime_video/generate`.

Open `index.html` directly in a browser, point it at an SGLang Diffusion server,
and generate. The app sends msgpack init / event messages and renders lossless
raw RGB frame batches on a canvas.

The first version is intentionally static: no npm install, no build step, and no
server-side dependencies. Presets are UI-side templates for prompt, LingBot
example images, album artwork references, and session parameters. The default
preset preloads a reference image so the demo can be tested without a file
upload.

By default, `Continuous session` is enabled for long-running camera control.
Keyboard and pointer controls send state transitions instead of scripted preset
actions. The telemetry `Chunk wait` measures request-to-chunk arrival time, not
client-side RGB decode time. Continuous playback adapts to the measured chunk
production rate so the canvas does not play a chunk at target FPS and then sit
on the last frame while waiting for the next chunk.

The interface shape follows camera-control-first video playgrounds such as
Reactor LingBot: reference image, scene prompt, enhancement, clip controls,
move/look camera controls, recordings history, and model telemetry.

## I2V and T2V

Deployments opt in to the mode selector through runtime config:

```json
{
  "generationModes": ["i2v", "t2v"],
  "defaultGenerationMode": "i2v",
  "t2vFrameStep": 4,
  "t2vDefaultNumFrames": 121
}
```

I2V sends `generation_mode: "i2v"` and requires `first_frame`. T2V sends
`generation_mode: "t2v"`, omits `first_frame`, and uses `num_frames` as the
complete output horizon. For MinWM, `num_frames` must equal `1 + N * 4`; the
adapter derives the exact chunk count, so the WebUI does not send `max_chunks`
for T2V. The `mode=i2v|t2v` query parameter can select an enabled mode.
