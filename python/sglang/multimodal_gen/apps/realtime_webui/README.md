# SGLang Diffusion Realtime WebUI

Standalone browser demo for `/v1/realtime_video/generate`.

Open `index.html` directly in a browser, point it at an SGLang Diffusion server,
and generate. The app sends msgpack init / event messages and renders raw RGB
frame batches on a canvas.

The first version is intentionally static: no npm install, no build step, and no
server-side dependencies. Presets are UI-side templates for prompt, camera
action chunks, LingBot example images, album artwork references, and session
parameters. The default preset preloads a reference image so the demo can be
tested without a file upload.

By default, the UI requests `max_chunks=1` so the server stops after the first
chunk. Enable `Continuous session` when testing long-running camera control.
Preset camera actions are included in the init request as initial condition
inputs, so the first chunk uses the selected preset instead of waiting for a
post-init event race. The telemetry `Chunk wait` measures request-to-chunk
arrival time, not client-side RGB decode time. Continuous playback adapts to the
measured chunk production rate so the canvas does not play a chunk at target FPS
and then sit on the last frame while waiting for the next chunk.

The interface shape follows camera-control-first video playgrounds such as
Reactor LingBot: reference image, scene prompt, enhancement, clip controls,
move/look camera controls, recordings history, and model telemetry.
