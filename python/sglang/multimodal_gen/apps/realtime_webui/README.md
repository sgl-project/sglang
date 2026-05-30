# SGLang Diffusion Realtime WebUI

Standalone browser demo for `/v1/realtime_video/generate`.

Open `index.html` directly in a browser, point it at an SGLang Diffusion server,
and generate. The app sends msgpack init / event messages and renders raw RGB
frame batches on a canvas.

The first version is intentionally static: no npm install, no build step, and no
server-side dependencies. Presets are UI-side templates for prompt, camera
action chunks, and session parameters.

The interface shape follows camera-control-first video playgrounds such as
Reactor LingBot: reference image, scene prompt, enhancement, clip controls,
move/look camera controls, recordings history, and model telemetry.
