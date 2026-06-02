const RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb";
const RAW_RGB_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgb-delta-gzip";
const RAW_RGBA_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgba-delta-gzip";
const WEBP_FRAME_CONTENT_TYPE = "image/webp";
const JPEG_FRAME_CONTENT_TYPE = "image/jpeg";

let lastFrame = null;

function reset() {
  lastFrame = null;
}

async function gunzipBytes(payload) {
  if (typeof DecompressionStream === "undefined") {
    throw new Error("This browser does not support gzip stream decoding");
  }
  const stream = new Blob([payload]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function restoreDeltaGzipFrames(header, payload) {
  const frameBytes = Number(header.bytes_per_frame);
  const count = Number(header.num_frames);
  const expectedSize = frameBytes * count;
  const restored = await gunzipBytes(payload);
  if (restored.length !== expectedSize) {
    throw new Error(`delta payload size mismatch: expected ${expectedSize}, got ${restored.length}`);
  }

  let previous = header.delta_reference === "previous-frame" ? lastFrame : null;
  if (header.delta_reference === "previous-frame") {
    if (!previous) throw new Error("Missing previous frame for delta payload");
    if (previous.byteLength !== frameBytes) {
      throw new Error("Previous frame size does not match current delta payload");
    }
  }

  for (let f = 0; f < count; f++) {
    const offset = f * frameBytes;
    if (previous) {
      for (let i = 0; i < frameBytes; i++) restored[offset + i] ^= previous[i];
    }
    previous = restored.slice(offset, offset + frameBytes);
  }
  lastFrame = previous;
  return restored;
}

function rawFramesToRgbaBuffers(header, payload) {
  const width = Number(header.width);
  const height = Number(header.height);
  const channels = Number(header.channels);
  const count = Number(header.num_frames);
  const frameBytes = Number(header.bytes_per_frame);
  const pixels = width * height;
  const buffers = [];

  for (let f = 0; f < count; f++) {
    const offset = f * frameBytes;
    if (channels === 4) {
      buffers.push(payload.buffer.slice(
        payload.byteOffset + offset,
        payload.byteOffset + offset + frameBytes,
      ));
      continue;
    }

    const rgba = new Uint8ClampedArray(pixels * 4);
    let src = offset;
    let dst = 0;
    for (let p = 0; p < pixels; p++) {
      rgba[dst++] = payload[src++];
      rgba[dst++] = payload[src++];
      rgba[dst++] = payload[src++];
      src += channels - 3;
      rgba[dst++] = 255;
    }
    buffers.push(rgba.buffer);
  }
  return buffers;
}

async function encodedFrameToRgbaBuffers(header, payload) {
  if (typeof createImageBitmap === "undefined" || typeof OffscreenCanvas === "undefined") {
    throw new Error("This browser does not support worker image decoding");
  }

  const blob = new Blob([payload], { type: header.content_type });
  const bitmap = await createImageBitmap(blob);
  const width = bitmap.width;
  const height = bitmap.height;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { alpha: false });
  ctx.drawImage(bitmap, 0, 0);
  const image = ctx.getImageData(0, 0, width, height);
  bitmap.close?.();
  return {
    width,
    height,
    frames: [image.data.buffer],
  };
}

async function decode(header, payload) {
  let rawPayload;
  if (
    header.content_type === WEBP_FRAME_CONTENT_TYPE ||
    header.content_type === JPEG_FRAME_CONTENT_TYPE
  ) {
    const decoded = await encodedFrameToRgbaBuffers(header, payload);
    return {
      id: header.__decode_id,
      width: decoded.width,
      height: decoded.height,
      chunk: Number(header.chunk_index),
      frames: decoded.frames,
    };
  } else if (header.content_type === RAW_RGB_CONTENT_TYPE) {
    rawPayload = new Uint8Array(payload);
    const frameBytes = Number(header.bytes_per_frame);
    const count = Number(header.num_frames);
    lastFrame = count > 0
      ? rawPayload.slice((count - 1) * frameBytes, count * frameBytes)
      : null;
  } else if (
    header.content_type === RAW_RGB_DELTA_GZIP_CONTENT_TYPE ||
    header.content_type === RAW_RGBA_DELTA_GZIP_CONTENT_TYPE
  ) {
    rawPayload = await restoreDeltaGzipFrames(header, payload);
  } else {
    throw new Error(`Unsupported content type ${header.content_type}`);
  }

  return {
    id: header.__decode_id,
    width: Number(header.width),
    height: Number(header.height),
    chunk: Number(header.chunk_index),
    frames: rawFramesToRgbaBuffers(header, rawPayload),
  };
}

self.onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === "reset") {
      reset();
      return;
    }
    const result = await decode(message.header, message.payload);
    self.postMessage(
      { type: "decoded", ...result },
      result.frames,
    );
  } catch (error) {
    self.postMessage({
      type: "error",
      id: message.header?.__decode_id,
      message: error.message || "decode failed",
    });
  }
};
