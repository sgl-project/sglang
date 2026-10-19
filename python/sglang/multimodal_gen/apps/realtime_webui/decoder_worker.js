const RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb";
const WEBP_FRAME_CONTENT_TYPE = "image/webp";
const JPEG_FRAME_CONTENT_TYPE = "image/jpeg";

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

function splitEncodedPayload(header, payload) {
  const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  const lengths = Array.isArray(header.payload_lengths) && header.payload_lengths.length
    ? header.payload_lengths.map(Number)
    : [bytes.byteLength];
  const payloads = [];
  let offset = 0;
  for (const length of lengths) {
    payloads.push(bytes.buffer.slice(
      bytes.byteOffset + offset,
      bytes.byteOffset + offset + length,
    ));
    offset += length;
  }
  return payloads;
}

async function encodedFramesToImageBitmaps(header, payload) {
  if (typeof createImageBitmap === "undefined") {
    throw new Error("This browser does not support worker image decoding");
  }

  const frames = await Promise.all(splitEncodedPayload(header, payload).map((framePayload) => (
    createImageBitmap(new Blob([framePayload], { type: header.content_type }))
  )));
  return {
    width: frames[0]?.width || 0,
    height: frames[0]?.height || 0,
    frame_type: "bitmap",
    frames,
  };
}

async function decode(header, payload) {
  let rawPayload;
  if (
    header.content_type === WEBP_FRAME_CONTENT_TYPE ||
    header.content_type === JPEG_FRAME_CONTENT_TYPE
  ) {
    const decoded = await encodedFramesToImageBitmaps(header, payload);
    return {
      id: header.__decode_id,
      width: decoded.width,
      height: decoded.height,
      chunk: Number(header.chunk_index),
      frame_type: decoded.frame_type,
      frames: decoded.frames,
    };
  } else if (header.content_type === RAW_RGB_CONTENT_TYPE) {
    rawPayload = new Uint8Array(payload);
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
    const result = await decode(message.header, message.payload);
    self.postMessage({ type: "decoded", ...result }, result.frames);
  } catch (error) {
    self.postMessage({
      type: "error",
      id: message.header?.__decode_id,
      message: error.message || "decode failed",
    });
  }
};
