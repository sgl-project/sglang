#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");

function parseArgs(argv) {
  const values = {};
  for (let index = 2; index < argv.length; index += 1) {
    const name = argv[index];
    if (!name.startsWith("--")) throw new Error(`unexpected argument ${name}`);
    values[name.slice(2)] = argv[++index];
  }
  if (!values.url || !values.output || !values["artifact-dir"]) {
    throw new Error("--url, --output, and --artifact-dir are required");
  }
  return {
    url: values.url,
    output: path.resolve(values.output),
    artifactDir: path.resolve(values["artifact-dir"]),
    referenceImage: values["reference-image"]
      ? path.resolve(values["reference-image"])
      : "",
    useDefaultReference: values["use-default-reference"] === "true",
    timeoutMs: Number(values["timeout-ms"] || 300000),
  };
}

async function run(args) {
  const { chromium } = require("playwright");
  const launch = { headless: true };
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE) {
    launch.executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;
  }
  fs.mkdirSync(args.artifactDir, { recursive: true });
  const browser = await chromium.launch(launch);
  const context = await browser.newContext({
    acceptDownloads: true,
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();
  const downloadPromises = [];
  page.on("download", (download) => {
    downloadPromises.push((async () => {
      const target = path.join(args.artifactDir, download.suggestedFilename());
      await download.saveAs(target);
      return target;
    })());
  });

  try {
    const url = new URL(args.url);
    url.searchParams.set("mode", "i2v");
    await page.goto(url.toString(), { waitUntil: "networkidle", timeout: 60000 });
    await page.selectOption("#generationMode", "i2v");
    await page.fill("#fps", "24");
    await page.fill("#numFrames", "9");
    await page.fill("#guidance", "0");
    await page.fill(
      "#prompt",
      "A smooth forward flight preserving the reference composition and daylight",
    );
    await page.check("#continuous");
    if (args.useDefaultReference) {
      await page.waitForFunction(() => {
        const label = document.querySelector("#referenceName")?.textContent || "";
        return /Reactor LingBot preset/i.test(label);
      }, null, { timeout: 60000 });
    } else if (args.referenceImage) {
      await page.setInputFiles("#firstFrame", args.referenceImage);
    } else {
      const referenceResponse = await context.request.get(
        "https://raw.githubusercontent.com/robbyant/lingbot-world/main/examples/00/image.jpg",
      );
      if (!referenceResponse.ok()) {
        throw new Error(`reference fixture returned HTTP ${referenceResponse.status()}`);
      }
      await page.setInputFiles("#firstFrame", {
        name: "lingbot-reference.jpg",
        mimeType: "image/jpeg",
        buffer: await referenceResponse.body(),
      });
    }
    if (!args.useDefaultReference) {
      await page.waitForFunction(() => {
        const label = document.querySelector("#referenceName")?.textContent || "";
        return label && !/preset reference/i.test(label);
      }, null, { timeout: 60000 });
    }

    await page.click("#recordBtn");
    await page.waitForFunction(
      () => document.querySelector("#recordBtn")?.getAttribute("aria-pressed") === "true",
      null,
      { timeout: 10000 },
    );
    await page.click("#connectBtn");
    try {
      await page.waitForFunction(
        () => window.__sglangRealtimeDebug?.().socketReadyState === WebSocket.OPEN,
        null,
        { timeout: 60000, polling: 100 },
      );
    } catch (error) {
      const diagnostics = await page.evaluate(() => ({
        debug: window.__sglangRealtimeDebug?.(),
        history: document.querySelector("#historyList")?.textContent || "",
        reference: document.querySelector("#referenceName")?.textContent || "",
        status: document.querySelector("#statusText")?.textContent || "",
      }));
      throw new Error(`I2V WebSocket did not open: ${JSON.stringify(diagnostics)}`, {
        cause: error,
      });
    }

    await page.keyboard.down("w");
    await page.fill(
      "#prompt",
      "Continue forward and turn gently left while preserving the reference subject",
    );
    await page.click("#sendPromptBtn");
    await page.waitForTimeout(350);
    await page.keyboard.up("w");
    await page.waitForFunction(
      () => (window.__sglangRealtimeDebug?.().frames || 0) >= 9,
      null,
      { timeout: args.timeoutMs, polling: 250 },
    );

    await page.click("#tracePaneButton");
    await page.waitForFunction(
      () => {
        const text = (id) => document.querySelector(id)?.textContent?.trim() || "-";
        return Number(text("#traceEventCountText")) > 0
          && text("#traceSchedulerText") !== "-"
          && text("#traceDenoiseText") !== "-"
          && text("#traceVaeDecodeText") !== "-";
      },
      null,
      { timeout: 30000, polling: 500 },
    );
    const traceView = await page.evaluate(() => ({
      eventCount: Number(document.querySelector("#traceEventCountText")?.textContent || 0),
      traceId: document.querySelector("#traceIdText")?.textContent || "",
      chunk: document.querySelector("#traceChunkText")?.textContent || "",
      chunkTotal: document.querySelector("#traceChunkTotalText")?.textContent || "",
      scheduler: document.querySelector("#traceSchedulerText")?.textContent || "",
      denoise: document.querySelector("#traceDenoiseText")?.textContent || "",
      vaeDecode: document.querySelector("#traceVaeDecodeText")?.textContent || "",
    }));
    await page.click("#previewPaneButton");

    await page.click("#recordBtn");
    await page.waitForFunction(
      () => {
        const button = document.querySelector("#recordBtn");
        return button?.getAttribute("aria-pressed") === "false" && !button.disabled;
      },
      null,
      { timeout: 60000, polling: 250 },
    );
    const deadline = Date.now() + 30000;
    while (downloadPromises.length < 3 && Date.now() < deadline) {
      await page.waitForTimeout(100);
    }
    if (downloadPromises.length < 3) {
      throw new Error(`recording emitted ${downloadPromises.length} of 3 downloads`);
    }
    const downloaded = await Promise.all(downloadPromises);
    await page.click("#stopBtn");

    const jsonPath = downloaded.find((file) => file.endsWith(".json"));
    const htmlPath = downloaded.find((file) => file.endsWith(".html"));
    const videoPath = downloaded.find((file) => /\.(mp4|webm)$/i.test(file));
    if (!jsonPath || !htmlPath || !videoPath) {
      throw new Error(`recording downloads are incomplete: ${downloaded.join(", ")}`);
    }
    const artifact = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
    const replayHtml = fs.readFileSync(htmlPath, "utf8");
    const eventNames = new Set((artifact.events || []).map((event) => event.kind));
    const hasAction = (artifact.events || []).some((event) => (
      event.kind === "key_down" || event.kind === "camera_actions_sent"
    ));
    const hasPromptUpdate = (artifact.prompt_history || []).length >= 2;
    const reference = artifact.request?.reference_image || artifact.reference_image;
    if (artifact.request?.generation_mode !== "i2v") {
      throw new Error("recording artifact did not preserve I2V generation mode");
    }
    if (!reference) throw new Error("recording artifact did not preserve the reference image");
    if (!hasAction) throw new Error("recording artifact did not preserve keyboard actions");
    if (!hasPromptUpdate) throw new Error("recording artifact did not preserve prompt updates");
    if (!replayHtml.includes(path.basename(videoPath))) {
      throw new Error("replay HTML does not reference the recorded video");
    }
    if (!replayHtml.includes("replayInspector")) {
      throw new Error("replay HTML does not contain the cursor event inspector");
    }

    return {
      mode: artifact.request.generation_mode,
      reference: {
        source: reference.source || "",
        mime: reference.mime || "",
        bytes: Number(reference.bytes || 0),
      },
      recording: {
        frames: Number(artifact.recording?.frames || 0),
        videoBytes: fs.statSync(videoPath).size,
        files: downloaded.map((file) => path.basename(file)).sort(),
      },
      trajectory: {
        events: (artifact.events || []).length,
        eventTypes: [...eventNames].sort(),
        promptUpdates: (artifact.prompt_history || []).length,
      },
      trace: traceView,
    };
  } finally {
    await browser.close();
  }
}

if (require.main === module) {
  const args = parseArgs(process.argv);
  run(args)
    .then((result) => {
      fs.mkdirSync(path.dirname(args.output), { recursive: true });
      fs.writeFileSync(args.output, `${JSON.stringify(result, null, 2)}\n`);
      console.log(`browser feature probe passed: ${args.output}`);
    })
    .catch((error) => {
      console.error(error.stack || error);
      process.exitCode = 1;
    });
}

module.exports = { parseArgs, run };
