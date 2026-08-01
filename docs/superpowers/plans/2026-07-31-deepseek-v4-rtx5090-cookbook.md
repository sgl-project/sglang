# DeepSeek-V4 RTX 5090 Cookbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an unverified RTX 5090 option that generates the validated TP8 DeepSeek-V4-Flash-0731 low-latency command.

**Architecture:** Keep RTX 5090 model-specific by extending the plain-data DeepSeek-V4 config. Add one exact deployment cell so the shared deployment engine can select the hardware and generate the command without engine or benchmark changes.

**Tech Stack:** JSX configuration data, Node.js static assertions, Mintlify validation.

## Global Constraints

- The hardware id is `rtx5090`, label is `RTX 5090`, VRAM is `32GB`, and vendor group is `blackwell`.
- The only new tuple is `rtx5090 | flash-official | fp4 | low-latency | single`.
- The cell remains `verified: false`.
- Do not add benchmark data, HiCache, shared-catalog changes, or cookbook prose.
- The generated command uses TP8, `flashinfer_mxfp4`, `--mem-fraction-static 0.90`, and `--cuda-graph-max-bs-decode 32`.

---

### Task 1: Add and validate the RTX 5090 deployment cell

**Files:**
- Modify: `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`
- Test: focused Node.js assertions against `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`

**Interfaces:**
- Consumes: the deployment engine's existing `supportedHardware`, model-specific `hardware`, and `cells` config fields.
- Produces: hardware id `rtx5090` and one matching deployment cell for the command generator.

- [ ] **Step 1: Run the focused assertion before editing**

```bash
node --input-type=module <<'NODE'
import fs from "node:fs";
import vm from "node:vm";
const path = "docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx";
const source = fs.readFileSync(path, "utf8").replace("export const config =", "config =");
const sandbox = {};
vm.runInNewContext(source, sandbox);
const config = sandbox.config;
const match = { hw: "rtx5090", variant: "flash-official", quant: "fp4", strategy: "low-latency", nodes: "single" };
const cells = config.cells.filter((cell) => Object.entries(match).every(([key, value]) => cell.match[key] === value));
if (!config.supportedHardware.includes("rtx5090")) throw new Error("rtx5090 is absent from supportedHardware");
if (!config.hardware.some((item) => item.id === "rtx5090" && item.label === "RTX 5090" && item.vram === "32GB" && item.vendor === "blackwell")) throw new Error("RTX 5090 hardware metadata is absent");
if (cells.length !== 1) throw new Error(`expected one RTX 5090 cell, got ${cells.length}`);
NODE
```

Expected: FAIL with `rtx5090 is absent from supportedHardware`.

- [ ] **Step 2: Add the minimal config data**

In `supportedHardware`, add `"rtx5090"` next to `"rtx6000"`. In the model-specific `hardware` array, add:

```jsx
{ id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "blackwell" },
```

Immediately after the RTX PRO 6000 cells, add:

```jsx
{
  match: { hw: "rtx5090", variant: "flash-official", quant: "fp4", strategy: "low-latency", nodes: "single" },
  verified: false,
  env: [],
  flags: [
    "--trust-remote-code",
    "--model-path {{MODEL_NAME}}",
    "--tp 8",
    "--moe-runner-backend flashinfer_mxfp4",
    "--mem-fraction-static 0.90",
    "--cuda-graph-max-bs-decode 32",
    "--host {{HOST_IP}}",
    "--port {{PORT}}",
  ],
},
```

- [ ] **Step 3: Run the focused assertion after editing**

```bash
node --input-type=module <<'NODE'
import fs from "node:fs";
import vm from "node:vm";
const path = "docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx";
const source = fs.readFileSync(path, "utf8").replace("export const config =", "config =");
const sandbox = {};
vm.runInNewContext(source, sandbox);
const config = sandbox.config;
const match = { hw: "rtx5090", variant: "flash-official", quant: "fp4", strategy: "low-latency", nodes: "single" };
const cells = config.cells.filter((cell) => Object.entries(match).every(([key, value]) => cell.match[key] === value));
const expectedFlags = [
  "--trust-remote-code",
  "--model-path {{MODEL_NAME}}",
  "--tp 8",
  "--moe-runner-backend flashinfer_mxfp4",
  "--mem-fraction-static 0.90",
  "--cuda-graph-max-bs-decode 32",
  "--host {{HOST_IP}}",
  "--port {{PORT}}",
];
if (!config.supportedHardware.includes("rtx5090")) throw new Error("rtx5090 is absent from supportedHardware");
if (!config.hardware.some((item) => item.id === "rtx5090" && item.label === "RTX 5090" && item.vram === "32GB" && item.vendor === "blackwell")) throw new Error("RTX 5090 hardware metadata is absent");
if (cells.length !== 1) throw new Error(`expected one RTX 5090 cell, got ${cells.length}`);
if (cells[0].verified !== false) throw new Error("RTX 5090 cell must remain unverified");
if (cells[0].env.length !== 0) throw new Error("RTX 5090 cell must not add environment variables");
if (JSON.stringify(cells[0].flags) !== JSON.stringify(expectedFlags)) throw new Error("RTX 5090 flags do not match the validated command");
NODE
```

Expected: PASS with exit code 0.

- [ ] **Step 4: Run repository documentation checks**

```bash
git diff --check
cd docs_new
mint validate
mint broken-links
```

Expected: all installed checks pass. If Mintlify CLI is unavailable, retain the passing focused assertion and record the missing tool in the PR validation section.

- [ ] **Step 5: Review and commit the implementation**

```bash
git diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
git add docs/superpowers/plans/2026-07-31-deepseek-v4-rtx5090-cookbook.md docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
git commit -m "docs: add RTX 5090 DeepSeek-V4 recipe"
```

Expected: the implementation commit contains the plan and the one config change, with no benchmark or shared-engine files.
