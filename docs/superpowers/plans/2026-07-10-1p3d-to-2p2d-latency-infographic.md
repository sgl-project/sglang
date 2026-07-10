# 1P3D to 2P2D Latency Infographic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and verify one high-resolution PNG that matches the approved reference layout while accurately visualizing the executed latency path of the `121443` 1P3D-to-2P2D run.

**Architecture:** Use the supplied PNG only as a composition and styling reference. Generate a new infographic with the built-in image generation tool, inspect the rendered result for text and flow accuracy, make at most one focused correction pass, then copy the accepted image into the repository's `reports` directory.

**Tech Stack:** Built-in `image_gen`, Codex `view_image`, PowerShell filesystem verification, Git.

## Global Constraints

- The actual highlighted path is the `pd_flip_1p3d_to_2p2d_20260708_121443` run.
- The figure must end at `1P3D / SAFE`; `2P2D / SAFE` must be gray and labeled unexecuted.
- Use controller total `19.387 s`, controller migration `2.271 s`, and KV transfer `1.912 s target / 1.806 s source` exactly.
- Show the 2.009-second idle assertion timeout and all four rollback action latencies.
- Describe quiesce as a fixed observation window and KV transfer as a worker-side lifecycle proxy.
- State that prefill/decode KV stitching was not exercised.
- Do not overwrite the supplied reference image.
- Final path: `reports/pd_flip_1p3d_to_2p2d_full_chain_latency.png`.

---

### Task 1: Lock the generation brief

**Files:**
- Read: `docs/superpowers/specs/2026-07-10-1p3d-to-2p2d-latency-infographic-design.md`
- Read: `experiments/pd_flip_todo_20260708_raw_bundle/pd_flip_1p3d_to_2p2d_20260708_121443/pd_state_machine_full_chain_latency.csv`
- Read: `experiments/pd_flip_todo_20260708_raw_bundle/pd_flip_1p3d_to_2p2d_20260708_121443/01_1p3d_to_2p2d_two_phase/migration_link/controller_actions.csv`
- Reference image: `C:/Users/TIANCI~1/AppData/Local/Temp/codex-clipboard-0226fee3-a08f-4e80-9277-9fb7ad22dfc7.png`

**Interfaces:**
- Consumes: approved design and raw experiment values.
- Produces: one exact image-generation prompt with the reference image labeled as a style/layout reference.

- [ ] **Step 1: Re-open the reference image**

Use `view_image` at original detail on the exact reference path. Confirm a white 16:9 technical flowchart with blue arrows, orange timing badges, red rollback emphasis, rounded process boxes, and a small source note.

- [ ] **Step 2: Verify the output target is unused**

Run:

```powershell
Test-Path -LiteralPath 'reports\pd_flip_1p3d_to_2p2d_full_chain_latency.png'
```

Expected: `False` before generation.

- [ ] **Step 3: Assemble the exact generation prompt**

Use this exact prompt:

```text
Use case: infographic-diagram
Asset type: technical experiment result figure for a systems research report
Input images: Image 1 is a style and layout reference only, not an edit target. Create a new infographic.
Primary request: Create a polished 16:9 landscape flowchart titled exactly "1P3D → 2P2D：真实 KV 迁移全流程延迟". Visualize the actual executed path of run pd_flip_1p3d_to_2p2d_20260708_121443. The run transferred four KV entries, committed the target, and finished the source, but the post-migration idle assertion timed out; rollback ran and the final topology stayed 1P3D / SAFE. Do not depict this as a successful 2P2D transition.
Scene/backdrop: clean white technical-slide background.
Style/medium: crisp vector-like systems infographic matching Image 1: rounded white process boxes with thin black outlines, blue arrows, orange latency pills, cyan summary badges, red timeout/rollback emphasis, gray unexecuted path, modern sans-serif typography, generous whitespace.
Composition/framing: top title; below it a summary band; then three readable zones. Zone 1 shows pre-transfer preparation. Zone 2 shows migration, commit, finish, and a decision diamond. Zone 3 branches to the actual red rollback path and a gray unexecuted success tail. Keep arrows unambiguous and avoid crossing labels.
Text (verbatim):
"初始拓扑：1P3D"
"Controller total: 19.387 s"
"Controller migration: 2.271 s"
"4/4 KV entries transferred"
"router drain source" / "0.34 ms"
"pause source admission" / "5.47 ms"
"observe source quiesce（固定观察窗口）" / "15.020 s"
"scan running + waiting" / "0.02 ms"
"build manifests" / "0.22 ms"
"freeze waiting queue" / "0.01 ms"
"start source migration API" / "33.29 ms"
"target prepare receiver API" / "15.91 ms"
"KV transfer → target held" / "1.912 s target / 1.806 s source"
"target commit" / "8.65 ms"
"source finish" / "13.36 ms"
Decision diamond: "source idle?"
Red result tag: "NO — 2.009 s TIMEOUT"
Actual rollback boxes: "target abort" / "13.50 ms"; "source abort" / "6.87 ms"; "cleanup resume admission" / "10.11 ms"; "cleanup router undrain" / "0.41 ms"; terminal "1P3D / SAFE"
Gray unexecuted branch label: "本次未执行 / n/a"
Gray boxes: "set runtime role: n/a"; "refresh router role: n/a"; "normal resume admission: n/a"; "normal router undrain: n/a"; terminal "2P2D / SAFE（未到达）"
Bottom notes: "15.020 s 为固定观察窗口；1.912 s 为 worker-side transfer lifecycle proxy，并非纯网络延迟。" and "Prefill/decode KV stitching 未在本次 runner 中执行。"
Bottom source footer: "Latency source: pd_flip_1p3d_to_2p2d_20260708_121443"
Constraints: Render all quoted text verbatim and legibly. Preserve every number exactly. Highlight the actually executed path, including commit then timeout then rollback. Do not show Target held queue = 0 ms. Do not imply that stage labels add up to total time. No logos, no watermark, no decorative clip art, no extra metrics, no invented steps.
```

### Task 2: Generate the infographic

**Files:**
- Create indirectly: built-in generated image under the tool's returned `$CODEX_HOME/generated_images/...` path.

**Interfaces:**
- Consumes: the exact prompt from Task 1 and Image 1 as the sole reference.
- Produces: one landscape infographic candidate and its returned local path.

- [ ] **Step 1: Invoke built-in image generation**

Call the built-in image generation tool once with the reference image included and require a clean high-resolution 16:9 landscape PNG. Do not request transparent output, CLI mode, or direct destination control.

- [ ] **Step 2: Capture the generated path**

Use the tool result's generated-image path as `$generatedPath` for inspection and final copy. Do not assume an OS temporary directory.

### Task 3: Inspect and correct the candidate

**Files:**
- Read: `$generatedPath` returned by Task 2.

**Interfaces:**
- Consumes: the generated candidate.
- Produces: an accepted candidate whose visible text, flow direction, colors, and outcome match the approved design.

- [ ] **Step 1: Inspect at original detail**

Open `$generatedPath` with `view_image`. Confirm all of the following:

- Title reads `1P3D → 2P2D：真实 KV 迁移全流程延迟`.
- Summary shows `19.387 s`, `2.271 s`, and `4/4`.
- Executed path includes `0.34 ms`, `5.47 ms`, `15.020 s`, `0.02 ms`, `0.22 ms`, `0.01 ms`, `33.29 ms`, `15.91 ms`, `1.912 s target / 1.806 s source`, `8.65 ms`, `13.36 ms`, and `2.009 s TIMEOUT`.
- Rollback path includes `13.50 ms`, `6.87 ms`, `10.11 ms`, and `0.41 ms` and ends at `1P3D / SAFE`.
- Unexecuted success tail is gray, contains four `n/a` actions, and ends at `2P2D / SAFE（未到达）`.
- The figure does not label `Target held queue` as `0 ms`.
- The notes state fixed observation window, worker-side proxy, and stitching not exercised.

- [ ] **Step 2: Perform one focused correction pass only if validation fails**

Re-run built-in image generation with the candidate included as the edit target and use this exact correction instruction:

```text
Change only incorrect, misspelled, missing, or duplicated text labels and any arrow that misstates the execution order. Preserve the existing canvas, layout, colors, spacing, and every already-correct label. The actual path must be commit → source finish → source idle? → NO, 2.009 s TIMEOUT → target abort → source abort → cleanup resume admission → cleanup router undrain → 1P3D / SAFE. The gray n/a branch must remain visibly unexecuted and end at 2P2D / SAFE（未到达）. Preserve every approved numeric value exactly. Do not add Target held queue = 0 ms, logos, watermarks, or new steps.
```

Re-inspect at original detail after the correction.

### Task 4: Persist and verify the final artifact

**Files:**
- Create: `reports/pd_flip_1p3d_to_2p2d_full_chain_latency.png`

**Interfaces:**
- Consumes: accepted `$generatedPath`.
- Produces: the project-bound final PNG.

- [ ] **Step 1: Copy the accepted image into the repository**

Run with `$generatedPath` set to the exact returned built-in path:

```powershell
Copy-Item -LiteralPath $generatedPath -Destination 'reports\pd_flip_1p3d_to_2p2d_full_chain_latency.png'
```

Expected: one new PNG; the reference image remains unchanged.

- [ ] **Step 2: Verify file identity and dimensions**

Run:

```powershell
Get-Item -LiteralPath 'reports\pd_flip_1p3d_to_2p2d_full_chain_latency.png' | Select-Object FullName,Length,LastWriteTime
```

Expected: non-zero file size at the exact final path. Open the copied file with `view_image` at original detail and confirm it matches the accepted candidate.

- [ ] **Step 3: Check repository scope**

Run:

```powershell
git status --short -- 'reports/pd_flip_1p3d_to_2p2d_full_chain_latency.png'
```

Expected: only the intended image appears for this path.
