# GR00T-N1.7

[GR00T-N1.7](https://huggingface.co/nvidia/GR00T-N1.7-3B) is NVIDIA's Vision-Language-Action (VLA) model for humanoid and manipulator robots. Built on top of Qwen3-VL (Cosmos-Reason2-2B) with a flow-matching DiT action head, it takes multi-camera images, a natural-language instruction, an embodiment tag, and the current proprio state, and predicts a 40-step, 132-dim future action trajectory.

Key features:
- Multi-camera input (RGB images — one or more per request)
- Flow-matching action head (4-step Euler integration)
- 32 supported embodiments via an embodiment-id projector (DROID, Unitree G1, R1 Pro, LIBERO, SimplerEnv, ...)
- Output: `[action_horizon=40, max_action_dim=132]` tensor in the normalized/padded action space — client-side Isaac-GR00T processing converts it back to physical joint commands
- VL backbone hidden state is taken at `select_layer=16` and fed to the action head (matches Isaac-GR00T's `Gr00tPolicy`)

## Launch Server

```{note}
The custom DiT Euler loop in the action head composes more cleanly without CUDA graph capture, so `--disable-cuda-graph` is required.
```

The `--tokenizer-path` is optional. If omitted, SGLang will automatically download the tokenizer from `nvidia/Cosmos-Reason2-2B`:

```shell
python3 -m sglang.launch_server \
    --model-path nvidia/GR00T-N1.7-3B \
    --tokenizer-path nvidia/Cosmos-Reason2-2B \
    --port 30000 \
    --disable-cuda-graph
```

## Inference Example

GR00T-N1.7 uses the SGLang VLA contract: the client sends raw images + language instruction via the standard OpenAI `chat.completions` message format, and sends proprio state + embodiment tag via `extra_body={"history_traj": {...}}`. The server returns the predicted trajectory under `sglext.pred_traj`.


### Dependencies

For loading LeRobot-format demonstrations and for client-side state/action normalization (the statistics must match those the model was trained on), GR00T-N1.7 relies on NVIDIA's [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) package:

```shell
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T && pip install -e .

pip install requests pillow numpy
```

Or run your script through Isaac's `uv` environment:

```shell
cd /path/to/Isaac-GR00T
uv run python your_script.py
```

```{note}
The sglang server itself does NOT need Isaac-GR00T. The dependency is purely client-side, only for convenience helpers (dataset loading, normalize/unnormalize). If you produce normalized proprio state yourself, you can skip the Isaac dependency.
```

### Inference Script

The full tutorial script below (also shipped at `test_online_full.py` in the repo root) loads a DROID demonstration via `LeRobotEpisodeLoader`, drives the running sglang server frame-by-frame, converts normalized predictions back into physical joint commands via Isaac's `policy.processor.decode_action`, and reports physical-space MSE / MAE against the episode's ground-truth actions. Copy it into a file and run with `python3 test_online_full.py` once the sglang server launched above is serving on `localhost:30000`.

```python
"""GR00T-N1.7 on SGLang — end-to-end tutorial / online smoke test.

This script shows how to drive a running sglang server with the
GR00T-N1.7 port, using Isaac-GR00T's `LeRobotEpisodeLoader` to load a
real robot demonstration episode.  It mirrors Isaac-GR00T's
`standalone_inference_script.py` pipeline, except the model-forward
step hits sglang instead of a local `Gr00tPolicy`:

  1. Load a LeRobot-format demonstration episode via
     `LeRobotEpisodeLoader`.  This reuses Isaac's modality-config +
     image decoding plumbing, so the VLM sees the exact same temporal
     context the reference pipeline does.
  2. For each step, extract images + raw proprio state + language
     instruction; NORMALIZE the state with Isaac's
     `state_action_processor`.
  3. POST one OpenAI-format chat request to sglang with
     `extra_body={"history_traj": {"proprio_state": ..., "embodiment": ...}}`.
  4. Read `sglext.pred_traj` (shape [40, 132], normalized RELATIVE
     actions) from the response.
  5. UNNORMALIZE + RELATIVE→ABSOLUTE via `policy.processor.decode_action`.
  6. Compare to ground-truth actions from the episode; print per-step
     and aggregate MSE / MAE in physical units.

--------------------------------------------------------------------------------
PREREQUISITES
--------------------------------------------------------------------------------
1. A running sglang server that loaded GR00T-N1.7-3B (see the "Launch Server"
   section of the docs for the exact flags; `--disable-cuda-graph` is required).

2. NVIDIA's Isaac-GR00T package installed in the current Python env.
   It is only needed CLIENT-SIDE, for LeRobot dataset loading and
   for normalize/unnormalize parity with the official policy — the
   sglang server itself does NOT need it.  Install with:

     git clone https://github.com/NVIDIA/Isaac-GR00T.git
     cd Isaac-GR00T && pip install -e .

   Or run this script through Isaac's uv env:

     cd /path/to/Isaac-GR00T
     uv run test_online_full.py

3. A LeRobot-format demonstration episode.  The demo bundled with
   Isaac-GR00T lives at `Isaac-GR00T/demo_data/droid_sample` (DROID,
   embodiment `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`).  Set the
   `ISAAC_GR00T_ROOT` env var to your Isaac-GR00T checkout, or pass
   `--dataset` to point at your own.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  # with overrides:
  python3 test_online_full.py \
      --traj-id 1 --action-horizon 8 --steps 200

On success the script prints overall MSE / MAE against the episode's
ground-truth actions and returns exit code 0.  Reference number on this
box for DROID trajectory 1 via the official `Gr00tPolicy`:
MSE=0.003289, MAE=0.037619.
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import sys
import time

import numpy as np
import requests
from PIL import Image


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("groot_sglang_tutorial")


# -----------------------------------------------------------------------------
# Isaac-GR00T is required CLIENT-SIDE for data loading + normalization parity.
# Bail out with a clear install message if it isn't importable, so users know
# exactly what to do.
# -----------------------------------------------------------------------------
try:
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
except ImportError as exc:
    sys.stderr.write(
        "\n"
        "================================================================================\n"
        "WARNING: NVIDIA's Isaac-GR00T package is not importable in this Python env.\n"
        "This tutorial uses it client-side for LeRobot dataset loading and for the\n"
        "state/action normalization that must match the trained model's statistics.\n"
        "(The sglang server itself does NOT need Isaac-GR00T — this applies only to\n"
        "running this tutorial script.)\n"
        "\n"
        "Please clone and install it:\n"
        "\n"
        "    git clone https://github.com/NVIDIA/Isaac-GR00T.git\n"
        "    cd Isaac-GR00T && pip install -e .\n"
        "\n"
        "or invoke this script through Isaac's uv env:\n"
        "\n"
        "    cd /path/to/Isaac-GR00T\n"
        "    uv run python test_online_full.py\n"
        "\n"
        f"Underlying ImportError: {exc}\n"
        "================================================================================\n"
    )
    sys.exit(1)


# Override with GR00T_WEIGHTS_PATH if you have a local checkout.
MODEL_PATH = os.environ.get("GR00T_WEIGHTS_PATH", "nvidia/GR00T-N1.7-3B")
# DROID demo bundled with Isaac-GR00T.  Set ISAAC_GR00T_ROOT to your
# Isaac-GR00T checkout, or override with --dataset.
_ISAAC_ROOT = os.environ.get("ISAAC_GR00T_ROOT", ".")
DEFAULT_DATASET = os.path.join(_ISAAC_ROOT, "demo_data", "droid_sample")
DEFAULT_EMBODIMENT = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
DEFAULT_SGLANG_URL = "http://127.0.0.1:30000/v1/chat/completions"


def _img_to_data_url(arr: np.ndarray) -> str:
    """HWC uint8 ndarray → base64 JPEG data URL for OpenAI chat `image_url`."""
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _sglang_predict(
    sglang_url: str,
    images_by_view: dict,          # {view_name: (T, H, W, 3) uint8}
    pre_model_state_dict: dict,    # {state_key: list[float]} already normalized
    embodiment_tag_value: str,
    instruction: str,
    timeout_s: float = 120.0,
) -> np.ndarray:
    """POST one step to the sglang chat endpoint, return pred_traj (40, 132).

    The message embeds every temporal frame of every view so the VLM sees
    the same context Isaac's reference pipeline feeds it.  Pre-model
    (normalized) proprio state + embodiment tag travel in
    `history_traj`, which serving_chat accepts as a top-level field.
    """
    content = []
    for _view_name, frames in images_by_view.items():
        for t in range(frames.shape[0]):
            content.append({
                "type": "image_url",
                "image_url": {"url": _img_to_data_url(frames[t])},
            })
    content.append({"type": "text", "text": instruction})

    payload = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
        "history_traj": {
            "proprio_state": {
                k: [float(x) for x in np.asarray(v).flatten()]
                for k, v in pre_model_state_dict.items()
            },
            "embodiment": embodiment_tag_value,
        },
    }

    resp = requests.post(sglang_url, json=payload, timeout=timeout_s)
    if not resp.ok:
        raise RuntimeError(
            f"sglang HTTP {resp.status_code} {resp.reason}: {resp.text[:500]}"
        )
    data = resp.json()
    sglext = data.get("sglext")
    if sglext is None:
        raise RuntimeError(f"sglang response missing `sglext` block: {data}")
    pred = sglext.get("pred_traj")
    if pred is None:
        raise RuntimeError(f"sglang response missing `pred_traj`: {sglext}")
    arr = np.asarray(pred[0], dtype=np.float32)
    if arr.shape != (40, 132):
        raise RuntimeError(f"expected pred_traj (40, 132), got {arr.shape}")
    return arr


def main() -> int:
    p = argparse.ArgumentParser(
        description="GR00T-N1.7 on SGLang end-to-end tutorial."
    )
    p.add_argument("--traj-id", type=int, default=1,
                   help="Index of the trajectory in the LeRobot dataset to replay.")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="Path to a LeRobot-format dataset directory.")
    p.add_argument("--embodiment-tag", default=DEFAULT_EMBODIMENT,
                   help="Embodiment tag (enum value) for (un)normalization.")
    p.add_argument("--action-horizon", type=int, default=8,
                   help="Consecutive pred-action steps to use before re-planning.")
    p.add_argument("--steps", type=int, default=200,
                   help="Max dataset frames to replay.")
    p.add_argument("--sglang-url", default=DEFAULT_SGLANG_URL,
                   help="URL of the sglang OpenAI-compatible chat endpoint.")
    args = p.parse_args()

    emb = EmbodimentTag.resolve(args.embodiment_tag)
    log.info(f"[tutorial] loading Gr00tPolicy from {MODEL_PATH} "
             f"(embodiment={emb.value})")
    # We stand up Gr00tPolicy only to reuse its modality_configs +
    # state_action_processor (normalize/unnormalize statistics).
    # Model inference itself is delegated to the sglang server.
    policy = Gr00tPolicy(
        embodiment_tag=emb,
        model_path=MODEL_PATH,
        device="cuda:0",
        strict=False,
    )
    modality_configs = dict(policy.get_modality_config())
    sap = policy.processor.state_action_processor

    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset, modality_configs=modality_configs
    )
    log.info(f"[tutorial] dataset {args.dataset} has {len(loader)} trajectories")

    state_keys = modality_configs["state"].modality_keys
    action_keys = modality_configs["action"].modality_keys
    video_keys = modality_configs["video"].modality_keys

    traj = loader[args.traj_id]
    n = min(args.steps, len(traj))
    step_counts = list(range(0, n, args.action_horizon))
    log.info(f"[tutorial] trajectory {args.traj_id}: {n} frames, "
             f"{len(step_counts)} inference calls, action_horizon={args.action_horizon}")

    pred_across_time: list[np.ndarray] = []
    for idx, step in enumerate(step_counts):
        t0 = time.time()
        data_point = extract_step_data(traj, step, modality_configs, emb)

        raw_state = {
            k: np.asarray(data_point.states[k], dtype=np.float32)
            for k in state_keys
        }
        norm_state = sap.apply_state(state=raw_state, embodiment_tag=emb.value)
        imgs = {
            v: np.asarray(data_point.images[v], dtype=np.uint8)
            for v in video_keys
        }

        pred_norm = _sglang_predict(
            args.sglang_url, imgs, norm_state, emb.value, data_point.text
        )
        dt = time.time() - t0
        log.info(f"  [step {idx + 1}/{len(step_counts)}] pred_traj max-abs="
                 f"{np.abs(pred_norm).max():.3f}  ({dt:.2f}s)")

        # Unnormalize + convert RELATIVE→ABSOLUTE via Isaac's processor.
        # decode_action splits the 132-wide pad output back into per-action-
        # key chunks, inverts the normalization, and applies the current raw
        # state for relative-action reconstruction.
        unnorm = policy.processor.decode_action(
            pred_norm[None, ...], emb, state=raw_state
        )
        for j in range(args.action_horizon):
            concat = np.concatenate(
                [np.atleast_1d(np.atleast_1d(unnorm[k][0])[j]) for k in action_keys],
                axis=0,
            )
            pred_across_time.append(concat)

    pred_arr = np.asarray(pred_across_time, dtype=np.float32)

    # Ground-truth action stream, straight from the LeRobot DataFrame —
    # mirrors standalone_inference_script.py's extract_state_joints helper.
    gt_cols = []
    for key in action_keys:
        col = f"action.{key}"
        gt_cols.append(
            np.vstack([np.asarray(a, dtype=np.float32) for a in traj[col]])
        )
    gt_arr = np.concatenate(gt_cols, axis=-1)

    m = min(pred_arr.shape[0], gt_arr.shape[0])
    pred_arr = pred_arr[:m]
    gt_arr = gt_arr[:m]

    mse = float(np.mean((pred_arr - gt_arr) ** 2))
    mae = float(np.mean(np.abs(pred_arr - gt_arr)))
    log.info("\n" + "=" * 80)
    log.info("PHYSICAL-SPACE PARITY (sglang GR00T-N1.7 vs episode ground truth)")
    log.info(f"  traj {args.traj_id}, {m} action steps")
    log.info(f"  MSE = {mse:.6f}")
    log.info(f"  MAE = {mae:.6f}")
    log.info("Reference (Isaac Gr00tPolicy on the same DROID traj): "
             "MSE=0.003289  MAE=0.037619")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`pred_traj` in the model's normalized+padded action space — Isaac's `policy.processor.decode_action` (shown in `main()`) is what turns it back into physical joint commands.

## Request Schema

| Field                            | Type                         | Notes |
|----------------------------------|------------------------------|-------|
| `messages[*].content[*]`         | OpenAI chat array            | Standard `image_url` / `text` parts. Send all temporal frames your embodiment uses. |
| `max_tokens`                     | `int`                        | Set to `1`. GR00T's response is delivered via `sglext`, not via generated text. |
| `history_traj.proprio_state`     | `dict[str, list[float]]`     | Per-modality key → 1-D list of already-normalized state values. The order of values per key must match the embodiment's modality config. |
| `history_traj.embodiment`        | `str`                        | One of the tags in the [embodiment table](#embodiment-tag-table) below. |

`history_traj` may be sent at the top level or under `extra_body`; both are accepted by `/v1/chat/completions`.

## Response Schema

GR00T attaches its prediction to the standard OpenAI response under a `sglext` block:

```text
{
  "choices": [...],
  "sglext": {
    "pred_traj": [[[ ... ]]]   // shape: [batch, action_horizon=40, max_action_dim=132]
  }
}
```

- `sglext.pred_traj` — `List[Optional[List[List[float]]]]`, one entry per request in the batch. Each non-null entry is a nested list of shape `[action_horizon=40, max_action_dim=132]` in the model's normalized+padded action space. For a request that did not carry `history_traj`, the entry is `null`.

## Embodiment Tag Table

The `embodiment` string selects the per-embodiment projector index inside the action head. Tags map to `EMBODIMENT_TAG_TO_PROJECTOR_INDEX` in `python/sglang/srt/multimodal/processors/groot_n1d7.py`:

| Embodiment tag                                       | Projector index |
|------------------------------------------------------|-----------------|
| `simpler_env_google`                                 | 0               |
| `simpler_env_widowx`                                 | 1               |
| `libero_sim`                                         | 2               |
| `new_embodiment`                                     | 10              |
| `oxe_droid_relative_eef_relative_joint`              | 24              |
| `real_g1_relative_eef_relative_joints`               | 25              |
| `unitree_g1_full_body_with_waist_height_nav_cmd`     | 25              |
| `real_r1_pro_sharpa_relative_eef`                    | 26              |
| `real_r1_pro_sharpa_relative_eef_human`              | 26              |
| `real_r1_pro_sharpa_relative_eef_maxinsights`        | 26              |
| `real_r1_pro_sharpa_relative_eef_mecka`              | 26              |
| `xdof_relative_eef_relative_joint`                   | 27              |
| `xdof_relative_eef_relative_joint_subtask`           | 27              |

Unknown tags raise `ValueError` at the processor layer with the full list of supported tags.
