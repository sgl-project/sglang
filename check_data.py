#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_hidden_states.py

Inspect and materialise nested hidden-state columns inside a Parquet file
(e.g. `hidden_state`, `target_hidden_states`) into a regular NumPy tensor.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- basic helpers ------------------------------------------------- #
def read_df(path: Path, column: str):
    """Load a single Parquet column into a pandas DataFrame."""
    return pd.read_parquet(path, engine="pyarrow",
                           use_threads=True, columns=[column])


# ---------- conversion utilities ----------------------------------------- #
def layer_to_matrix(layer):
    """
    Convert one layer (various possible Python / NumPy container types)
    to a 2-D float16 ndarray of shape (seq_len, hidden_dim).
    Return None if layer is empty.
    """
    # Empty list / ndarray → None
    if layer is None or (hasattr(layer, "__len__") and len(layer) == 0):
        return None

    # ─ pure Python list ─────────────────────────────────────────────────── #
    if isinstance(layer, list):
        return np.asarray(layer, dtype=np.float16)

    # ─ NumPy arrays ─────────────────────────────────────────────────────── #
    if isinstance(layer, np.ndarray):
        # ndim == 1  → only one token vector
        if layer.ndim == 1:
            return layer[np.newaxis, :].astype(np.float16)

        # ndim == 2  → already (seq_len, hidden_dim)
        if layer.ndim == 2:
            return layer.astype(np.float16)

        # object-dtype 1-D ndarray whose elements are token vectors
        if layer.ndim == 1 and layer.dtype == object:
            return np.stack(layer.astype(np.float16))

    raise TypeError(
        f"Unsupported layer format: {type(layer)}, ndim={getattr(layer, 'ndim', None)}"
    )


def sample_to_tensor(layers, pad_val=0.0):
    """
    Combine *all* layers of one sample into a 3-D tensor:
    (n_layer, max_seq_len, max_hidden_dim).  Right-pads with `pad_val`.
    Returns None if no layer contains data.
    """
    matrices = [m for layer in layers if (m := layer_to_matrix(layer)) is not None]
    if not matrices:
        return None

    n_layer = len(matrices)
    max_seq = max(m.shape[0] for m in matrices)
    max_dim = max(m.shape[1] for m in matrices)

    out = np.full((n_layer, max_seq, max_dim), pad_val, dtype=np.float16)
    for i, mat in enumerate(matrices):
        s, h = mat.shape
        out[i, :s, :h] = mat
    return out


# ---------- CLI ----------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Inspect hidden-state Parquet column and convert to NumPy tensor"
    )
    parser.add_argument("path", help="Path to .parquet file")
    parser.add_argument(
        "--column", default="target_hidden_states",
        help="Name of the nested column to inspect (default: target_hidden_states)"
    )
    parser.add_argument(
        "--idx", type=int, default=0,
        help="Row index to inspect (default: 0)"
    )
    args = parser.parse_args()

    path = Path(args.path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    df = read_df(path, args.column)
    if args.idx >= len(df):
        raise IndexError(f"DataFrame has {len(df)} rows, but idx={args.idx} requested")

    layers = df.loc[args.idx, args.column]
    tensor = sample_to_tensor(layers)

    if tensor is None:
        print(f"Row {args.idx} in column '{args.column}' is empty.")
        return

    n_layer, seq_len, hidden_dim = tensor.shape
    print(f"Row {args.idx} → tensor shape: (layers={n_layer}, "
          f"seq_len={seq_len}, hidden_dim={hidden_dim}), dtype={tensor.dtype}")

    # show a tiny slice for sanity check
    print("tensor[0, :2, :8] =\n", tensor[0, :2, :8])


if __name__ == "__main__":
    main()
