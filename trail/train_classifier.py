"""TRAIL MLP Classifier Training Script (Memory-Efficient).

Streams embedding files from disk in batches to avoid OOM.
Processes embeddings in two passes:
  Pass 1: Scan all files to build (rid -> max_generated_len) mapping
  Pass 2: Stream embeddings, compute remaining_len, discretize, and train

Usage:
    python train_classifier.py \
        --embedding-dir /tmp/trail_embeddings \
        --output-path ~/sglang-trail/trail/trail_classifier.pt \
        --num-bins 32 --max-output-len 512
"""

import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def scan_request_lengths(embedding_dir: str):
    """Pass 1: Find max generated_len per request ID."""
    pt_files = sorted(glob.glob(os.path.join(embedding_dir, "trail_embeddings_*.pt")))
    print(f"Found {len(pt_files)} embedding files")

    rid_max_len = {}
    total_entries = 0
    for i, fpath in enumerate(pt_files):
        entries = torch.load(fpath, weights_only=False, map_location="cpu")
        for entry in entries:
            rid = entry["rid"]
            gen_len = entry["generated_len"]
            if rid not in rid_max_len or gen_len > rid_max_len[rid]:
                rid_max_len[rid] = gen_len
            total_entries += 1
        del entries
        if (i + 1) % 200 == 0:
            print(f"  Scanned {i+1}/{len(pt_files)} files, {total_entries} entries, {len(rid_max_len)} requests")

    print(f"Total: {total_entries} entries from {len(rid_max_len)} unique requests")
    return rid_max_len, pt_files


def stream_and_sample(pt_files, rid_max_len, max_samples=200000, seed=42):
    """Pass 2: Stream files, compute remaining_len, reservoir sample if needed."""
    rng = random.Random(seed)
    X_list = []
    Y_list = []
    seen = 0

    for i, fpath in enumerate(pt_files):
        entries = torch.load(fpath, weights_only=False, map_location="cpu")
        for entry in entries:
            rid = entry["rid"]
            remaining = rid_max_len[rid] - entry["generated_len"]
            emb = entry["embedding"].float()

            seen += 1
            if len(X_list) < max_samples:
                X_list.append(emb)
                Y_list.append(remaining)
            else:
                # Reservoir sampling
                j = rng.randint(0, seen - 1)
                if j < max_samples:
                    X_list[j] = emb
                    Y_list[j] = remaining
        del entries

        if (i + 1) % 200 == 0:
            print(f"  Streamed {i+1}/{len(pt_files)} files, sampled {len(X_list)}/{seen}")

    print(f"Sampled {len(X_list)} from {seen} total entries")
    X = torch.stack(X_list)
    Y = np.array(Y_list)
    return X, Y


def build_model(input_dim: int, num_bins: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, num_bins),
    )


def train(X_train, Y_train, X_test, Y_test, num_bins, num_epochs=30, batch_size=32, lr=0.01, device="cuda"):
    input_dim = X_train.shape[1]
    model = build_model(input_dim, num_bins).to(device)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=0.0)

    perm = torch.randperm(X_train.size(0))
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    best_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch_start in range(0, len(X_train), batch_size):
            X_batch = X_train[batch_start:batch_start + batch_size]
            Y_batch = Y_train[batch_start:batch_start + batch_size]
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_preds = test_logits.argmax(dim=1)
            acc = (test_preds == Y_test).float().mean().item()

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest test accuracy: {best_acc:.4f}")
    return model, best_state


def main():
    parser = argparse.ArgumentParser(description="TRAIL Classifier Training")
    parser.add_argument("--embedding-dir", type=str, default="/tmp/trail_embeddings")
    parser.add_argument("--output-path", type=str, default="trail_classifier.pt")
    parser.add_argument("--num-bins", type=int, default=32)
    parser.add_argument("--max-output-len", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=200000,
                        help="Max training+test samples (reservoir sampled if more)")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Pass 1: Scanning request lengths...")
    rid_max_len, pt_files = scan_request_lengths(args.embedding_dir)

    print(f"\nPass 2: Streaming embeddings (max {args.max_samples} samples)...")
    X, Y_raw = stream_and_sample(pt_files, rid_max_len, args.max_samples, args.seed)

    # Discretize
    bins = np.linspace(0, args.max_output_len, args.num_bins + 1)
    Y_binned = np.digitize(Y_raw, bins) - 1
    Y_binned = np.clip(Y_binned, 0, args.num_bins - 1)
    print(f"\nLabel range: [{Y_binned.min()}, {Y_binned.max()}], output len range: [{Y_raw.min()}, {Y_raw.max()}]")

    # Train/test split
    n = len(X)
    n_test = int(n * args.test_size)
    indices = list(range(n))
    random.Random(args.seed).shuffle(indices)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = X[train_idx]
    Y_train = torch.tensor(Y_binned[train_idx], dtype=torch.long)
    X_test = X[test_idx]
    Y_test = torch.tensor(Y_binned[test_idx], dtype=torch.long)

    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Hidden dim: {X_train.shape[1]}")

    model, best_state = train(
        X_train, Y_train, X_test, Y_test,
        num_bins=args.num_bins, num_epochs=args.num_epochs,
        batch_size=args.batch_size, lr=args.lr, device=args.device,
    )

    save_dict = {
        "model_state_dict": best_state,
        "input_dim": X_train.shape[1],
        "hidden_dim": 512,
        "num_bins": args.num_bins,
        "max_output_len": args.max_output_len,
        "bin_edges": bins.tolist(),
    }
    torch.save(save_dict, args.output_path)
    print(f"\nClassifier saved to {args.output_path}")


if __name__ == "__main__":
    main()
