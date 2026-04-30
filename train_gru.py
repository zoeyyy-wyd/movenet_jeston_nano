"""
train_gru.py
============
Train the FallDetectionGRU on the per-frame CSV produced by
build_dataset.py.

Usage:
    cd ~/movenet_jeston_nano
    python train_gru.py \\
        --csv datasets/le2i_keypoints.csv \\
        --out models/fall_gru.pth

    # quick smoke test
    python train_gru.py --csv datasets/le2i_keypoints.csv --epochs 2

    # tweak window labeling
    python train_gru.py --csv ... --min-pos-frames 8 --stride 3

What it does
------------
1. Reads the CSV, splits videos into train/val (video-level, stratified by
   "has any fall frame" so both splits get fall videos).
2. Builds two FallKeypointCSVDataset instances, one per split.
3. Computes a class-imbalance-aware loss weight from the train set
   (to handle the typical ~10:1 normal:fall ratio).
4. Trains the GRU with Adam + ReduceLROnPlateau on val_f1.
5. Saves the best checkpoint by val F1 (not accuracy -- accuracy is
   misleading on imbalanced data).
6. Prints train/val loss + accuracy + precision + recall + F1 each epoch.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from fall_model import (
    FallDetectionGRU,
    FallKeypointCSVDataset,
    split_videos_by_id,
)


# =====================================================================
# Metrics
# =====================================================================
def compute_metrics(y_true, y_pred):
    """
    Binary metrics. y_true / y_pred are 1-D int arrays.

    Returns dict with: acc, precision, recall, f1, tp, fp, fn, tn.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    n = max(tp + fp + fn + tn, 1)
    acc = (tp + tn) / n
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


# =====================================================================
# Train / eval loops
# =====================================================================
def run_epoch(model, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_n = 0
    all_true, all_pred = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for seq, label in loader:
            seq = seq.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logits = model(seq)
            loss = criterion(logits, label)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            bs = label.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            all_true.append(label.detach().cpu().numpy())
            all_pred.append(pred)

    avg_loss = total_loss / max(total_n, 1)
    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    metrics = compute_metrics(y_true, y_pred)
    metrics['loss'] = avg_loss
    return metrics


# =====================================================================
# Main
# =====================================================================
def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    p.add_argument('--csv', required=True,
                   help='per-frame CSV from build_dataset.py')
    p.add_argument('--seq-len', type=int, default=30)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--min-pos-frames', type=int, default=5,
                   help='Window labeled 1 iff >= this many fall frames inside.')
    p.add_argument('--val-frac', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    # model
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    # train
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=5,
                   help='LR scheduler patience (epochs).')
    p.add_argument('--use-sampler', action='store_true',
                   help='Oversample the fall class with WeightedRandomSampler '
                        'instead of (or in addition to) class-weighted loss.')
    # io
    p.add_argument('--out', default='models/fall_gru.pth')
    p.add_argument('--device', default='auto',
                   help='auto / cpu / cuda. auto picks cuda if available.')
    args = p.parse_args()

    # device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print('[INFO] device:', device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------------- video-level split ----------------
    print('\n=== Splitting videos into train / val ===')
    train_ids, val_ids = split_videos_by_id(args.csv, val_frac=args.val_frac,
                                            seed=args.seed)
    print('  train videos: {}'.format(len(train_ids)))
    print('  val   videos: {}'.format(len(val_ids)))

    # ---------------- datasets ----------------
    print('\n=== Building train dataset ===')
    train_ds = FallKeypointCSVDataset(
        args.csv,
        seq_len=args.seq_len, stride=args.stride,
        min_pos_frames=args.min_pos_frames,
        video_ids=train_ids, verbose=True)

    print('\n=== Building val dataset ===')
    val_ds = FallKeypointCSVDataset(
        args.csv,
        seq_len=args.seq_len, stride=args.stride,
        min_pos_frames=args.min_pos_frames,
        video_ids=val_ids, verbose=True)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print('[ERROR] one of the splits is empty. Try --stride smaller, '
              '--seq-len smaller, or --min-pos-frames smaller.')
        sys.exit(1)

    train_dist = train_ds.class_distribution()
    val_dist = val_ds.class_distribution()
    print('\n  train dist:', train_dist)
    print('  val   dist:', val_dist)

    if 1 not in train_dist or train_dist.get(1, 0) == 0:
        print('[ERROR] no positive (fall) windows in train set. '
              'Try --min-pos-frames smaller or --stride smaller.')
        sys.exit(1)
    if 1 not in val_dist or val_dist.get(1, 0) == 0:
        print('[WARN]  no positive windows in val set; val F1 will be 0.')

    # ---------------- imbalance handling ----------------
    n_neg = train_dist.get(0, 0)
    n_pos = train_dist.get(1, 0)
    pos_weight = n_neg / max(n_pos, 1)
    print('\n  class weight  (for CE loss): [1.0, {:.2f}]'.format(pos_weight))

    class_weights = torch.tensor([1.0, pos_weight],
                                 dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if args.use_sampler:
        # build per-sample weights so each minibatch is roughly 50/50
        sample_w = []
        for _, lbl in train_ds.samples:
            sample_w.append(1.0 / max(n_pos, 1) if lbl == 1
                            else 1.0 / max(n_neg, 1))
        sampler = WeightedRandomSampler(sample_w, num_samples=len(train_ds),
                                        replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=2,
                                  pin_memory=(device == 'cuda'))
        print('  using WeightedRandomSampler for train loader')
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=2,
                                  pin_memory=(device == 'cuda'))

    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2,
                            pin_memory=(device == 'cuda'))

    # ---------------- model ----------------
    model = FallDetectionGRU(
        input_dim=train_ds.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print('\n  model params: {:,}'.format(model.num_parameters()))

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=args.patience, min_lr=1e-6)

    # ---------------- training loop ----------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.',
                exist_ok=True)
    best_f1 = -1.0
    best_epoch = -1

    print('\n=== Training ===')
    print('{:>3s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} '
          '{:>5s} {:>5s} {:>5s}'.format(
              'ep', 'tr_loss', 'tr_f1', 'va_loss', 'va_acc',
              'va_prec', 'va_rec', 'tp', 'fp', 'fn'))

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, criterion,
                       device, train=True)
        va = run_epoch(model, val_loader, optimizer, criterion,
                       device, train=False)

        scheduler.step(va['f1'])
        elapsed = time.time() - t0

        print('{:3d} {:8.4f} {:8.3f} {:8.4f} {:8.3f} {:8.3f} {:8.3f} '
              '{:5d} {:5d} {:5d}  ({:.1f}s)'.format(
                  ep, tr['loss'], tr['f1'],
                  va['loss'], va['acc'], va['precision'], va['recall'],
                  va['tp'], va['fp'], va['fn'], elapsed))

        if va['f1'] > best_f1:
            best_f1 = va['f1']
            best_epoch = ep
            model.save(args.out, extra={
                'epoch': ep,
                'val_metrics': va,
                'train_metrics': tr,
                'args': vars(args),
                'seq_len': args.seq_len,
            })

    print('\n=== Done ===')
    print('  best epoch    : {}'.format(best_epoch))
    print('  best val F1   : {:.3f}'.format(best_f1))
    print('  saved to      : {}'.format(args.out))


if __name__ == '__main__':
    main()
