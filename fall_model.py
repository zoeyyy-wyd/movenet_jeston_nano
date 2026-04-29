"""
fall_model.py
=============
Fall-detection GRU model + CSV-based dataset.

Contains:
    - FallDetectionGRU      : pure GRU classifier (input 51, output 2)
    - FallKeypointCSVDataset: load keypoints from a CSV, slice into windows

Why a pure GRU (no attention):
    - A fall is a very characteristic motion pattern; the final GRU hidden
      state already summarizes the sequence well.
    - Fewer parameters -> faster on Nano, less overfit risk on small datasets.

CSV format expected:
    Each row = one frame from one video. Columns:
        video_id   : str/int, identifies which video the frame belongs to
                     (the sliding window will NOT cross videos)
        frame_idx  : int,   frame number within the video (used for ordering)
        label      : int,   0 = normal, 1 = fall (same for every frame
                     of a given video)
        x0, y0, s0 : keypoint 0 (x, y, score)
        x1, y1, s1 : keypoint 1
        ...
        x16, y16, s16

    => 3 metadata columns + 17 * 3 = 51 keypoint columns = 54 total.

Example:
    video_id,frame_idx,label,x0,y0,s0,x1,y1,s1,...,x16,y16,s16
    fall_001,0,1,320.5,100.2,0.95,310.1,90.8,0.93,...
    fall_001,1,1,321.0,102.5,0.94,...
    fall_001,2,1,...
    ...
    normal_001,0,0,...
    ...

Helper `dataframe_to_keypoint_dict` is provided so that you can convert
NPZ-style data into the same in-memory format if you start from npz later.
"""

import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from features import KeypointFeatureExtractor, FEATURE_DIM


# =====================================================================
# Model: pure GRU
# =====================================================================
class FallDetectionGRU(nn.Module):
    """
    Pure GRU fall-detection classifier.

    Input  : (B, T, 51)
    Output : (B, 2)   logits for [normal, fall]

    Usage:
        model = FallDetectionGRU(input_dim=51, hidden_dim=64, num_layers=2)
        logits = model(seq)
        loss = F.cross_entropy(logits, labels)

        # Inference
        with torch.no_grad():
            probs = torch.softmax(model(seq), 1)
            fall_prob = probs[:, 1]
    """

    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2,
                 num_classes=2, dropout=0.3):
        super(FallDetectionGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # LayerNorm on input stabilizes features of different scales.
        self.input_norm = nn.LayerNorm(input_dim)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Use the last layer's final hidden state for classification.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        """
        x : (B, T, input_dim)
        return logits : (B, num_classes)
        """
        x = self.input_norm(x)
        out, h_n = self.gru(x)
        # h_n : (num_layers, B, hidden_dim) -> take top layer
        last = h_n[-1]
        return self.classifier(last)

    # ---------- Save / load ----------
    def save(self, path, extra=None):
        """Save weights + config so load_from() can reconstruct exactly."""
        ckpt = {
            'state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'num_classes': self.num_classes,
            },
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    @classmethod
    def load_from(cls, path, map_location='cpu'):
        """Returns (model, ckpt_dict). ckpt_dict carries seq_len, val_acc, etc."""
        ckpt = torch.load(path, map_location=map_location)
        cfg = ckpt['config']
        model = cls(**cfg)
        model.load_state_dict(ckpt['state_dict'])
        return model, ckpt

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# =====================================================================
# CSV loading helpers
# =====================================================================
def _expected_keypoint_columns():
    """Return the 51 expected keypoint column names: x0,y0,s0,x1,y1,s1,..."""
    cols = []
    for k in range(17):
        cols.extend(['x{}'.format(k), 'y{}'.format(k), 's{}'.format(k)])
    return cols


def load_csv_to_videos(csv_path, verbose=True):
    """
    Read a CSV and return {video_id: (keypoints_array, label)}.

    keypoints_array : np.ndarray (T, 17, 3) float32, sorted by frame_idx
    label           : int

    Validates required columns and label consistency within each video.
    """
    df = pd.read_csv(csv_path)

    required = {'video_id', 'frame_idx', 'label'}
    kp_cols = _expected_keypoint_columns()
    missing = required - set(df.columns)
    missing |= set(kp_cols) - set(df.columns)
    if missing:
        raise ValueError(
            "CSV is missing columns: {}\n"
            "Expected columns: video_id, frame_idx, label, "
            "x0..x16, y0..y16, s0..s16".format(sorted(missing))
        )

    videos = {}
    for vid, group in df.groupby('video_id', sort=False):
        # Sort by frame_idx so the sequence is in chronological order.
        group = group.sort_values('frame_idx')

        labels = group['label'].unique()
        if len(labels) != 1:
            raise ValueError(
                "video_id={} has inconsistent labels: {}".format(vid, labels))
        label = int(labels[0])

        # (T, 51) -> (T, 17, 3)
        kp_flat = group[kp_cols].to_numpy(dtype=np.float32)
        T = kp_flat.shape[0]
        kp_arr = kp_flat.reshape(T, 17, 3)

        videos[vid] = (kp_arr, label)

    if verbose:
        n_normal = sum(1 for _, lbl in videos.values() if lbl == 0)
        n_fall = sum(1 for _, lbl in videos.values() if lbl == 1)
        print("[INFO] Loaded {} videos from {} ({} normal, {} fall)".format(
            len(videos), csv_path, n_normal, n_fall))
    return videos


# =====================================================================
# CSV-based Dataset
# =====================================================================
class FallKeypointCSVDataset(Dataset):
    """
    Fall-detection dataset loaded from a CSV file.

    Each row of the CSV is one frame; rows for the same video share a
    `video_id`. The sliding window is applied per video so that windows
    never cross video boundaries.

    Usage:
        ds = FallKeypointCSVDataset("data/fall_keypoints.csv",
                                    seq_len=30, stride=5)
        loader = DataLoader(ds, batch_size=16, shuffle=True)
        for seq, label in loader:
            # seq:   (B, 30, 51) float32
            # label: (B,) int64
            ...
    """

    def __init__(self, csv_path, seq_len=30, stride=5, verbose=True):
        """
        Parameters
        ----------
        csv_path : path to the CSV file
        seq_len  : window size in frames (30 ≈ 1 s at 30 fps)
        stride   : window step. Smaller = more samples but more correlated.
        """
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.stride = stride
        self.feature_dim = FEATURE_DIM
        self.samples = []   # list of (feature_seq (T, 51), label int)

        self._load(verbose)

    def _load(self, verbose):
        videos = load_csv_to_videos(self.csv_path, verbose=verbose)
        ext = KeypointFeatureExtractor()

        per_class_videos = Counter()
        per_class_samples = Counter()

        for vid, (kp_arr, label) in videos.items():
            T_total = kp_arr.shape[0]
            if T_total < self.seq_len:
                if verbose:
                    print("[WARN] {} too short ({} < seq_len={}), skipped".format(
                        vid, T_total, self.seq_len))
                continue

            # Extract features for the whole video first
            # (stateful: speed depends on the previous frame).
            feat_seq = ext.extract_sequence(kp_arr)   # (T, 51)

            T = feat_seq.shape[0]
            n_added = 0
            for s in range(0, T - self.seq_len + 1, self.stride):
                self.samples.append((feat_seq[s:s + self.seq_len], label))
                n_added += 1

            per_class_videos[label] += 1
            per_class_samples[label] += n_added

        if verbose:
            for label in sorted(per_class_videos):
                cls_name = 'fall' if label == 1 else 'normal'
                print("[INFO] {:6s} (label={}): {} videos -> {} samples".format(
                    cls_name, label,
                    per_class_videos[label],
                    per_class_samples[label]))
            print("[INFO] Total samples: {}".format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return (
            torch.from_numpy(seq).float(),
            torch.tensor(label, dtype=torch.long),
        )

    def class_distribution(self):
        """Return {label: number_of_samples}."""
        return dict(Counter(s[1] for s in self.samples))
