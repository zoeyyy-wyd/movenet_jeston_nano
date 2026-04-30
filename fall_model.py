"""
fall_model.py
=============
Fall-detection GRU model + CSV-based dataset.

Contains:
    - FallDetectionGRU       : pure GRU classifier (input 51, output 2)
    - FallKeypointCSVDataset : load keypoints from a CSV, slice into windows,
                               compute window-level labels by vote.

Why a pure GRU (no attention):
    - A fall is a fairly characteristic motion pattern; the final GRU
      hidden state already summarizes the sequence well.
    - Fewer parameters -> faster on Nano, less overfit risk on small
      datasets.

CSV format expected (matches build_dataset.py's output)
-------------------------------------------------------
    Each row = ONE FRAME from one video. Columns (54 total):
        video_id    : str, identifies which video the frame belongs to
                      (the sliding window will NOT cross videos)
        frame_idx   : int, frame number within the video (0-indexed)
        label       : int, 0 = normal, 1 = fall  *** PER FRAME ***
        nose_x, nose_y, nose_score, ..., right_ankle_x/y/score
        (3 metadata + 17 * 3 = 54)

    Note: the per-frame `label` is 1 only inside the [end, end+post_fall]
    region of fall videos (see build_dataset.py); a single video usually
    contains BOTH 0 and 1 frames, so we cannot collapse it to a single
    video-level label.

Window labeling
---------------
For each sliding window of length `seq_len`, we count how many frames
inside the window have label==1. The window is labeled 1 iff that
count >= `min_pos_frames`.

    min_pos_frames default = 5 (out of seq_len=30, i.e. ~17%)

This catches windows that contain the post-impact moment without
over-weighting brief noise.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from features import KeypointFeatureExtractor, FEATURE_DIM


# COCO-17 names, must match build_dataset.py / movenet_trt.py
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]


# =====================================================================
# Model: pure GRU
# =====================================================================
class FallDetectionGRU(nn.Module):
    """
    Pure GRU fall-detection classifier.

    Input  : (B, T, 51)
    Output : (B, 2)   logits for [normal, fall]
    """

    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2,
                 num_classes=2, dropout=0.3):
        super(FallDetectionGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.input_norm = nn.LayerNorm(input_dim)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

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
        last = h_n[-1]   # (B, hidden_dim)
        return self.classifier(last)

    def save(self, path, extra=None):
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
        ckpt = torch.load(path, map_location=map_location)
        cfg = ckpt['config']
        model = cls(**cfg)
        model.load_state_dict(ckpt['state_dict'])
        return model, ckpt

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# =====================================================================
# CSV loading: per-video (T, 17, 3) keypoints + per-frame labels
# =====================================================================
def _expected_keypoint_columns():
    """The 51 keypoint column names in build_dataset.py order."""
    cols = []
    for name in KEYPOINT_NAMES:
        cols.extend(['{}_x'.format(name),
                     '{}_y'.format(name),
                     '{}_score'.format(name)])
    return cols


def load_csv_to_videos(csv_path, verbose=True):
    """
    Read a per-frame CSV and return {video_id: (kp_arr, label_arr)}.

    kp_arr    : np.ndarray (T, 17, 3) float32, sorted by frame_idx
    label_arr : np.ndarray (T,)        int64,   per-frame labels

    Validates required columns.
    """
    df = pd.read_csv(csv_path)

    required = {'video_id', 'frame_idx', 'label'}
    kp_cols = _expected_keypoint_columns()
    missing = required - set(df.columns)
    missing |= set(kp_cols) - set(df.columns)
    if missing:
        raise ValueError(
            "CSV is missing columns: {}\n"
            "Expected: video_id, frame_idx, label, "
            "nose_x..right_ankle_score (17 keypoints x 3)".format(
                sorted(missing))
        )

    videos = {}
    for vid, group in df.groupby('video_id', sort=False):
        group = group.sort_values('frame_idx')

        kp_flat = group[kp_cols].to_numpy(dtype=np.float32)   # (T, 51)
        T = kp_flat.shape[0]
        kp_arr = kp_flat.reshape(T, 17, 3)

        label_arr = group['label'].to_numpy(dtype=np.int64)   # (T,)
        videos[vid] = (kp_arr, label_arr)

    if verbose:
        n_total = len(videos)
        n_with_fall = sum(1 for _, lbls in videos.values() if (lbls == 1).any())
        n_pure_normal = n_total - n_with_fall
        total_frames = sum(len(lbls) for _, lbls in videos.values())
        total_fall_frames = sum(int((lbls == 1).sum())
                                for _, lbls in videos.values())
        print("[INFO] Loaded {} videos from {}".format(n_total, csv_path))
        print("       videos with any fall frames : {}".format(n_with_fall))
        print("       pure-normal videos          : {}".format(n_pure_normal))
        print("       total frames                : {}".format(total_frames))
        print("       fall frames                 : {} ({:.2f}%)".format(
            total_fall_frames, 100 * total_fall_frames / max(total_frames, 1)))
    return videos


# =====================================================================
# CSV-based Dataset
# =====================================================================
class FallKeypointCSVDataset(Dataset):
    """
    Sliding-window dataset built from a per-frame CSV.

    Pipeline per video:
        1. Sort frames by frame_idx
        2. Run KeypointFeatureExtractor over the whole video to produce
           (T, 51) features.
           The extractor is STATEFUL (speed depends on previous frame),
           so it must be reset and run sequentially per video.
        3. Slice into windows of seq_len with given stride.
        4. For each window, count how many frames have label==1.
           Window label = 1 iff count >= min_pos_frames, else 0.

    The min_pos_frames threshold is the key knob to tune:
        - too low  -> windows that just clip the start of the post-fall
                     region get labeled 1, but they don't really show
                     the "stillness" signal yet -> noisy positives
        - too high -> too few positive samples, training collapses

    With seq_len=30 and post_fall_frames=50, default min_pos_frames=5
    means "at least 5 of 30 frames inside the window are post-impact".
    """

    def __init__(self, csv_path, seq_len=30, stride=5,
                 min_pos_frames=5,
                 video_ids=None,
                 verbose=True):
        """
        Parameters
        ----------
        csv_path       : path to the per-frame CSV
        seq_len        : window size in frames (30 ≈ 1.2 s @ 25 fps)
        stride         : window step
        min_pos_frames : window labeling threshold (see class docstring)
        video_ids      : optional list of video_ids to keep (for train/val
                         split by video). If None, all videos are used.
        verbose        : print loading stats
        """
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.stride = stride
        self.min_pos_frames = min_pos_frames
        self.feature_dim = FEATURE_DIM
        self.samples = []   # list of (feature_seq (seq_len, 51), label)

        self._load(video_ids, verbose)

    def _load(self, video_ids, verbose):
        videos = load_csv_to_videos(self.csv_path, verbose=verbose)

        if video_ids is not None:
            wanted = set(video_ids)
            videos = {k: v for k, v in videos.items() if k in wanted}
            if verbose:
                print("[INFO] Filtered to {} videos".format(len(videos)))

        ext = KeypointFeatureExtractor()

        n_videos_used = 0
        n_videos_skipped_short = 0
        per_class_samples = Counter()

        for vid, (kp_arr, label_arr) in videos.items():
            T_total = kp_arr.shape[0]
            if T_total < self.seq_len:
                n_videos_skipped_short += 1
                continue

            feat_seq = ext.extract_sequence(kp_arr)   # (T, 51)
            T = feat_seq.shape[0]

            for s in range(0, T - self.seq_len + 1, self.stride):
                e = s + self.seq_len
                window_labels = label_arr[s:e]
                pos_count = int((window_labels == 1).sum())
                window_label = 1 if pos_count >= self.min_pos_frames else 0

                self.samples.append(
                    (feat_seq[s:e].astype(np.float32), window_label))
                per_class_samples[window_label] += 1

            n_videos_used += 1

        if verbose:
            print("[INFO] Windowed dataset built")
            print("       seq_len={}, stride={}, min_pos_frames={}".format(
                self.seq_len, self.stride, self.min_pos_frames))
            print("       videos used    : {}".format(n_videos_used))
            print("       videos skipped : {} (too short, < seq_len)".format(
                n_videos_skipped_short))
            print("       windows total  : {}".format(len(self.samples)))
            for lbl in sorted(per_class_samples):
                cls_name = 'fall  ' if lbl == 1 else 'normal'
                print("       class {} ({}) : {} windows".format(
                    lbl, cls_name, per_class_samples[lbl]))
            if 0 in per_class_samples and 1 in per_class_samples:
                ratio = per_class_samples[0] / max(per_class_samples[1], 1)
                print("       imbalance (n:p): {:.1f}:1".format(ratio))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return (
            torch.from_numpy(seq).float(),
            torch.tensor(label, dtype=torch.long),
        )

    def class_distribution(self):
        return dict(Counter(s[1] for s in self.samples))


# =====================================================================
# Helper: train/val video-level split (NO leakage of frames across splits)
# =====================================================================
def split_videos_by_id(csv_path, val_frac=0.2, seed=42):
    """
    Returns (train_video_ids, val_video_ids).

    Splits at the VIDEO level: all frames from one video go to either
    train or val, never both. Stratifies by 'video has any fall frame'
    so both splits get a similar fall-video proportion.
    """
    df = pd.read_csv(csv_path, usecols=['video_id', 'label'])
    per_video = df.groupby('video_id')['label'].max().reset_index()
    per_video.columns = ['video_id', 'has_fall']

    rng = np.random.default_rng(seed)
    train_ids, val_ids = [], []
    for has_fall, sub in per_video.groupby('has_fall'):
        ids = sub['video_id'].tolist()
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_frac)))
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    return sorted(train_ids), sorted(val_ids)
