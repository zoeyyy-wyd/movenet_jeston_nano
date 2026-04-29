"""
features.py
===========
Convert (17, 3) keypoints into a 51-dim feature vector for the GRU.

Feature layout (51 dims):
    - normalized coords  : 17 * 2 = 34
    - inter-frame speed  : 17
    -----------------------------
    total                : 51

Why 51 and not 76:
    - normalized coords + speed already capture the two core fall signals:
        * sudden pose change   -> jump in normalized coords
        * fast motion          -> large speed values
    - Smaller input dim => smaller GRU => faster on Nano, less overfit risk.
    - The GRU can learn angles / ratios on its own from raw coordinates.

Design notes:
    - Normalize by torso scale, not absolute pixels, so the same pose at
      different distances/sizes yields the same features.
    - Speed depends on the previous frame, so the extractor is *stateful*.
      Call reset() before processing a new video.
    - We do NOT zero out low-confidence keypoints. Empirically that
      introduces zero-spikes that hurt the GRU.
"""

import numpy as np


FEATURE_DIM = 51


class KeypointFeatureExtractor(object):
    """
    Convert (17, 3) keypoints frame-by-frame into a (51,) feature vector.

    Usage:
        ext = KeypointFeatureExtractor()
        ext.reset()                    # call before each new video
        for kpts in keypoint_seq:      # kpts: (17, 3)
            feat = ext.extract(kpts)   # feat: (51,)

    Or process a whole sequence at once:
        feats = ext.extract_sequence(keypoint_seq)   # (T,17,3) -> (T,51)
    """

    def __init__(self):
        self.prev_coords = None
        self.ref_center = None
        self.ref_scale = None

    def reset(self):
        """Call before processing a new video."""
        self.prev_coords = None
        self.ref_center = None
        self.ref_scale = None

    def _compute_reference(self, coords, scores):
        """
        Initialize or update the normalization reference (center, scale).

        Strategy:
            - First call: estimate from current frame's valid keypoints.
            - Subsequent: exponential moving average to smooth jitter.
            - All-invalid frame: keep previous reference.
        """
        valid = scores > 0.2
        if not np.any(valid):
            if self.ref_center is None:
                self.ref_center = np.array([0.0, 0.0], dtype=np.float32)
                self.ref_scale = 1.0
            return

        cx = coords[valid, 0].mean()
        cy = coords[valid, 1].mean()
        scale = max(np.ptp(coords[valid, 0]),
                    np.ptp(coords[valid, 1]),
                    1.0)

        if self.ref_center is None:
            self.ref_center = np.array([cx, cy], dtype=np.float32)
            self.ref_scale = float(scale)
        else:
            alpha = 0.1
            self.ref_center[0] = (1 - alpha) * self.ref_center[0] + alpha * cx
            self.ref_center[1] = (1 - alpha) * self.ref_center[1] + alpha * cy
            self.ref_scale = (1 - alpha) * self.ref_scale + alpha * float(scale)

    def extract(self, keypoints):
        """
        Parameters
        ----------
        keypoints : (17, 3) float32

        Returns
        -------
        features : (51,) float32
                   [0:34]  normalized coords (x0, y0, x1, y1, ..., x16, y16)
                   [34:51] inter-frame speed (in normalized space)
        """
        kpts = np.asarray(keypoints, dtype=np.float32)
        coords = kpts[:, :2]
        scores = kpts[:, 2]

        # Update normalization reference
        self._compute_reference(coords, scores)
        center = self.ref_center
        scale = max(self.ref_scale, 1.0)

        # 1. Normalized coords (34)
        norm_coords = (coords - center) / scale     # (17, 2)
        coord_feat = norm_coords.flatten()          # (34,)

        # 2. Inter-frame speed (17)
        # Use normalized coords so speed is invariant to camera distance.
        if self.prev_coords is not None:
            prev_norm = (self.prev_coords - center) / scale
            speed = np.linalg.norm(norm_coords - prev_norm, axis=1)
        else:
            speed = np.zeros(17, dtype=np.float32)
        speed_feat = speed.astype(np.float32)

        self.prev_coords = coords.copy()

        feat = np.concatenate([coord_feat, speed_feat]).astype(np.float32)
        assert feat.shape == (FEATURE_DIM,), \
            "feature dim mismatch: got {}, expected {}".format(feat.shape, FEATURE_DIM)
        return feat

    def extract_sequence(self, keypoints_seq):
        """
        Process a whole keypoint sequence. Calls reset() automatically.

        Parameters
        ----------
        keypoints_seq : (T, 17, 3) ndarray

        Returns
        -------
        features : (T, 51) ndarray
        """
        self.reset()
        T = keypoints_seq.shape[0]
        out = np.zeros((T, FEATURE_DIM), dtype=np.float32)
        for t in range(T):
            out[t] = self.extract(keypoints_seq[t])
        return out
