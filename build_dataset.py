"""
build_dataset.py
================
Build a fall-detection training CSV from the Le2i Fall Dataset.

Uses fire717's PyTorch MoveNet model + .pth weights for keypoint
extraction (no ONNX, no TensorRT).

How to run
----------
This script must be run from the `movenet/` directory because
fire717's code does `from lib import ...`. The bash workflow is:

    cd ~/movenet_jeston_nano/movenet
    python ../build_dataset.py \\
        --data-root ../data_Le2i \\
        --weights output/movenet.pth \\
        --out-dir ../datasets

The script also accepts being called from the project root:

    cd ~/movenet_jeston_nano
    python build_dataset.py        # auto-cd's into movenet/

Output CSV format
-----------------
ONE ROW = ONE FRAME. Train-time code slices into sliding windows.

Columns (54 total):
    video_id       : str, identifies which video
    frame_idx      : int, frame number within the video (0-indexed)
    label          : int, 0 = normal, 1 = fall (FRAME-LEVEL)
    nose_x, nose_y, nose_score, ...                  (17 keypoints x 3)
    ..., right_ankle_x, right_ankle_y, right_ankle_score

Preprocessing (CRITICAL)
------------------------
Matches fire717's TensorDatasetTest exactly:
    1. cv2.imread (BGR) -> we receive BGR from imageio's BGR conversion
    2. cv2.cvtColor BGR -> RGB
    3. cv2.resize directly to 192x192 (NO letterbox; non-uniform scaling)
    4. astype(float32)  -- DO NOT divide by 255: the model's backbone
       does `x = x/127.5 - 1` internally, expecting input in [0, 255].

Doing letterbox or /255 produces broken keypoints (heatmap_max ~0.3
instead of ~0.8) because the model never saw padded inputs or [0,1]
inputs during training.

Frame-level label rules
-----------------------
For each Le2i annotation (line 1 = start frame, line 2 = end frame):
    - non-fall video (start=0, end=0):           every frame -> 0
    - fall video, frame < end:                   label = 0  (before impact)
    - fall video, frame in [end, end + pf]:      label = 1  (just landed)
    - fall video, frame > end + pf:              label = 0  (lying still long enough)

  where `pf` = post_fall_frames (default 50 = 2 seconds at 25 fps).

Stage A (MoveNet inference) is independent of post_fall_frames, so you
can re-run with --skip-extract --post-fall-frames N to try different
values quickly.

Le2i quirks handled
-------------------
- nested directory: Coffee_room_01/Coffee_room_01/Videos
- inconsistent annotation folder name: Annotation_files vs Annotations_files
- malformed annotation files where line 1/2 are bbox data instead of frame numbers
- Windows CRLF line endings
- rawvideo .avi files that confuse OpenCV's FFmpeg (we use imageio fallback)
"""

import argparse
import csv
import glob
import os
import re
import sys
import time
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------
# COCO-17 keypoint order (matches movenet/movenet_trt.py)
# ---------------------------------------------------------------------
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

LE2I_SCENES = ['Coffee_room_01', 'Coffee_room_02',
               'Home_01', 'Home_02',
               'Lecture_room', 'Office']


# =====================================================================
# Annotation parsing
# =====================================================================
def parse_annotation(ann_path):
    """
    Parse a Le2i annotation file.

    Returns (start, end, is_malformed) where:
        start, end   : int frame numbers, (0, 0) means no fall
        is_malformed : True if line 1/2 weren't simple integers (some
                       Le2i files have bbox data on those lines).
                       Treated as no-fall in that case.
    """
    with open(ann_path, 'r') as f:
        lines = f.read().splitlines()    # handles \r\n
    if len(lines) < 2:
        return 0, 0, True
    line1 = lines[0].strip()
    line2 = lines[1].strip()
    if ',' in line1 or ',' in line2:
        return 0, 0, True
    try:
        return int(line1), int(line2), False
    except ValueError:
        return 0, 0, True


def find_annotation_dir(scene_inner_dir):
    """Le2i is inconsistent: 'Annotation_files' vs 'Annotations_files'."""
    for name in ('Annotation_files', 'Annotations_files'):
        cand = os.path.join(scene_inner_dir, name)
        if os.path.isdir(cand):
            return cand
    return None


_VIDEO_NUM_RE = re.compile(r'video\s*\((\d+)\)', re.IGNORECASE)


def video_number(filename):
    m = _VIDEO_NUM_RE.search(filename)
    return int(m.group(1)) if m else None


# =====================================================================
# Video reading: imageio first (handles Le2i rawvideo), cv2 fallback
# =====================================================================
def _iter_frames_imageio(video_path):
    import imageio.v3 as iio
    for frame_rgb in iio.imiter(video_path):
        # imageio yields RGB; convert to BGR to match the rest of the
        # pipeline (which assumes OpenCV-style BGR input).
        yield frame_rgb[:, :, ::-1].copy()


def _iter_frames_cv2(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def iter_video_frames(video_path):
    """Yield BGR frames; imageio preferred, OpenCV fallback."""
    try:
        import imageio  # noqa: F401
        try:
            yield from _iter_frames_imageio(video_path)
            return
        except Exception as e:
            print("    [info] imageio failed ({}), falling back to OpenCV".format(e))
    except ImportError:
        print("    [info] imageio not installed, using OpenCV "
              "(may fail on Le2i rawvideo)")
    yield from _iter_frames_cv2(video_path)


# =====================================================================
# MoveNet PyTorch inference (fire717's model + .pth weights)
# =====================================================================
class MoveNetTorch(object):
    """
    Inference wrapper around fire717's MoveNet PyTorch model.

    Pre/postprocess matches fire717's TensorDatasetTest exactly:
        - RGB
        - cv2.resize directly to img_size x img_size (no letterbox)
        - float32 in [0, 255] range (NO /255 -- the backbone does
          `x/127.5 - 1` internally)

    IMPORTANT: this class does `from lib import ...` and `from config
    import cfg`, which only resolve when sys.path includes the
    `movenet/` directory. The CLI handles this by chdir'ing before
    instantiating.
    """

    def __init__(self, weights_path, img_size=192, device=None):
        import cv2
        self._cv2 = cv2

        import torch
        self._torch = torch

        # We import fire717's MoveNet class but skip their Task wrapper,
        # because Task.__init__ hardcodes `self.device = torch.device('cuda')`
        # and crashes on CPU-only machines.
        from lib import init, MoveNet
        from config import cfg

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.img_size = img_size

        init(cfg)

        # Build model in TRAIN mode -- gives us the 4 output heads we need
        # (heatmap, center, regs, offsets). 'test' mode is unimplemented.
        model = MoveNet(
            num_classes=cfg['num_classes'],
            width_mult=cfg['width_mult'],
            mode='train',
        )

        # Load weights manually (bypass fire717's Task.modelLoad which
        # hardcodes cuda and may not match our checkpoint format).
        ckpt = torch.load(weights_path, map_location=device)
        sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

        # Strip common DataParallel / lightning prefixes
        cleaned = {}
        for k, v in sd.items():
            nk = k
            for prefix in ('module.', 'model.'):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v

        # Use strict=True now that we know the keys match -- we want to
        # fail loudly if they ever drift.
        model.load_state_dict(cleaned, strict=True)
        print("[OK] Weights loaded cleanly: {}".format(weights_path))

        model = model.to(device)
        model.eval()
        self.model = model

        print("[OK] MoveNet PyTorch loaded")
        print("     device : {}".format(device))
        print("     input  : (1, 3, {0}, {0})".format(img_size))

    def _preprocess(self, frame_bgr):
        """
        BGR uint8 (H, W, 3) -> torch tensor (1, 3, 192, 192) float32 on device.

        Matches fire717's TensorDatasetTest exactly:
            BGR -> RGB -> resize(192, 192) -> float32 (no /255).

        Also returns (scale_x, scale_y) so the caller can map predicted
        coords from the 192-space back to the original frame.
        """
        cv2 = self._cv2
        torch = self._torch

        h, w = frame_bgr.shape[:2]

        # 1. BGR -> RGB
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 2. resize directly (NON-uniform; no letterbox / no padding)
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)

        # 3. float32, [0, 255] range. Backbone does x/127.5 - 1 internally.
        blob = img.astype(np.float32)
        blob = np.transpose(blob, (2, 0, 1))[None]      # (1,3,H,W)

        # Reverse-mapping factors. Because we resized non-uniformly, x
        # and y need separate scale factors.
        scale_x = float(w) / self.img_size
        scale_y = float(h) / self.img_size

        return torch.from_numpy(blob).to(self.device), scale_x, scale_y

    def _postprocess(self, outs, scale_x, scale_y):
        """
        Decode 4 output heads -> (17, 3) keypoints in original-frame coords.

        outs is a list:
            outs[0]: (1, 17, Hf, Wf)  keypoint heatmaps  (post-sigmoid)
            outs[1]: (1,  1, Hf, Wf)  person center heatmap (post-sigmoid)
            outs[2]: (1, 34, Hf, Wf)  keypoint regression (offset from center)
            outs[3]: (1, 34, Hf, Wf)  per-keypoint sub-pixel offset

        Decoding follows MoveNet's reference flow:
            1. argmax on the center heatmap -> person center (cx, cy)
            2. for each kp, regs gives a vector from center to kp's grid cell
            3. offsets gives sub-pixel refinement at that grid cell
            4. score = heatmap[k, ky, kx] (the kp's heatmap value at the
               regress-derived grid cell)
        """
        heatmap = outs[0].detach().cpu().numpy()
        center  = outs[1].detach().cpu().numpy()
        regs    = outs[2].detach().cpu().numpy()
        offsets = outs[3].detach().cpu().numpy()

        Hf, Wf = heatmap.shape[2], heatmap.shape[3]
        stride = float(self.img_size) / Hf  # how many input-image pixels per grid cell

        cmap = center[0, 0]
        cy, cx = np.unravel_index(np.argmax(cmap), cmap.shape)

        keypoints = np.zeros((17, 3), dtype=np.float32)
        for k in range(17):
            rx = float(regs[0, 2 * k,     cy, cx])
            ry = float(regs[0, 2 * k + 1, cy, cx])
            kx = int(round(np.clip(cx + rx, 0, Wf - 1)))
            ky = int(round(np.clip(cy + ry, 0, Hf - 1)))
            ox = float(offsets[0, 2 * k,     ky, kx])
            oy = float(offsets[0, 2 * k + 1, ky, kx])

            # Coords in the 192x192 input space
            x_in = (kx + ox) * stride
            y_in = (ky + oy) * stride

            # Map back to original frame (non-uniform inverse of the resize)
            keypoints[k, 0] = x_in * scale_x
            keypoints[k, 1] = y_in * scale_y
            keypoints[k, 2] = float(heatmap[0, k, ky, kx])
        return keypoints

    def infer(self, frame_bgr):
        """frame_bgr: BGR ndarray (H, W, 3) uint8 -> (17, 3) float32"""
        torch = self._torch
        tensor, sx, sy = self._preprocess(frame_bgr)
        with torch.no_grad():
            outs = self.model(tensor)
        return self._postprocess(outs, sx, sy)


# =====================================================================
# Stage A: extract keypoints per video, save .npz
# =====================================================================
def stage_a_extract(data_root, raw_dir, weights_path, force=False):
    """
    For each video in each scene:
        1. parse the annotation
        2. run MoveNet on every frame -> (T, 17, 3)
        3. save (keypoints + start + end + malformed flag) to .npz

    Labels are NOT computed here -- they're built in stage B from start/
    end + post_fall_frames, so changing post_fall_frames doesn't require
    re-running MoveNet (the slow part).
    """
    movenet = MoveNetTorch(weights_path)
    os.makedirs(raw_dir, exist_ok=True)

    grand_total_frames = 0
    grand_total_time = 0.0

    for scene in LE2I_SCENES:
        scene_outer = os.path.join(data_root, scene)
        scene_inner = os.path.join(scene_outer, scene)
        if not os.path.isdir(scene_inner):
            print("[WARN] scene dir not found: {}".format(scene_inner))
            continue

        videos_dir = os.path.join(scene_inner, 'Videos')
        ann_dir = find_annotation_dir(scene_inner)
        if not os.path.isdir(videos_dir) or ann_dir is None:
            print("[WARN] missing Videos/ or Annotation_files/ in {}".format(
                scene_inner))
            continue

        out_scene_dir = os.path.join(raw_dir, scene)
        os.makedirs(out_scene_dir, exist_ok=True)

        video_files = sorted(glob.glob(os.path.join(videos_dir, '*.avi')))
        if not video_files:
            print("[WARN] no .avi in {}".format(videos_dir))
            continue

        print("\n=== {} ({} videos) ===".format(scene, len(video_files)))

        for vp in video_files:
            video_name = os.path.splitext(os.path.basename(vp))[0]
            num = video_number(video_name)
            if num is None:
                print("  [SKIP] cannot parse number from: {}".format(video_name))
                continue

            ann_path = os.path.join(ann_dir, "video ({}).txt".format(num))
            if not os.path.isfile(ann_path):
                ann_path = os.path.join(ann_dir, "video({}).txt".format(num))
            if not os.path.isfile(ann_path):
                print("  [SKIP] annotation missing for: {}".format(video_name))
                continue

            out_npz = os.path.join(out_scene_dir, video_name + '.npz')
            if os.path.exists(out_npz) and not force:
                print("  [SKIP cached] {}".format(video_name))
                continue

            start, end, malformed = parse_annotation(ann_path)
            tag = "fall" if (start > 0 and end > 0) else "normal"
            if malformed:
                tag = "normal (malformed annotation)"

            print("  {}  [{}, start={}, end={}] ...".format(
                video_name, tag, start, end), end=' ', flush=True)

            t0 = time.time()
            kpts_list = []
            try:
                for frame in iter_video_frames(vp):
                    kpts_list.append(movenet.infer(frame))
            except Exception as e:
                print("FAIL ({})".format(e))
                continue

            if not kpts_list:
                print("FAIL (no frames decoded)")
                continue

            kp_arr = np.stack(kpts_list, axis=0).astype(np.float32)
            T = kp_arr.shape[0]

            np.savez_compressed(
                out_npz,
                keypoints=kp_arr,
                video_id="{}_{}".format(scene, video_name).replace(' ', '_'),
                scene=scene,
                start=start, end=end,
                malformed=malformed,
                source_video=os.path.basename(vp),
            )

            elapsed = time.time() - t0
            print("T={}, {:.1f}s ({:.1f} FPS), kp_score_max={:.3f}".format(
                T, elapsed, T / max(elapsed, 1e-6),
                float(kp_arr[..., 2].max())))
            grand_total_frames += T
            grand_total_time += elapsed

    print("\n" + "=" * 60)
    print("[Stage A done] {} frames total, {:.1f} s".format(
        grand_total_frames, grand_total_time))
    if grand_total_time > 0:
        print("               avg {:.1f} FPS".format(
            grand_total_frames / grand_total_time))


# =====================================================================
# Stage B: flatten npz files + compute frame labels -> CSV
# =====================================================================
def stage_b_flatten(raw_dir, out_csv, post_fall_frames=50):
    """
    Walk every npz, build per-frame labels, write a flat CSV.

    Labels:
        - non-fall video (start=0, end=0):       all frames -> 0
        - fall video, frame in [end, end+pf]:    label = 1
        - else:                                  label = 0
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            "raw_keypoints dir not found: {}\n"
            "Run stage A first.".format(raw_dir))

    npz_files = []
    for scene in LE2I_SCENES:
        scene_dir = os.path.join(raw_dir, scene)
        if os.path.isdir(scene_dir):
            npz_files.extend(sorted(glob.glob(os.path.join(scene_dir, '*.npz'))))

    if not npz_files:
        raise RuntimeError("No .npz files in {}".format(raw_dir))

    print("\n[Stage B] flattening {} videos -> CSV".format(len(npz_files)))
    print("          post_fall_frames = {} (= {:.1f}s @ 25fps)".format(
        post_fall_frames, post_fall_frames / 25.0))

    columns = ['video_id', 'frame_idx', 'label']
    for name in KEYPOINT_NAMES:
        columns.extend(['{}_x'.format(name), '{}_y'.format(name),
                        '{}_score'.format(name)])
    assert len(columns) == 54

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)

    total_rows = 0
    label_count = Counter()
    video_count = 0

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for npz_path in npz_files:
            data = np.load(npz_path, allow_pickle=True)
            kp = data['keypoints']
            video_id = str(data['video_id'])
            T = kp.shape[0]
            start = int(data['start']) if 'start' in data.files else 0
            end = int(data['end']) if 'end' in data.files else 0
            malformed = bool(data['malformed']) if 'malformed' in data.files else False

            labels = np.zeros(T, dtype=np.int64)
            if start > 0 and end > 0 and not malformed:
                s = max(0, min(end, T - 1))
                e = max(0, min(end + post_fall_frames, T - 1))
                labels[s:e + 1] = 1

            for t in range(T):
                row = [video_id, t, int(labels[t])]
                for k in range(17):
                    row.append(float(kp[t, k, 0]))
                    row.append(float(kp[t, k, 1]))
                    row.append(float(kp[t, k, 2]))
                writer.writerow(row)
                total_rows += 1
                label_count[int(labels[t])] += 1
            video_count += 1

    print("\n" + "=" * 60)
    print("[Stage B done] CSV: {}".format(out_csv))
    print("  videos       : {}".format(video_count))
    print("  total frames : {}".format(total_rows))
    print("  per-class    : {}".format(dict(label_count)))
    if 1 in label_count and 0 in label_count:
        ratio = label_count[0] / max(label_count[1], 1)
        print("  ratio normal:fall = {:.1f}:1".format(ratio))


# =====================================================================
# CLI
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Build a fall-detection training CSV from Le2i.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data-root', default='../data_Le2i',
                   help="Path to Le2i root, relative to the movenet/ dir "
                        "(or absolute). Should contain Coffee_room_01/, etc.")
    p.add_argument('--out-dir', default='../datasets',
                   help="Output dir for raw_keypoints/ and CSV")
    p.add_argument('--weights', default='output/movenet.pth',
                   help="Path to fire717's .pth weights, relative to movenet/")
    p.add_argument('--csv-name', default='le2i_keypoints.csv',
                   help="Output CSV filename (placed under --out-dir)")
    p.add_argument('--post-fall-frames', type=int, default=50,
                   help="Frames after `end` labeled as fall. 50 = 2s @ 25fps. "
                        "Computed in stage B; rerun with --skip-extract to "
                        "change this without re-running MoveNet.")
    p.add_argument('--skip-extract', action='store_true',
                   help="Skip stage A (assume raw_keypoints/ already populated)")
    p.add_argument('--skip-flatten', action='store_true',
                   help="Skip stage B (only run keypoint extraction)")
    p.add_argument('--force', action='store_true',
                   help="Stage A: re-run on videos that already have .npz")
    return p.parse_args()


def main():
    args = parse_args()

    # We need to run inside the movenet/ directory so that
    # `from lib import ...` works. Auto-detect and chdir if needed.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    movenet_dir = None

    candidates = [
        os.getcwd(),
        os.path.join(os.getcwd(), 'movenet'),
        os.path.join(script_dir, 'movenet'),
        script_dir,
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'lib')) and \
           os.path.isfile(os.path.join(c, 'config.py')):
            movenet_dir = c
            break

    if movenet_dir is None:
        print("[ERROR] Could not locate fire717's movenet/ directory.")
        print("        Looked for a directory containing both 'lib/' and 'config.py'.")
        print("        Either cd into movenet/ before running, or run from")
        print("        the project root (where movenet/ is a subdirectory).")
        sys.exit(1)

    if os.getcwd() != movenet_dir:
        print("[INFO] chdir to {} (so that `from lib import ...` works)".format(
            movenet_dir))
        os.chdir(movenet_dir)
    sys.path.insert(0, movenet_dir)

    raw_dir = os.path.join(args.out_dir, 'raw_keypoints')
    out_csv = os.path.join(args.out_dir, args.csv_name)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.skip_extract:
        print("[INFO] --skip-extract: reusing {}".format(raw_dir))
    else:
        if not os.path.exists(args.weights):
            print("[ERROR] weights not found: {}".format(args.weights))
            print("        (looked relative to {})".format(os.getcwd()))
            sys.exit(1)
        if not os.path.isdir(args.data_root):
            print("[ERROR] data root not found: {}".format(args.data_root))
            print("        (looked relative to {})".format(os.getcwd()))
            sys.exit(1)
        stage_a_extract(args.data_root, raw_dir, args.weights, force=args.force)

    if args.skip_flatten:
        print("[INFO] --skip-flatten: stage A only.")
    else:
        stage_b_flatten(raw_dir, out_csv,
                        post_fall_frames=args.post_fall_frames)

    print("\nNext step:")
    print("  Open train_gru.ipynb and set CSV_PATH = '{}'".format(
        os.path.abspath(out_csv)))


if __name__ == '__main__':
    main()