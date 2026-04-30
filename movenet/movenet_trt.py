"""
movenet_trt.py
==============
MoveNet TensorRT inference wrapper for Jetson Nano deployment.

Core API:
    movenet = MoveNetTRT("movenet/output/pose_fp16.engine")
    keypoints = movenet.infer(frame)   # frame: BGR ndarray (H, W, 3)
                                        # keypoints: (17, 3) float32

Output contract:
    keypoints[k] = (x, y, score)
    x, y    : pixel coordinates in the original frame
    score   : confidence in [0, 1]
    k order : COCO-17 (see KEYPOINT_NAMES)

Preprocessing (MUST match fire717 training)
-------------------------------------------
fire717's TensorDatasetTest does:
    BGR -> RGB -> resize(192, 192) -> astype(float32)
And the model's Backbone.forward does `x = x/127.5 - 1` internally.

So at inference time we must:
    1. NO letterbox -- just direct (non-uniform) resize to 192x192
    2. NO division by 255 -- model expects [0, 255] range as float32
    3. Use independent x/y scale factors to map predicted coords back
       to original frame size (since the resize was non-uniform).

Doing letterbox or /255 yields broken keypoints (heatmap_max ~0.3 instead
of ~0.8) and skeleton stuck in a fixed corner of the frame.
"""

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (initializes the CUDA context)


# COCO-17 keypoint names (matches fire717/movenet.pytorch order)
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

# Skeleton edges for visualization
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class MoveNetTRT(object):
    """
    MoveNet TensorRT inference.

    Preprocess: non-uniform resize + BGR->RGB + raw [0, 255] float32.
    Output    : (17, 3) numpy, mapped back to original-frame coords.
    """

    REQUIRED_OUTPUTS = ['heatmap', 'center', 'regs', 'offsets']

    def __init__(self, engine_path, img_size=192):
        self.img_size = img_size
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine: {}".format(engine_path))
        self.context = self.engine.create_execution_context()

        # Bookkeeping for input/output bindings
        self.bindings_info = []
        self.binding_addrs = []
        self.input_idx = -1
        self.output_idx = {}    # name -> binding index

        for i in range(self.engine.num_bindings):
            name = self.engine[i]
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(max(1, s) for s in shape)   # handle dynamic axes

            host_mem = np.empty(shape, dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            is_input = self.engine.binding_is_input(i)
            self.bindings_info.append({
                'name': name, 'shape': shape, 'dtype': dtype,
                'is_input': is_input,
                'host': host_mem, 'device': device_mem,
            })
            self.binding_addrs.append(int(device_mem))

            if is_input:
                self.input_idx = i
            else:
                self.output_idx[name] = i

        self.stream = cuda.Stream()

        missing = [n for n in self.REQUIRED_OUTPUTS if n not in self.output_idx]
        if missing:
            print("[WARN] engine is missing expected outputs: {}".format(missing))
            print("       found outputs: {}".format(list(self.output_idx.keys())))
            print("       Make sure the ONNX was exported with fire717/movenet "
                  "in train mode.")

        print("[OK] MoveNetTRT loaded: {}".format(engine_path))
        print("     input : {}".format(self.bindings_info[self.input_idx]['shape']))
        for n, idx in self.output_idx.items():
            print("     output: {} {}".format(n, self.bindings_info[idx]['shape']))

    # ---------- Preprocess (matches fire717 TensorDatasetTest) ----------
    def _preprocess(self, frame):
        """
        BGR (H, W, 3) uint8 -> (1, 3, 192, 192) float32

        Steps:
            1) BGR -> RGB
            2) Direct resize to 192x192 (NON-uniform, no letterbox)
            3) astype(float32), DO NOT divide by 255
               (the backbone does x/127.5 - 1 internally)
            4) HWC -> CHW + batch dim

        Returns the blob plus (scale_x, scale_y) so we can map predicted
        coords back to the original frame.
        """
        h, w = frame.shape[:2]

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)

        blob = img.astype(np.float32)                      # KEEP [0, 255]
        blob = np.transpose(blob, (2, 0, 1))[None]         # (1, 3, H, W)

        scale_x = float(w) / self.img_size
        scale_y = float(h) / self.img_size
        return blob, scale_x, scale_y

    # ---------- TRT forward ----------
    def _trt_forward(self, blob):
        b = self.bindings_info[self.input_idx]
        inp = np.ascontiguousarray(blob.astype(b['dtype']))
        cuda.memcpy_htod_async(b['device'], inp, self.stream)
        self.context.execute_async_v2(self.binding_addrs, self.stream.handle)

        outs = {}
        for name, idx in self.output_idx.items():
            ob = self.bindings_info[idx]
            host = np.empty(ob['shape'], dtype=ob['dtype'])
            cuda.memcpy_dtoh_async(host, ob['device'], self.stream)
            outs[name] = host
        self.stream.synchronize()
        return {k: v.astype(np.float32) for k, v in outs.items()}

    # ---------- Postprocess ----------
    def _postprocess(self, outs, scale_x, scale_y):
        """
        4-head decoder for fire717/movenet:
            1. argmax over center heatmap -> body center (cy, cx)
            2. regs[k] regresses from center to keypoint k (on 48x48 grid)
            3. offsets refines sub-pixel position
            4. heatmap[k] gives confidence
            5. independent x/y scaling to map back to original frame
        """
        heatmap = outs['heatmap']    # (1, 17, 48, 48)
        center  = outs['center']     # (1,  1, 48, 48)
        regs    = outs['regs']       # (1, 34, 48, 48)
        offsets = outs['offsets']    # (1, 34, 48, 48)

        Hf = heatmap.shape[2]
        Wf = heatmap.shape[3]
        stride = float(self.img_size) / Hf   # typically 4.0

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

            x_in = (kx + ox) * stride
            y_in = (ky + oy) * stride

            # Map back to original frame (non-uniform inverse of the resize)
            keypoints[k, 0] = x_in * scale_x
            keypoints[k, 1] = y_in * scale_y
            keypoints[k, 2] = float(heatmap[0, k, ky, kx])
        return keypoints

    # ---------- Public API ----------
    def infer(self, frame):
        """
        Parameters
        ----------
        frame : np.ndarray (H, W, 3) uint8, BGR

        Returns
        -------
        keypoints : np.ndarray (17, 3) float32
                    (x, y, score) with x/y in original-frame pixels.
        """
        blob, scale_x, scale_y = self._preprocess(frame)
        outs = self._trt_forward(blob)
        return self._postprocess(outs, scale_x, scale_y)


# ---------- Visualization ----------
def draw_keypoints(frame, keypoints, conf_thr=0.3,
                   color_pt=(0, 255, 0), color_line=(200, 200, 200)):
    """Draw skeleton + keypoints onto frame (debug helper)."""
    vis = frame.copy()
    for (i, j) in SKELETON:
        if keypoints[i, 2] > conf_thr and keypoints[j, 2] > conf_thr:
            p1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            p2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
            cv2.line(vis, p1, p2, color_line, 2)
    for k in range(17):
        x, y, s = keypoints[k]
        if s > conf_thr:
            cv2.circle(vis, (int(x), int(y)), 4, color_pt, -1)
    return vis
