import os
import sys
from typing import Any, Dict

import torch
import onnx

from lib import init, MoveNet, Task
from config import cfg


# ============================================================
# User config (Jetson Nano / TensorRT 8.2.1 friendly)
# ============================================================
WEIGHTS_PATH = "output/movenet.pth"       # change to your .pth path
OUTPUT_PATH = "output/pose_raw.onnx"      # recommended deployment ONNX
IMG_SIZE = 192                             # MoveNet input size for this repo
OPSET = 13                                 # safer for TensorRT 8.2.1 on Nano
USE_SIMPLIFIER = False                     # keep False first for max compatibility
USE_DYNAMIC_BATCH = False                  # static batch is easier for TRT on Nano
EXPORT_MODE = "train"
VERIFY_WITH_ORT = False                    # optional; can enable on PC if needed


# ============================================================
# Robust checkpoint loading
# ============================================================

def _clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove common wrappers like module./model. prefixes."""
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        cleaned[nk] = v
    return cleaned


def _extract_state_dict(ckpt: Any) -> Dict[str, Any]:
    """Try common checkpoint layouts."""
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return _clean_state_dict(ckpt[key])
        # maybe the dict itself is already a state_dict
        if ckpt and all(isinstance(k, str) for k in ckpt.keys()):
            return _clean_state_dict(ckpt)
    raise RuntimeError("Could not extract state_dict from checkpoint")


def robust_load_weights(task: Task, weights_path: str, device: str) -> None:
    """
    First try repo's original modelLoad().
    If that fails on PC/CPU or with different checkpoint wrappers,
    fall back to manual torch.load + load_state_dict.
    """
    try:
        task.modelLoad(weights_path)
        print("[OK] Loaded weights with repo modelLoad()")
        return
    except Exception as e:
        print(f"[WARN] modelLoad() failed: {e}")
        print("[INFO] Falling back to manual checkpoint loading...")

    ckpt = torch.load(weights_path, map_location=device)
    state_dict = _extract_state_dict(ckpt)

    missing, unexpected = task.model.load_state_dict(state_dict, strict=False)
    print("[OK] Loaded weights manually with strict=False")
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}):")
        for k in missing[:20]:
            print(f"    {k}")
        if len(missing) > 20:
            print("    ...")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}):")
        for k in unexpected[:20]:
            print(f"    {k}")
        if len(unexpected) > 20:
            print("    ...")


# ============================================================
# Export
# ============================================================

def export_test_onnx(cfg, weights_path: str, output_path: str) -> str:
    print("\n" + "=" * 70)
    print("Export MoveNet PTH -> ONNX (Nano / TRT 8.2.1 friendly)")
    print("=" * 70)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Weights: {weights_path}")
    print(f"[INFO] Output : {output_path}")
    print(f"[INFO] Opset  : {OPSET}")

    # Build model in TEST mode for deployment
    model = MoveNet(
        num_classes=cfg["num_classes"],
        width_mult=cfg["width_mult"],
        mode=EXPORT_MODE,
    )
    task = Task(cfg, model)
    task.model.to(device)

    robust_load_weights(task, weights_path, device)
    task.model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    input_names = ["input"]
    output_names = ["heatmap", "center", "regs", "offsets"]
    dynamic_axes = None
    if USE_DYNAMIC_BATCH:
        dynamic_axes = {
            "input": {0: "batch"},
            "keypoints": {0: "batch"},
        }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            task.model,
            dummy_input,
            output_path,
            export_params=True,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            opset_version=OPSET,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Exported ONNX: {output_path} ({size_mb:.2f} MB)")

    # Validate ONNX structure
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX structural check passed")

    print(f"[INFO] ONNX IR version: {onnx_model.ir_version}")
    opsets = [(imp.domain if imp.domain else "ai.onnx", imp.version) for imp in onnx_model.opset_import]
    print(f"[INFO] Opset imports  : {opsets}")

    for inp in onnx_model.graph.input:
        shape = [d.dim_param if d.dim_param else d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"[INFO] Input : {inp.name} -> {shape}")
    for out in onnx_model.graph.output:
        shape = [d.dim_param if d.dim_param else d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"[INFO] Output: {out.name} -> {shape}")

    # Optional simplify (disabled by default for compatibility)
    if USE_SIMPLIFIER:
        try:
            import onnxsim
            print("[INFO] Running onnx-simplifier...")
            sim_model, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(sim_model, output_path)
                new_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"[OK] Simplified ONNX: {size_mb:.2f} MB -> {new_size_mb:.2f} MB")
            else:
                print("[WARN] Simplifier returned check=False, keeping original export")
        except ImportError:
            print("[SKIP] onnx-simplifier not installed")
        except Exception as e:
            print(f"[WARN] onnx-simplifier failed: {e}")

    # Optional ORT verification (keep off by default if ORT/exporter combo is noisy)
    if VERIFY_WITH_ORT:
        try:
            import numpy as np
            import onnxruntime as ort

            sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
            ort_outs = sess.run(None, {"input": dummy_input.detach().cpu().numpy().astype(np.float32)})
            print("[OK] ONNX Runtime inference check passed")
            for name, out in zip(output_names, ort_outs):
                print(
                    f"[INFO] {name}: shape={tuple(out.shape)}, "
                    f"min={out.min():.6f}, max={out.max():.6f}"
                )
        except ImportError:
            print("[SKIP] onnxruntime not installed")
        except Exception as e:
            print(f"[WARN] ONNX Runtime verification failed: {e}")

    print("\n[ALL DONE]")
    print("Next step on Jetson Nano:")
    print("  /usr/src/tensorrt/bin/trtexec --onnx=output/pose_test.onnx --saveEngine=output/pose_fp16.engine --fp16")

    return output_path


if __name__ == "__main__":
    init(cfg)
    try:
        export_test_onnx(cfg, WEIGHTS_PATH, OUTPUT_PATH)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
