"""
trt_builder.py
==============
Build a TensorRT engine from an ONNX model.

Must run on the target device (Jetson Nano) -- engine files are tied to
the GPU architecture; an engine built on a PC won't load on the Nano.

Usage:
    from movenet.trt_builder import build_engine
    build_engine("movenet/output/pose_raw.onnx",
                 "movenet/output/pose.engine",
                 fp16=True)
"""

import os
import time

import tensorrt as trt


def build_engine(onnx_path,
                 engine_path,
                 fp16=True,
                 workspace_gb=1,
                 rebuild=False,
                 verbose=False):
    """
    Build a TensorRT engine from an ONNX model.

    Parameters
    ----------
    onnx_path : str
        Path to the input ONNX file.
    engine_path : str
        Path to write the serialized engine.
    fp16 : bool, default True
        Enable FP16 (Nano's Maxwell GPU supports FP16, recommended).
        Maxwell does NOT support INT8, so there is no INT8 option here.
    workspace_gb : int, default 1
        Workspace size in GB during build. Nano has limited memory;
        1 GB is usually enough for MoveNet.
    rebuild : bool, default False
        If engine_path already exists: False = skip build, True = rebuild.
    verbose : bool, default False
        True = INFO-level TensorRT logger; False = WARNING (quiet).

    Returns
    -------
    str : engine_path

    Raises
    ------
    FileNotFoundError : ONNX file not found
    RuntimeError      : ONNX parse failed / engine build failed (often OOM)
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            "ONNX file not found: {}\n"
            "Run pth2onnx.py on the PC first, then scp it to the Nano."
            .format(onnx_path)
        )

    if os.path.exists(engine_path) and not rebuild:
        size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        print("[SKIP] Engine already exists: {} ({:.1f} MB)".format(
            engine_path, size_mb))
        print("       Pass rebuild=True to force a rebuild.")
        return engine_path

    out_dir = os.path.dirname(engine_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    onnx_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print("[INFO] ONNX  : {} ({:.1f} MB)".format(onnx_path, onnx_mb))
    print("[INFO] Engine: {}".format(engine_path))
    print("[INFO] Config: FP16={}, workspace={} GB".format(fp16, workspace_gb))

    log_level = trt.Logger.INFO if verbose else trt.Logger.WARNING
    TRT_LOGGER = trt.Logger(log_level)
    print("[INFO] TensorRT {}".format(trt.__version__))
    print("[INFO] Building engine (5-15 min on Nano)...")

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 1. Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parsing failed:\n" + "\n".join(errors))

    print("[OK] ONNX parsed")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print("  Input  : {} {}".format(inp.name, tuple(inp.shape)))
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print("  Output : {} {}".format(out.name, tuple(out.shape)))

    # 2. Builder config
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_gb * (1 << 30)
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[OK] FP16 enabled")
        else:
            print("[WARN] platform has no fast FP16, falling back to FP32")

    # 3. Build & serialize
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(
            "Engine build failed. Common causes:\n"
            "  1. OOM - enable swap (4 GB recommended)\n"
            "  2. workspace too small - try a larger workspace_gb\n"
            "  3. ONNX contains unsupported ops - run with verbose=True"
        )

    with open(engine_path, 'wb') as f:
        f.write(serialized)

    elapsed = time.time() - t0
    eng_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print("\n" + "=" * 50)
    print("[DONE] Engine saved: {} ({:.1f} MB)".format(engine_path, eng_mb))
    print("       Build time : {:.0f} s".format(elapsed))
    print("=" * 50)
    return engine_path
