import torch
import os
import onnx
from onnxsim import simplify

# Paths (adjust if needed)
MODEL_DIR = "./"
PT_MODEL = os.path.join(MODEL_DIR, "mobilefacenet_scripted.pt")
ONNX_MODEL = os.path.join(MODEL_DIR, "mobilefacenet.onnx")
ONNX_SIMPLIFIED = os.path.join(MODEL_DIR, "mobilefacenet_sim.onnx")

#os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading TorchScript model...")
model = torch.jit.load(PT_MODEL, map_location="cpu")
model.eval()

dummy_input = torch.randn(1, 3, 112, 112)

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    ONNX_MODEL,
    input_names=["input"],
    output_names=["output"],
    opset_version=12,               # Better compatibility with ncnn
    do_constant_folding=True,
    export_params=True,
)

print(f"ONNX model saved: {ONNX_MODEL}")

# Quick test with onnxruntime (optional but strongly recommended)
print("Quick validation with onnxruntime...")
import onnxruntime as ort
ort_session = ort.InferenceSession(ONNX_MODEL)
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)
print("ONNX runtime test passed (output shape):", ort_outs[0].shape)  # should be [1, 128] or [1, 512]

print("Simplifying ONNX model...")
model_onnx = onnx.load(ONNX_MODEL)
model_simp, check = simplify(model_onnx)
assert check, "Simplified ONNX model check failed!"
onnx.save(model_simp, ONNX_SIMPLIFIED)
print(f"Simplified ONNX saved: {ONNX_SIMPLIFIED}")

print("\nDone!\n")
print("Next steps on your laptop or another machine with ncnn tools:")
print(f"  onnx2ncnn {ONNX_SIMPLIFIED} mobilefacenet.param mobilefacenet.bin")
print("  ncnnoptimize mobilefacenet.param mobilefacenet.bin mobilefacenet-opt.param mobilefacenet-opt.bin 0   # 0 = fp32, 65536 = fp16 if you want")