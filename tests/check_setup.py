import torch

print("=== GPU CHECK ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM: {total_mem:.2f} GB")
else:
    print("WARNING: CUDA not available, training will be extremely slow on CPU")