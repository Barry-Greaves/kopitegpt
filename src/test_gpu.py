import torch

print("=" * 50)
print("KopiteGPT Environment Check")
print("=" * 50)

print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"CUDA Runtime    : {torch.version.cuda}")

    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM            : {total:.2f} GB")

    x = torch.randn(10000, 10000, device="cuda")
    y = x @ x

    print("\nGPU test successful ✅")
    print(f"Tensor device   : {y.device}")
else:
    print("CUDA not detected")
    