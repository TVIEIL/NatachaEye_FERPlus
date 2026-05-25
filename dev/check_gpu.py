import torch

print(f"CUDA disponible : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Name : {torch.cuda.get_device_name(0)}")
