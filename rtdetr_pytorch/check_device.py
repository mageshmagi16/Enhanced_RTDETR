import torch

print(" Checking PyTorch Device Configuration")

# Print available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Additional GPU info (only if CUDA is available)
if torch.cuda.is_available():
    print("CUDA is available")
    print("Current GPU:", torch.cuda.get_device_name(0))
    print("Total Memory (MB):", torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
    print("Memory Allocated (MB):", torch.cuda.memory_allocated(0) // (1024 * 1024))
    print("Memory Reserved (MB):", torch.cuda.memory_reserved(0) // (1024 * 1024))
else:
    print(" CUDA is not available. Using CPU.")
