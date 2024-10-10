import torch

def check_gpu_memory():
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU Name: {gpu.name}")
        print(f"Total GPU Memory: {gpu.total_memory / 1024**3:.2f} GB")
        
        # Current memory usage
        print(f"Allocated GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Cached GPU Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("CUDA is not available. No GPU detected.")

if __name__ == "__main__":
    check_gpu_memory()