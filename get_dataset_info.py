from datasets import load_dataset

def get_dataset_info():
    dataset = load_dataset('google/fleurs', 'en_us')
    print(f"Dataset size: {len(dataset['train'])} train samples, {len(dataset['test'])} test samples")

if __name__ == "__main__":
    get_dataset_info()
