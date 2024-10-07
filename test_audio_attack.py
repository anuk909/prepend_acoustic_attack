import torch
import argparse
from src.attacker.audio_raw.learn_attack import AudioAttack
from src.tools.args import core_args, attack_args
from src.models.load_model import load_model
from src.data.load_data import load_data

# Parse arguments
core_args, _ = core_args()
attack_args, _ = attack_args()

# Ensure model_name is a list
if not isinstance(core_args.model_name, list):
    core_args.model_name = [core_args.model_name]

# Set batch size for testing
attack_args.bs = 2  # Small batch size for testing

def test_audio_attack():
    # Load model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(core_args, device=device)
    data, _ = load_data(core_args)

    print(f"Loaded {len(data)} samples")

    # Initialize AudioAttack
    attacker = AudioAttack(attack_args, model, device)

    # Prepare DataLoader
    train_dl = attacker._prep_dl(data, bs=attack_args.bs, shuffle=True)

    # Process a few batches
    for i, batch in enumerate(train_dl):
        if i >= 3:  # Process 3 batches
            break
        print(f'Processing batch {i+1}, shape: {batch.shape}')

        # Print first sample in the batch
        print(f'First sample in batch {i+1}: {batch[0]}')

        # Simulate forward pass
        with torch.no_grad():
            _ = attacker.audio_attack_model(batch.to(device), model)

        print(f'Batch {i+1} processed successfully')

        # Print current GPU memory usage
        if torch.cuda.is_available():
            print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')
            print(f'GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB')

    print('Test completed successfully')

if __name__ == "__main__":
    test_audio_attack()
