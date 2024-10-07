import torch
from src.attacker.mel.soft_prompt_attack import SoftPromptAttack
from src.tools.args import core_args, attack_args
from src.model.load_model import load_model
from src.data.load_data import load_data

def test_batch_processing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and data
    model = load_model(core_args, device=device)
    data, _ = load_data(core_args)

    # Initialize SoftPromptAttack
    attacker = SoftPromptAttack(attack_args, model, device)

    # Prepare DataLoader
    train_dl = attacker._prep_dl(data, bs=attack_args.bs, shuffle=True)

    # Process a few batches
    for i, batch in enumerate(train_dl):
        if i >= 5:  # Process 5 batches
            break
        print(f"Processing batch {i+1}, shape: {batch.shape}")

        # Simulate forward pass
        with torch.no_grad():
            _ = attacker.softprompt_model(batch.to(device), model)

        print(f"Batch {i+1} processed successfully")

    print("Test completed successfully")

if __name__ == "__main__":
    test_batch_processing()
