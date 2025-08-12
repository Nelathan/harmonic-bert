import torch
from transformers import AutoTokenizer, AdamW
import math

import config
from model.harmonic_bert import HarmonicBert
from training.data import get_dataloaders
from training.trainer import HarmonicTrainer
from training.scheduler import get_wsd_scheduler

def main():
    print("--- Initializing Harmonic SFT Experiment ---")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # 2. Data
    print("Loading data...")
    train_loader, _ = get_dataloaders(config, tokenizer)

    # 3. Model
    print("Initializing model...")
    model = HarmonicBert(
        model_name=config.MODEL_NAME,
        h_exp_initial=config.H_EXP_INITIAL
    ).to(config.DEVICE)

    # --- ASSEMBLY CODE ---

    for name, param in model.named_parameters():
        if "dist_head" in name:
            param.requires_grad = True
        else:
            # Freeze all other parameters
            param.requires_grad = False

    # Filter parameters for the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")

    # Instantiate optimizer
    optimizer = AdamW(
        trainable_params,
        lr=config.LEARNING_RATE,
        betas=config.OPTIMIZER_BETAS
    )

    # Instantiate scheduler
    num_training_steps = len(train_loader) * config.NUM_EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(num_training_steps * config.WARMUP_RATIO)
    scheduler = get_wsd_scheduler(optimizer, warmup_steps, num_training_steps)

    # Instantiate the Trainer
    trainer = HarmonicTrainer(model, optimizer, scheduler, train_loader, config)

    # Run the training
    trainer.run()

    print("--- Experiment Complete ---")


if __name__ == "__main__":
    main()
