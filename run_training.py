# run_training.py
# Main script to assemble and run the experiment.

import torch
from transformers import AutoTokenizer

import config
from model.harmonic_bert import HarmonicBert
from training.data import get_dataloaders
from training.trainer import HarmonicTrainer
# from training.scheduler import get_wsd_scheduler # You'll import this

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

    # --- YOUR ASSEMBLY CODE GOES HERE ---

    # TODO: Surgically set requires_grad and dtypes for model parameters

    # TODO: Filter parameters for the optimizer
    # trainable_params = [p for p in model.parameters() if p.requires_grad]

    # TODO: Instantiate optimizer
    # optimizer = torch.optim.AdamW(
    #     trainable_params,
    #     lr=config.LEARNING_RATE,
    #     betas=config.OPTIMIZER_BETAS
    # )

    # TODO: Instantiate scheduler
    # num_training_steps = len(train_loader) * config.NUM_EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
    # warmup_steps = int(num_training_steps * config.WARMUP_RATIO)
    # scheduler = get_wsd_scheduler(optimizer, warmup_steps, num_training_steps)

    # TODO: Instantiate the Trainer
    # trainer = HarmonicTrainer(model, optimizer, scheduler, train_loader, config)

    # TODO: Run the training
    # trainer.run()

    print("--- Experiment Setup Complete ---")
    print("Ready for you to assemble and run.")


if __name__ == "__main__":
    import math # Add math import for scheduler
    main()
