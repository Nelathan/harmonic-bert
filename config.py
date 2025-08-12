# config.py
# Centralized configuration for the experiment

import torch

# --- Model & Tokenizer ---
MODEL_NAME = 'neoblock/neobert-dclm-base'
CONTEXT_LENGTH = 4096

# --- Training ---
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
EFFECTIVE_BATCH_SIZE = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = 2e-4
OPTIMIZER_BETAS = (0.95, 0.95)
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1 # 10% of total steps for warmup

# --- Harmonic Loss ---
H_EXP_INITIAL = -12.0
H_EXP_FINAL = -4.0

# --- Hardware & Precision ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PRECISION_FROZEN = torch.bfloat16
PRECISION_TRAINABLE_HEAD = torch.float32

# --- Dataset ---
DATASET_NAME = 'dclm/dclm-10k' # Placeholder, adjust as needed

# --- Checkpointing & Logging ---
USE_GRADIENT_CHECKPOINTING = False # Set to True if OOM
LOGGING_STEPS = 10