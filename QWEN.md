# Qwen Code Context for `/Users/daniel/code/harmonic-bert`

## Project Overview

This directory contains the plan and potentially the implementation for an experimental project named "Harmonic SFT". This project is inspired by the paper "Harmonic Loss Trains Interpretable AI Models" and explores a novel fine-tuning methodology for language models that replaces the standard `CrossEntropyLoss` with a metric learning objective.

The core idea is to use a custom "Harmonic Loss" function and a "Distance-based Prediction Head" (`DistLayer`) for fine-tuning. Instead of predicting token classes directly, the model learns to position its output vectors in a semantic space relative to token prototypes. This aims to create richer internal representations where geometric distances are semantically meaningful, potentially improving generalization and generation quality. The approach also draws inspiration from "nGPT: Normalized Transformer with Representation Learning on the Hypersphere" through its use of unit sphere normalization.

The base model for this experiment is `chandar-lab/NeoBERT`.

## Key Components (from plan)

1.  **`DistLayer` Head**: A custom layer that replaces the standard linear classification head. It holds prototype vectors for each vocabulary token and calculates distances from the model's hidden state to these prototypes.
2.  **`HarmonicLoss` Function**: A custom loss function that works on the distances from `DistLayer`. It inverts distances, normalizes them into a probability distribution, and uses negative log-likelihood. It includes a hyperparameter `h_exp` (Harmonic Exponent) for temperature control.
3.  **Unit Sphere Normalization**: All vectors (hidden states and prototypes) are projected onto the unit sphere before distance calculations. This uses cosine similarity (efficient via `matmul`) and stabilizes training by preventing prototype vector collapse/divergence.
4.  **Training Strategy**:
    *   **Model**: `chandar-lab/NeoBERT` with context length 4096.
    *   **Modified Head**: The final decoder head is replaced by `DistLayer`, initialized from input embeddings.
    *   **Trainable Parameters**: Only the final encoder block (`transformer_encoder.27`) and the new `dist_head` are trained; the rest are frozen.
    *   **Precision**: `bfloat16` for frozen body and trainable encoder, `float32` for the trainable `dist_head`.
    *   **Optimizer**: `torch.optim.AdamW` with `lr=2e-4`, `betas=(0.95, 0.95)`.
    *   **Scheduler**: Warmup-Stable-Decay (WSD) with 10% warmup and cosine decay.
    *   **`h_exp` Annealing**: Starts at `-12.0` and ends at `-4.0` over training.
    *   **Regularization**: No `weight_decay`; explicit unit sphere normalization of `dist_head` weights post-optimizer step.
5.  **Hardware & Batching**:
    *   Target: Single RTX 4090.
    *   Per-device batch size: `2`.
    *   Gradient accumulation steps: `16`.
    *   Effective batch size: `32`.
    *   Attention: Uses `scaled_dot_product_attention` for FlashAttention.
    *   Memory: Plan A is no gradient checkpointing; Plan B enables it on frozen blocks if needed.
6.  **Dataset**: Intended for use with a high-quality instruction/chat dataset like `mlfoundations/dclm-baseline-1.0-parquet`.

## Current Status

Based on the files present (`README.md`, `QWEN.md`), this directory currently holds the detailed plan for the "Harmonic SFT" project. The actual implementation code is not present in the root directory.