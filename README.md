# Project: Harmonic SFT

This project is inspired by and implements concepts from the paper "Harmonic Loss Trains Interpretable AI Models" ([arXiv](https://arxiv.org/abs/2502.01628), [Twitter](https://x.com/dbaek__/status/1886781418115862544), [Github](https://github.com/KindXiaoming/grow-crystals)). We explore applying the Harmonic Loss methodology to fine-tuning language models.

## Vision

This project explores a novel fine-tuning methodology for language models that replaces the standard `CrossEntropyLoss` with a metric learning objective. The core hypothesis is that the one-hot encoding target used in standard Supervised Fine-Tuning (SFT) is a semantically poor objective. It punishes the model equally for predicting a close synonym ("automobile") as it does for predicting a completely unrelated token ("banana") when the target is "car".

We will implement a **Harmonic Loss** function coupled with a **Distance-based Prediction Head** (`DistLayer`). This architecture reframes the next-token prediction task from "which class is correct?" to "where should the output vector lie in a high-dimensional semantic space?". The goal is to train a model that produces richer, more nuanced representations, where the geometric distance between output vectors and token prototypes is semantically meaningful.

This experiment also draws inspiration from the concept of representation learning on the hypersphere, as explored in "nGPT: Normalized Transformer with Representation Learning on the Hypersphere" ([arXiv](https://arxiv.org/html/2410.01131v1)). By constraining vectors to the unit hypersphere (as done with our Unit Sphere Normalization), we aim to leverage potential benefits like faster learning and improved representation structure.

This experiment aims to validate if this approach leads to a model with a better-structured internal representation space, potentially improving generalization, reducing brittleness, and enabling more nuanced generation.

## Installation

This project uses `uv` for dependency management. To set up the environment:

1. Install `uv` if you haven't already:
   ```bash
   pip install uv
   ```

2. Sync the dependencies:
   ```bash
   uv sync
   ```

### CUDA Support

For CUDA-enabled workstations (Linux/Windows), you can install the CUDA version of PyTorch after syncing:

```bash
uv pip install torch --index https://download.pytorch.org/whl/cu129
```

This will install the CUDA-compatible version of PyTorch for better performance on NVIDIA GPUs.

## Experiment Tracking

This project uses [Trackio](https://huggingface.co/blog/trackio) for experiment tracking, which is a lightweight, open-source alternative to Weights & Biases. Trackio provides:

- Local dashboard for visualizing training metrics
- Easy sharing and embedding of results
- Seamless integration with Hugging Face Spaces
- Drop-in replacement for wandb API

To view the tracking dashboard, run:
```bash
python view_tracking.py
```

Or from the command line:
```bash
trackio show
```

## Core Mechanics

### 1. The `DistLayer` Head

The standard `nn.Linear` classification head is replaced with a custom `DistLayer`.

-   **Structure:** A layer containing a weight matrix of shape `(vocab_size, hidden_dim)`, where each row is a "prototype" vector for a token in the vocabulary.
-   **Initialization:** The prototypes are initialized with the pre-trained input embedding weights from the base model (`chandar-lab/NeoBERT`). This provides a semantically rich starting point.
-   **Forward Pass:** Instead of a matrix multiplication to produce logits, this layer calculates the distance from the model's final hidden state to every prototype vector in the vocabulary.

### 2. The `HarmonicLoss` Function

This custom loss function operates on the distances produced by the `DistLayer`.

-   **Calculation:**
    1.  Distances are inverted using a negative exponent: `inverted_dist = distances ** h_exp`.
    2.  The inverted distances are normalized to form a probability distribution.
    3.  The loss is the negative log-likelihood of the target token's probability.
-   **`h_exp` (Harmonic Exponent):** This is a critical hyperparameter that acts like a temperature control.
    -   A very negative `h_exp` (e.g., -12.0) creates a "hard" distribution, closely mimicking an `argmax` and the one-hot nature of the pre-trained model.
    -   A less negative `h_exp` (e.g., -4.0) creates a "softer" distribution, allowing the model to assign meaningful probabilities to semantically similar tokens.

### 3. Unit Sphere Normalization

To ensure computational feasibility and training stability, all vectors are projected onto the unit sphere before distance calculation.

-   **Reasoning:**
    1.  **Efficiency:** The squared Euclidean distance between two unit vectors `x` and `w` simplifies to `2 * (1 - cosine_similarity(x, w))`. Cosine similarity is a standard, highly-optimized matrix multiplication (`matmul`), which avoids a memory-exploding intermediate tensor that a naive distance calculation would require.
    2.  **Stability:** Constraining the prototypes to the unit sphere prevents them from collapsing to zero or diverging to infinity. This is a hard constraint enforced by re-normalizing the `dist_head` weights after every optimizer step, which is a cleaner alternative to `weight_decay`.

## Experiment Specification

### Model & Architecture

-   **Base Model:** `chandar-lab/NeoBERT`
    -   **Details:** NeoBERT is a next-generation encoder model for English text representation, pre-trained from scratch on the RefinedWeb dataset. It integrates state-of-the-art advancements in architecture, modern data, and optimized pre-training methodologies. It is designed for seamless adoption: it serves as a plug-and-play replacement for existing base models, relies on an optimal depth-to-width ratio, and leverages an extended context length of 4,096 tokens. Despite its compact 250M parameter footprint, it is the most efficient model of its kind and achieves state-of-the-art results on the massive MTEB benchmark, outperforming BERT large, RoBERTa large, NomicBERT, and ModernBERT under identical fine-tuning conditions.
-   **Context Length:** 4096 tokens
-   **Modified Head:** The `decoder` head is replaced with our `DistLayer`, initialized from the base model's input embeddings.

### Training Strategy

-   **Trainable Parameters:** Only the **final encoder block (`transformer_encoder.27`)** and the new **`dist_head`** will be unfrozen and trained. All other layers remain frozen.
-   **Precision:**
    -   **Frozen Body (`encoder.0` to `encoder.26`):** `bfloat16`
    -   **Trainable Encoder (`encoder.27`):** `bfloat16`
    -   **Trainable Head (`dist_head`):** **`float32`** for maximum numerical precision on the core learning objective.
-   **Optimizer:** `torch.optim.AdamW`
    -   `lr`: `2e-4`
    -   `betas`: `(0.95, 0.95)`. Chosen for improved stability with small batch sizes and noisy gradients, providing a smoother descent path.
-   **Scheduler:** Warmup-Stable-Decay (WSD).
    -   **Warmup:** 10% of total steps.
    -   **Decay:** Cosine annealing.
-   **Harmonic Exponent Annealing:** The `h_exp` will be annealed over the course of training.
    -   **Start Value:** `-12.0` (to align with the pre-trained model's "one-hot" bias).
    -   **End Value:** `-4.0` (to guide the model towards a richer, "softer" semantic landscape).
-   **Regularization:**
    -   No `weight_decay`.
    -   The `dist_head.weight` prototypes will be explicitly re-normalized to the unit sphere after each `optimizer.step()`.

### Dataset

-   **Source:** A high-quality instruction or chat dataset, such as `mlfoundations/dclm-baseline-1.0-parquet`.

This document outlines a complete, well-reasoned plan to test a fundamental hypothesis about language model fine-tuning. The choices made reflect a balance of theoretical rigor, computational pragmatism, and specific insights gained from prior experimentation.
