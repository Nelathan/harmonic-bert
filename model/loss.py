# model/loss.py

import torch
import torch.nn as nn

class HarmonicLoss(nn.Module):
    """
    Calculates loss based on distances, using a tunable harmonic exponent.
    """
    def __init__(self, harmonic_exp: float):
        super().__init__()
        self.harmonic_exp = harmonic_exp

    def forward(self, distances: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances (torch.Tensor): Tensor of shape (batch_size, vocab_size)
                                      containing squared Euclidean distances.
            targets (torch.Tensor):   Tensor of shape (batch_size,) with target token IDs.

        Returns:
            torch.Tensor: A scalar loss value.
        """
        # Add a small epsilon to prevent division by zero or log(0)
        # if a distance is exactly zero.
        distances = distances + 1e-8

        inverted_dist = distances ** self.harmonic_exp

        # Normalize to get probabilities
        probs = inverted_dist / torch.sum(inverted_dist, dim=-1, keepdim=True)

        # Get the probabilities of the correct target tokens
        target_probs = probs.gather(1, targets.unsqueeze(-1)).squeeze()

        # Calculate negative log-likelihood loss
        loss = -torch.log(target_probs).mean()

        return loss
