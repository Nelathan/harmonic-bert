# model/loss.py

import torch
import torch.nn as nn

class HarmonicLoss(nn.Module):
    """
    Calculates loss based on distances, using a tunable harmonic exponent.
    
    What is Harmonic Loss?
    - Harmonic logit d_i is defined as the l_2 distance between the weight vector w_i 
      and the input (query) x:  d_i = ||w_i - x||_2.
    - The probability p_i is computed using the harmonic max function:
      p_i = (d_i^n) / sum_j(d_j^n)
      where n is the harmonic exponent—a hyperparameter that controls the 
      heavy-tailedness of the probability distribution.
    - Harmonic Loss achieves (1) nonlinear separability, (2) fast convergence, 
      (3) scale invariance, (4) interpretability by design, properties that are 
      not available in cross-entropy loss.
      
    Changes from reference implementation:
    - The original implementation might use d_i^n directly in the loss calculation,
      but this version uses (d_i^2 + epsilon)^n to ensure numerical stability
      with squared Euclidean distances and avoid issues with negative distances
      when the harmonic exponent is negative (which is common).
    - Added a small epsilon (1e-8) to prevent division by zero or log(0) if a 
      distance is exactly zero.
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
