import torch
import torch.nn as nn
import torch.nn.functional as F

class DistLayer(nn.Linear):
    """
    A layer that calculates distances to a set of prototype vectors.
    Inherits from nn.Linear for convenience but overrides the forward pass.
    The 'weight' attribute of this layer is our prototype matrix.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        # Prototypes should not have a bias term.
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the squared Euclidean distance between input vectors and prototypes.
        Assumes both x and self.weight (prototypes) are L2-normalized.

        d^2 = ||x - w||^2 = ||x||^2 - 2<x,w> + ||w||^2
            = 1 - 2<x,w> + 1  (since ||x|| = ||w|| = 1)
            = 2 * (1 - <x,w>)

        where <x,w> is the cosine similarity, calculated via F.linear.
        """
        # F.linear(x, w.T) is equivalent to x @ w.T
        cosine_similarity = F.linear(x, self.weight)

        # Clamp to avoid numerical instability with values slightly > 1.0
        cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)

        # Calculate squared Euclidean distance
        dist_sq = 2.0 * (1.0 - cosine_similarity)
        return dist_sq
