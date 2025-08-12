import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from .dist_layer import DistLayer
from .loss import HarmonicLoss

class HarmonicBert(nn.Module):
    def __init__(self, model_name: str, h_exp_initial: float):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Store the initial h_exp value
        self.h_exp_initial = h_exp_initial
        self.h_exp = h_exp_initial

        hidden_dim = self.bert.config.hidden_size
        vocab_size = self.bert.config.vocab_size

        # Create the new head
        self.dist_head = DistLayer(hidden_dim, vocab_size)

        # Initialize prototypes from the original input embeddings
        # (assuming the base model has tied weights or a get_input_embeddings method)
        input_embeddings = self.bert.get_input_embeddings()
        self.dist_head.weight = nn.Parameter(
            input_embeddings.weight.clone().detach()
        )
        
        # Initialize the loss function
        self.loss_fn = HarmonicLoss(harmonic_exp=self.h_exp)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):
        # Get final hidden state from the BERT body
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        h_final = outputs.last_hidden_state

        # Normalize both hidden states and prototypes for stable distance calc
        h_final_norm = F.normalize(h_final, p=2, dim=-1)
        prototypes_norm = F.normalize(self.dist_head.weight, p=2, dim=-1)

        # Pass the normalized hidden state to the dist_head
        # The dist_head needs the normalized prototypes, so we pass them in.
        # A more encapsulated way would be to normalize inside the DistLayer,
        # but this makes the normalization explicit.

        # Re-implementing the dist_layer logic here for clarity with normalization
        cosine_similarity = F.linear(h_final_norm, prototypes_norm)
        cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
        distances = 2.0 * (1.0 - cosine_similarity)
        
        loss = None
        if labels is not None:
            # Update the loss function with current h_exp
            self.loss_fn.harmonic_exp = self.h_exp
            loss = self.loss_fn(distances, labels.view(-1))

        return type('obj', (object,), {'loss': loss, 'distances': distances})()
