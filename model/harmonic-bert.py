import torch
import torch.nn as nn
from transformers import AutoModel
from .dist_layer import DistLayer

class HarmonicBert(nn.Module):
    def __init__(self, model_name: str, h_exp_initial: float):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
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

        return distances
