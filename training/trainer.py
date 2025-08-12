# training/trainer.py

import torch
import torch.nn.functional as F
from tqdm import tqdm

class HarmonicTrainer:
    def __init__(self, model, optimizer, scheduler, train_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.config = config
        self.loss_fn = None # You will instantiate this here
        self.scaler = torch.cuda.amp.GradScaler() # For potential mixed precision

    def _anneal_h_exp(self, global_step, total_steps):
        """Linearly anneals the harmonic exponent."""
        progress = global_step / total_steps
        h_exp_range = self.config.H_EXP_FINAL - self.config.H_EXP_INITIAL
        current_h_exp = self.config.H_EXP_INITIAL + progress * h_exp_range
        return current_h_exp

    def _project_prototypes(self):
        """Projects the dist_head prototypes back to the unit sphere."""
        with torch.no_grad():
            self.model.dist_head.weight.data = F.normalize(
                self.model.dist_head.weight.data, p=2, dim=-1
            )

    def train_epoch(self):
        self.model.train()

        # Example loop structure
        for step, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            # Move batch to device
            # Forward pass
            # Calculate loss
            # Backward pass with gradient accumulation

            if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                # Unscale, clip grad norm
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Our custom steps
                self._project_prototypes()

                # Update h_exp in the loss function
                # self.loss_fn.harmonic_exp = self._anneal_h_exp(...)

                # Logging
                # ...

        pass # End of epoch

    def run(self):
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"--- Epoch {epoch+1}/{self.config.NUM_EPOCHS} ---")
            self.train_epoch()
