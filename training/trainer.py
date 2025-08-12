# training/trainer.py

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Import trackio as wandb for compatibility
try:
    import trackio as wandb
    TRACKING_AVAILABLE = True
except ImportError:
    print("Trackio not installed. Install with 'pip install trackio' for experiment tracking.")
    TRACKING_AVAILABLE = False
    wandb = None

class HarmonicTrainer:
    def __init__(self, model, optimizer, scheduler, train_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.config = config
        self.scaler = torch.cuda.amp.GradScaler() # For potential mixed precision
        
        # Initialize tracking
        self.tracking_enabled = self.config.TRACKING_ENABLED and TRACKING_AVAILABLE
        if self.tracking_enabled:
            wandb.init(
                project=self.config.TRACKING_PROJECT_NAME,
                config={
                    "model_name": self.config.MODEL_NAME,
                    "batch_size": self.config.PER_DEVICE_BATCH_SIZE,
                    "gradient_accumulation_steps": self.config.GRADIENT_ACCUMULATION_STEPS,
                    "effective_batch_size": self.config.EFFECTIVE_BATCH_SIZE,
                    "learning_rate": self.config.LEARNING_RATE,
                    "num_epochs": self.config.NUM_EPOCHS,
                    "warmup_ratio": self.config.WARMUP_RATIO,
                    "h_exp_initial": self.config.H_EXP_INITIAL,
                    "h_exp_final": self.config.H_EXP_FINAL,
                }
            )

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

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        
        # Calculate total steps for h_exp annealing
        total_training_steps = len(self.train_loader) // self.config.GRADIENT_ACCUMULATION_STEPS

        for step, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            # Move batch to device
            batch = {k: v.to(self.config.DEVICE) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids=batch['input_ids'], 
                                   attention_mask=batch['attention_mask'], 
                                   labels=batch['labels'])
                loss = outputs.loss
            
            # Backward pass with gradient accumulation
            self.scaler.scale(loss).backward()
            
            total_loss += loss.item()
            total_steps += 1

            if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                # Unscale gradients and clip
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Scheduler step
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Our custom steps
                self._project_prototypes()

                # Update h_exp in the loss function
                global_step = (epoch * total_training_steps) + (step // self.config.GRADIENT_ACCUMULATION_STEPS)
                current_h_exp = self._anneal_h_exp(global_step, total_training_steps * self.config.NUM_EPOCHS)
                
                # Update the model's h_exp
                self.model.h_exp = current_h_exp

                # Logging
                if (step + 1) % (self.config.GRADIENT_ACCUMULATION_STEPS * self.config.LOGGING_STEPS) == 0:
                    avg_loss = total_loss / total_steps
                    learning_rate = self.scheduler.get_last_lr()[0]
                    
                    log_dict = {
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step,
                        "train_loss": avg_loss,
                        "learning_rate": learning_rate,
                        "h_exp": current_h_exp,
                    }
                    
                    # Log to console
                    print(f"Epoch {epoch}, Step {step}, Loss: {avg_loss:.4f}, LR: {learning_rate:.2e}, H_exp: {current_h_exp:.2f}")
                    
                    # Log to trackio
                    if self.tracking_enabled:
                        wandb.log(log_dict)
                    
                    # Reset for next logging interval
                    total_loss = 0.0
                    total_steps = 0

    def run(self):
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"--- Epoch {epoch+1}/{self.config.NUM_EPOCHS} ---")
            self.train_epoch(epoch)
        
        # Finish tracking
        if self.tracking_enabled:
            wandb.finish()
