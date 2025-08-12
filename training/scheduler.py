from torch.optim.lr_scheduler import LambdaLR

def get_wsd_scheduler(optimizer, warmup_steps, total_training_steps):
    """
    Implements a Warmup-Stable-Decay (WSD) scheduler.
    - Linear warmup for `warmup_steps`.
    - Constant learning rate for a "stability" phase.
    - Cosine decay for the remainder.

    Note: A simple implementation is provided. You might find pre-built
    WSD schedulers in libraries like `timm` or can refine this.
    For this project, a simple warmup-cosine decay is a great start.
    """
    def lr_lambda(current_step):
        # Linear warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay after warmup
        progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
