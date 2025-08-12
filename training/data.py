# training/data.py

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

def get_dataloaders(config, tokenizer):
    """
    Creates DataLoaders for training and evaluation.

    For large datasets, we use streaming to avoid loading everything into memory.
    No shuffling is performed, and the full dataset is used for one epoch.
    Steps (batches) are the primary measure of progress.
    """
    raw_dataset = load_dataset(config.DATASET_NAME, streaming=True, split="train")

    # No shuffling for now

    # Tokenize the dataset
    # This is a placeholder. You'll need to implement the actual tokenization logic
    # based on your dataset's format and requirements.
    # For example, if your dataset has a 'text' column:
    # def tokenize_function(example):
    #     return tokenizer(example['text'], truncation=True, padding='max_length', max_length=config.CONTEXT_LENGTH)
    #
    # tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

    # For now, we'll continue using dummy data to keep the structure
    dummy_data = {
        "input_ids": [[101] * config.CONTEXT_LENGTH for _ in range(100)],
        "attention_mask": [[1] * config.CONTEXT_LENGTH for _ in range(100)],
        "labels": [[102] * config.CONTEXT_LENGTH for _ in range(100)],
    }

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100
        def __getitem__(self, idx):
            return {k: torch.tensor(v[idx]) for k, v in dummy_data.items()}

    train_dataset = DummyDataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.PER_DEVICE_BATCH_SIZE,
        shuffle=False  # No shuffling
    )

    # You would also create a validation loader here
    eval_loader = None

    return train_loader, eval_loader
