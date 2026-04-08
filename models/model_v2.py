import torch
import torch.nn as nn


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def create_model():
    return nn.Sequential(

        # (Batch Size, Sequence Length) -> (Batch Size, one channel, Sequence Length)
        LambdaLayer(lambda x: x.unsqueeze(1)),

        # --- Block 1 ---
        nn.Conv1d(1, 8, kernel_size=7, padding=3),
        nn.BatchNorm1d(8),
        nn.ReLU(),

        # --- Block 2 ---
        nn.Conv1d(8, 16, kernel_size=9, padding=4),
        nn.BatchNorm1d(16),
        nn.ReLU(),

        nn.MaxPool1d(2),  # downsample

        # --- Block 3 ---
        nn.Conv1d(16, 32, kernel_size=11, padding=5),
        nn.BatchNorm1d(32),
        nn.ReLU(),

        # --- Block 4 ---
        nn.Conv1d(32, 32, kernel_size=11, padding=5),
        nn.BatchNorm1d(32),
        nn.ReLU(),

        nn.MaxPool1d(2),

        # --- Block 5 ---
        nn.Conv1d(32, 64, kernel_size=15, padding=7),
        nn.BatchNorm1d(64),
        nn.ReLU(),

        # --- Global pooling ---
        nn.AdaptiveAvgPool1d(1),

        # (B, C, 1) → (B, C)
        LambdaLayer(lambda x: x.reshape(x.size(0), -1)),

        # --- Classifier ---
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(32, 1)  # output logit
    )
