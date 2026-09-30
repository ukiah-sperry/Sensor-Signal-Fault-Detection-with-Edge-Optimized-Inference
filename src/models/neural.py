"""Small PyTorch MLP for bearing fault classification and edge quantization."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FaultMLP(nn.Module):
    def __init__(self, input_dim: int = 516, hidden_dim: int = 64, n_classes: int = 4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    hidden_dim: int = 64,
    random_state: int = 42,
) -> FaultMLP:
    """Train FaultMLP on numpy feature matrix X with integer labels y."""
    torch.manual_seed(random_state)
    input_dim = X.shape[1]
    n_classes = len(np.unique(y))

    model = FaultMLP(input_dim=input_dim, hidden_dim=hidden_dim, n_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    return model


def predict_mlp(model: FaultMLP, X: np.ndarray) -> np.ndarray:
    """Return integer class predictions for feature matrix X."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
    return logits.argmax(dim=1).numpy()
