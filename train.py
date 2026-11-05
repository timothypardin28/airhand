import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def create_data_loaders(dataset, batch_size=32, test_size=0.2, random_state=42):
    """Create training and validation data loaders from a dataset."""
    train_dataset, val_dataset = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
        )
    return train_loader, val_loader

def train(model, train_loader, val_loader, epochs, lr, device):
    """Train a model on the given data loaders."""
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                outputs = model(x)
                val_loss += criterion(outputs, y).item()
                pred, _ = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {loss.item():.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {correct/total:.4f}")