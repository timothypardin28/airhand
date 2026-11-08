import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def process_dataset(path, batch_size=32, test_size=0.2, random_state=42):
    """Create training and validation data loaders from a dataset."""
    df = pd.read_csv(path).drop(columns=['person', 'hand', 'session'])
    x = df.drop('label').to_numpy().astype('float32')
    y = df['label'].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state
    )

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
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

    return train_loader, val_loader, le

def train(model, train_loader, val_loader, epochs, lr, device):
    """Train a model on data loaders, evaluate per epoch, and save artifacts."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for x, y in tqdm(train_loader, desc="Training"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validating"):
                x, y = x.to(device), y.to(device)
                outputs = model(x)

                loss = criterion(outputs, y)
                running_val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                total += y.size(0)
                correct += (preds == y).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1:02d}/{epochs:02d}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

    return history