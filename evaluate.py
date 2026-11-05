import torch

@torch.no_grad()
def evaluate(model, test_loader, device="cpu"):
    """Evaluate a model on the given test data loader."""
    model.eval()
    model.to(device)
    correct, total = 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        pred, _ = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (pred == y).sum().item()

    print(f"Test Accuracy: {correct/total:.4f}")