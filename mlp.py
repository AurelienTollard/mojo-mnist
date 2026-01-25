from argparse import ArgumentParser
from logging import INFO, basicConfig, getLogger
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import mnist

LOGGER = getLogger(__name__)

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
HIDDEN_SIZE = 256


class MLP(nn.Module):
    def __init__(
        self, input_size: int = 784, hidden_size: int = 256, num_classes: int = 10
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.layers(x)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> None:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    LOGGER.info(
        "Epoch %s: Train Loss %.4f, Train Accuracy %.2f%%", epoch, avg_loss, accuracy
    )


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    LOGGER.info("Test Loss %.4f, Test Accuracy %.2f%%", avg_loss, accuracy)
    return accuracy


def run_training(export_path: str) -> None:
    device = get_device()
    LOGGER.info("Using device: %s", device)

    train_data = mnist.download_training_set()
    validation_data = mnist.download_validation_set()

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)

    model = MLP(hidden_size=HIDDEN_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    LOGGER.info("Model architecture:\n%s", model)
    LOGGER.info("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, criterion, optimizer, device, epoch)

    LOGGER.info("Evaluating on test set...")
    test_accuracy = evaluate(model, test_loader, criterion, device)

    torch.save(model.state_dict(), export_path)
    LOGGER.info("Model saved to %s", export_path)
    LOGGER.info("Final test accuracy %.2f%%", test_accuracy)


def main() -> None:
    basicConfig(level=INFO, format="%(levelname)s: %(message)s")

    parser = ArgumentParser(description="MNIST MLP trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the MLP model")
    train_parser.add_argument(
        "--export",
        default=Path(__file__).parent / ".data" / "mlp_mnist.pth",
        help="Path to save the trained model",
    )

    args = parser.parse_args()

    if args.command == "train":
        run_training(args.export)


if __name__ == "__main__":
    main()
