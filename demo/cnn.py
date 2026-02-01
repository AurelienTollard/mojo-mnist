from argparse import ArgumentParser
from logging import INFO, basicConfig, getLogger
from pathlib import Path

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from .mnist import download_training_set, download_validation_set

LOGGER = getLogger(__name__)

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.classifier(x)


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

    train_data = download_training_set()
    validation_data = download_validation_set()

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN().to(device)
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


def run_validation(model_path: str) -> None:
    device = get_device()
    LOGGER.info("Using device: %s", device)

    validation_data = download_validation_set()
    test_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)

    state_dict = torch.load(model_path, map_location="cpu")
    model = CNN().to(device)
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()
    LOGGER.info("Evaluating model on validation set...")
    accuracy = evaluate(model, test_loader, criterion, device)
    LOGGER.info("Validation accuracy %.2f%%", accuracy)


def main() -> None:
    basicConfig(level=INFO, format="%(levelname)s: %(message)s", force=True)

    parser = ArgumentParser(description="MNIST CNN trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the CNN model")
    train_parser.add_argument(
        "--export",
        default=Path(__file__).parent.parent / ".data" / "cnn_mnist.pth",
        help="Path to save the trained model",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Run validation with a trained model"
    )
    validate_parser.add_argument(
        "--model",
        default=Path(__file__).parent.parent / ".data" / "cnn_mnist.pth",
        help="Path to the trained model weights",
    )

    args = parser.parse_args()

    if args.command == "train":
        run_training(args.export)
    elif args.command == "validate":
        run_validation(args.model)


if __name__ == "__main__":
    if __package__ in (None, ""):
        raise RuntimeError("Run as a module: python -m demo.cnn [args]")
    main()
