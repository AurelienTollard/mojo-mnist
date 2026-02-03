from argparse import ArgumentParser
from logging import INFO, basicConfig, getLogger
from pathlib import Path
from typing import Callable, Optional, cast

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from .mnist import download_training_set, download_validation_set

ConvFn = Callable[..., None]
LinearFn = Callable[[Tensor, Tensor, Tensor, Tensor], None]

conv2d_maxpool_relu: Optional[ConvFn]
linear_relu: Optional[LinearFn]
linear: Optional[LinearFn]
_mojo_import_error: Optional[Exception] = None

try:
    from mojo_mnist.ops.conv import conv2d_maxpool_relu
    from mojo_mnist.ops.linear import linear, linear_relu
except Exception as exc:  # pragma: no cover - optional dependency
    conv2d_maxpool_relu = None
    linear_relu = None
    linear = None
    _mojo_import_error = exc

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


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        if conv2d_maxpool_relu is None or linear_relu is None or linear is None:
            raise RuntimeError(
                "Mojo ops are unavailable. Install the 'modular' package and "
                "ensure mojo_mnist.ops.conv can be imported."
            ) from _mojo_import_error

        self._conv = cast(ConvFn, conv2d_maxpool_relu)
        self._linear_relu = cast(LinearFn, linear_relu)
        self._linear = cast(LinearFn, linear)

        self.register_buffer("conv1_w", torch.empty(32, 1, 3, 3))
        self.register_buffer("conv1_b", torch.empty(32))
        self.register_buffer("conv2_w", torch.empty(64, 32, 3, 3))
        self.register_buffer("conv2_b", torch.empty(64))
        self.register_buffer("fc1_w_t", torch.empty(64 * 7 * 7, 128))
        self.register_buffer("fc1_b", torch.empty(128))
        self.register_buffer("fc2_w_t", torch.empty(128, num_classes))
        self.register_buffer("fc2_b", torch.empty(num_classes))

    def load_from_cnn(self, cnn: CNN) -> None:
        conv1 = cast(nn.Conv2d, cnn.features[0])
        conv2 = cast(nn.Conv2d, cnn.features[3])
        fc1 = cast(nn.Linear, cnn.classifier[1])
        fc2 = cast(nn.Linear, cnn.classifier[3])

        self.conv1_w.copy_(conv1.weight.detach())
        self.conv1_b.copy_(conv1.bias.detach())
        self.conv2_w.copy_(conv2.weight.detach())
        self.conv2_b.copy_(conv2.bias.detach())
        self.fc1_w_t.copy_(fc1.weight.detach().t().contiguous())
        self.fc1_b.copy_(fc1.bias.detach())
        self.fc2_w_t.copy_(fc2.weight.detach().t().contiguous())
        self.fc2_b.copy_(fc2.bias.detach())

    def load_from_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        self.conv1_w.copy_(state_dict["features.0.weight"])
        self.conv1_b.copy_(state_dict["features.0.bias"])
        self.conv2_w.copy_(state_dict["features.3.weight"])
        self.conv2_b.copy_(state_dict["features.3.bias"])
        self.fc1_w_t.copy_(state_dict["classifier.1.weight"].t().contiguous())
        self.fc1_b.copy_(state_dict["classifier.1.bias"])
        self.fc2_w_t.copy_(state_dict["classifier.3.weight"].t().contiguous())
        self.fc2_b.copy_(state_dict["classifier.3.bias"])

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        conv1_out = torch.empty(batch_size, 32, 14, 14, device=x.device, dtype=x.dtype)
        self._conv(
            conv1_out,
            x,
            self.conv1_w,
            self.conv1_b,
            stride=1,
            padding=1,
            pool_size=2,
            pool_stride=2,
            tile_size=16,
        )

        conv2_out = torch.empty(batch_size, 64, 7, 7, device=x.device, dtype=x.dtype)
        self._conv(
            conv2_out,
            conv1_out,
            self.conv2_w,
            self.conv2_b,
            stride=1,
            padding=1,
            pool_size=2,
            pool_stride=2,
            tile_size=16,
        )

        flat = conv2_out.reshape(batch_size, 64 * 7 * 7)
        hidden = torch.empty(batch_size, 128, device=x.device, dtype=x.dtype)
        self._linear_relu(hidden, flat, self.fc1_w_t, self.fc1_b)

        output = torch.empty(batch_size, 10, device=x.device, dtype=x.dtype)
        self._linear(output, hidden, self.fc2_w_t, self.fc2_b)
        return output


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
