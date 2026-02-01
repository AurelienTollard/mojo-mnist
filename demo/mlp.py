from argparse import ArgumentParser
from logging import INFO, basicConfig, getLogger
from pathlib import Path
from typing import Callable, cast

import torch
from torch import Tensor, empty, nn, optim
from torch.utils.data import DataLoader

from mojo_mnist.ops.linear import linear, linear_relu

from .mnist import download_training_set, download_validation_set

LOGGER = getLogger(__name__)

LinearFn = Callable[[Tensor, Tensor, Tensor, Tensor], None]

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
        self.layers: nn.Sequential = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        return self.layers(x)


class MojoMLP(nn.Module):
    def __init__(
        self, input_size: int = 784, hidden_size: int = 256, num_classes: int = 10
    ) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self._linear_relu = cast(LinearFn, linear_relu)
        self._linear = cast(LinearFn, linear)

        # Store weights as buffers so .to(device) moves them
        self.register_buffer("w1_t", empty(input_size, hidden_size))
        self.register_buffer("b1", empty(hidden_size))
        self.register_buffer("w2_t", empty(hidden_size, hidden_size))
        self.register_buffer("b2", empty(hidden_size))
        self.register_buffer("w3_t", empty(hidden_size, num_classes))
        self.register_buffer("b3", empty(num_classes))

    def load_from_mlp(self, mlp: MLP) -> None:
        layer1 = cast(nn.Linear, mlp.layers[0])
        layer2 = cast(nn.Linear, mlp.layers[2])
        layer3 = cast(nn.Linear, mlp.layers[4])

        self.w1_t.copy_(layer1.weight.t().contiguous())
        self.b1.copy_(layer1.bias)
        self.w2_t.copy_(layer2.weight.t().contiguous())
        self.b2.copy_(layer2.bias)
        self.w3_t.copy_(layer3.weight.t().contiguous())
        self.b3.copy_(layer3.bias)

    def load_from_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        self.w1_t.copy_(state_dict["layers.0.weight"].t().contiguous())
        self.b1.copy_(state_dict["layers.0.bias"])
        self.w2_t.copy_(state_dict["layers.2.weight"].t().contiguous())
        self.b2.copy_(state_dict["layers.2.bias"])
        self.w3_t.copy_(state_dict["layers.4.weight"].t().contiguous())
        self.b3.copy_(state_dict["layers.4.bias"])

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        batch_size = x.shape[0]

        hidden1 = empty(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        self._linear_relu(hidden1, x, self.w1_t, self.b1)

        hidden2 = empty(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        self._linear_relu(hidden2, hidden1, self.w2_t, self.b2)

        output = empty(batch_size, self.num_classes, device=x.device, dtype=x.dtype)
        self._linear(output, hidden2, self.w3_t, self.b3)

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


def run_validation(backend: str, model_path: str) -> None:
    device = get_device()
    LOGGER.info("Using device: %s", device)

    validation_data = download_validation_set()
    test_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)

    state_dict = torch.load(model_path, map_location="cpu")

    if backend == "pytorch":
        model = MLP(hidden_size=HIDDEN_SIZE)
        model.load_state_dict(state_dict)
        model = model.to(device)
    elif backend == "mojo":
        if device.type != "cuda":
            raise RuntimeError("Mojo backend requires CUDA")
        model = MojoMLP(hidden_size=HIDDEN_SIZE).to(device)
        model.load_from_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    criterion = nn.CrossEntropyLoss()

    LOGGER.info("Evaluating %s model on validation set...", backend)
    accuracy = evaluate(model, test_loader, criterion, device)
    LOGGER.info("Validation accuracy %.2f%%", accuracy)


def main() -> None:
    basicConfig(level=INFO, format="%(levelname)s: %(message)s", force=True)

    parser = ArgumentParser(description="MNIST MLP trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the MLP model")
    train_parser.add_argument(
        "--export",
        default=Path(__file__).parent / ".data" / "mlp_mnist.pth",
        help="Path to save the trained model",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Run validation with a trained model"
    )
    validate_parser.add_argument(
        "--backend",
        choices=["pytorch", "mojo"],
        default="pytorch",
        help="Backend to use for validation",
    )
    validate_parser.add_argument(
        "--model",
        default=Path(__file__).parent / ".data" / "mlp_mnist.pth",
        help="Path to the trained model weights",
    )

    args = parser.parse_args()

    if args.command == "train":
        run_training(args.export)
    elif args.command == "validate":
        run_validation(args.backend, args.model)


if __name__ == "__main__":
    if __package__ in (None, ""):
        raise RuntimeError("Run as a module: python -m mojo_mnist.demo.mlp [args]")
    main()
