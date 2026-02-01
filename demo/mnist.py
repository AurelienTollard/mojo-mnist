from argparse import ArgumentParser
from logging import INFO, basicConfig, getLogger
from pathlib import Path
from shutil import rmtree

from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

LOGGER = getLogger(__name__)
DEFAULT_DATASET_PATH = Path(__file__).parent / ".data"


def download_validation_set(root: Path | str = DEFAULT_DATASET_PATH) -> Dataset:
    LOGGER.info("Downloading validation set...")
    set = datasets.MNIST(root=root, train=False, download=True, transform=ToTensor())
    LOGGER.info("done.")
    return set


def download_training_set(root: Path | str = DEFAULT_DATASET_PATH) -> Dataset:
    LOGGER.info("Downloading training set...")
    set = datasets.MNIST(root=root, train=True, download=True, transform=ToTensor())
    LOGGER.info("done.")
    return set


def clear_dataset(root: Path | str = DEFAULT_DATASET_PATH) -> None:
    path = Path(root)
    if path.exists():
        rmtree(path)
        LOGGER.info("Cleared dataset at %s.", path)
    else:
        LOGGER.info("Dataset path %s does not exist.", path)


def main() -> None:
    basicConfig(level=INFO, format="%(levelname)s: %(message)s")

    parser = ArgumentParser(description="MNIST dataset utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Download training and validation sets")
    subparsers.add_parser("clear", help="Remove downloaded dataset")

    args = parser.parse_args()

    if args.command == "download":
        download_training_set()
        download_validation_set()
        return

    if args.command == "clear":
        clear_dataset()


if __name__ == "__main__":
    main()
