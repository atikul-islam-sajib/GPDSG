import logging
import argparse
import sys
import joblib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append("/src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    filemode="w",
    filename="./logs/data_loader.log/",
)


class Loader:
    """
    This script is designed to handle the downloading and loading of the MNIST dataset for training machine learning models. It utilizes PyTorch's torchvision datasets and DataLoader functionalities to facilitate easy and efficient data handling. The script includes command-line arguments for flexibility in usage.

    Classes:
        Loader: Handles the downloading of the MNIST dataset and creation of a DataLoader.

    Methods:
        download_mnist_digit(): Downloads the MNIST dataset and returns the dataset object.
        create_dataloader(mnist_digit_data): Creates a DataLoader from the provided MNIST dataset.

    Command-Line Arguments:
        --batch_size (int): Specifies the batch size for the DataLoader. Default is 64.
        --download_mnist (flag): Triggers the download of the MNIST dataset.

    Usage:
        Run the script with desired arguments. For example:
        `python data_loader_script.py --batch_size 64 --download_mnist`
        This command will download the MNIST dataset and create a DataLoader with a batch size of 64.

    Example:
        >>> loader = Loader(batch_size=64)
        >>> mnist_data = loader.download_mnist_digit()
        >>> loader.create_dataloader(mnist_digit_data=mnist_data)

    Logging:
        The script logs its progress and any errors encountered to `./logs/data_loader.log/`.

    Notes:
        - Ensure that the `torchvision` and `joblib` packages are installed.
        - The script is intended for use with the MNIST dataset, a collection of handwritten digits commonly used in machine learning.

    For more detailed usage and information, please refer to the script's documentation in MkDocs.
    """

    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def download_mnist_digit(self):
        logging.info("Downloading MNIST dataset".capitalize())

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        mnist_data = datasets.MNIST(
            root="./data/raw/",
            train=True,
            transform=transforms.ToTensor(),
            download=True,
        )

        logging.info("MNIST dataset downloaded".capitalize())

        return mnist_data

    def create_dataloader(self, mnist_digit_data):
        logging.info("Creating dataloader".capitalize())

        dataloader = DataLoader(
            mnist_digit_data, batch_size=self.batch_size, shuffle=True
        )

        try:
            logging.info("Dataloader created".capitalize())
            joblib.dump(value=dataloader, filename="./data/processed/dataloader.pkl")

        except Exception as e:
            logging.error(f"Error creating dataloader: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Loader".title())

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training".capitalize(),
    )
    parser.add_argument(
        "--download_mnist",
        action="store_true",
        help="Download MNIST dataset".capitalize(),
    )

    args = parser.parse_args()

    if args.download_mnist:
        if args.batch_size > 1:
            loader = Loader(batch_size=args.batch_size)
            mnist_data = loader.download_mnist_digit()
            loader.create_dataloader(mnist_digit_data=mnist_data)
        else:
            logging.exception("batch size should be greater than 1".capitalize())
    else:
        logging.exception("Please provide --download_mnist".capitalize())
