import logging
import argparse
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/discriminator.log",
)


class Discriminator(nn.Module):
    """
    Discriminator Class for Deep Convolutional Generative Adversarial Networks (DCGAN)

    The Discriminator is a fundamental component of DCGAN, tasked with the classification of images as real or generated (fake). This module utilizes convolutional neural networks (CNNs) to distinguish between authentic and artificially generated images, playing a vital role in the adversarial learning process.

    Classes:
    --------
    Discriminator(nn.Module):
        Implements a discriminator network for DCGANs, composed of multiple layers of convolutional neural networks.

    Constructor:
    ------------
    __init__(self):
        Initializes the Discriminator model with a pre-defined architecture.

    Methods:
    -------
    forward(self, x):
        Conducts a forward pass through the discriminator network.

        Parameters:
            x (torch.Tensor): Input tensor representing the image batch.

        Returns:
            torch.Tensor: Output tensor representing the probability of the input being real.

    Attributes:
    -----------
    model (torch.nn.Sequential):
        The sequential model constituting the discriminator's architecture.

    out (torch.nn.Sequential):
        Final layers of the model that flatten the output and apply sigmoid activation.

    Architecture:
    -------------
    1. Convolutional Layers:
    - Extract and downsample features from the input image.
    2. LeakyReLU Activation:
    - Introduces non-linearity and mitigates the dying ReLU problem.
    3. Batch Normalization:
    - Normalizes the input to each layer, enhancing model stability.
    4. Fully Connected Layer & Sigmoid Activation:
    - Flattens the output and transforms it to a probability score.

    Script Usage:
    -------------
    The script can be executed from the command line, allowing users to initialize the Discriminator model and report its total parameter count.

    Command-Line Arguments:
    -----------------------
    --discriminator:
        Flag to initialize and evaluate the Discriminator model.

    Example:
    --------
    Command to run the script:

        python discriminator_script.py --discriminator

    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=0
            ),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.out = nn.Sequential(
            nn.Linear(in_features=256 * 4 * 4, out_features=1), nn.Sigmoid()
        )

    def forward(self, x):
        if x is not None:
            x = self.model(x)
            x = x.reshape(x.shape[0], -1)
            x = self.out(x)
        else:
            x = "ERROR"
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discriminator model is defined".capitalize()
    )
    parser.add_argument(
        "--discriminator",
        help="Discriminator model is defined".capitalize(),
    )

    args = parser.parse_args()

    if args.discriminator:
        discriminator = Discriminator()

        total_parameters = 0

        for _, params in discriminator.named_parameters():
            total_parameters += params.numel()

        print("Total number of parameters # {} ".format(total_parameters).upper())

        logging.info("Total parameters of Discriminator # {} ".format(total_parameters))
    else:
        logging.exception("Discriminator model is not defined".capitalize())
