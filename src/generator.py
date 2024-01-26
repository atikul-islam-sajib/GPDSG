import logging
import argparse
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/generator.log",
)


class Generator(nn.Module):
    """
    Deep Convolutional Generative Adversarial Network (DCGAN) Generator Module.

    This module contains the `Generator` class, which is a key component of a DCGAN,
    designed to generate images from random noise using transposed convolutional layers.
    The script provides a command-line interface to initialize the Generator model
    with customizable parameters and to report the total number of parameters in the model.

    Classes:
        Generator: Implements a generator for DCGAN with customizable latent space size.

    Functions:
        __init__(self, latent_space=100): Initializes the Generator model.
        forward(self, x): Defines the forward pass of the model.

    Command-Line Arguments:
        --latent_space (int): Sets the size of the latent space. Default is 100.
        --generator (flag): Triggers the initialization of the Generator model.

    Example:
        To initialize a Generator with a latent space of 100 and report its parameter count:
        ```
        python generator_script.py --latent_space 100 --generator
        ```

    Class `Generator`:
    ------------------
        The Generator class is responsible for creating a deep learning model capable of
        generating images from a latent noise vector.

        Methods:
            __init__(self, latent_space=100):
                Constructs the Generator model.
                Parameters:
                    latent_space (int): The dimensionality of the input latent vector.

            forward(self, x):
                Performs the forward pass of the model.
                Parameters:
                    x (torch.Tensor): A tensor of the latent noise vector.
                Returns:
                    torch.Tensor: The generated image.

    Script Usage:
    -------------
        This script, when run from the command line, allows for the initialization of the
        Generator model with a specified latent space size. It also calculates and prints
        the total number of parameters in the model.

        Example:
            Run the script with the following command to initialize the model and
            print the parameter count:
            ```
            python generator_script.py --latent_space 100 --generator
            ```

    Notes:
        - The model architecture includes several transposed convolutional layers,
          batch normalization, and ReLU activations, ending with a Tanh activation.
        - This implementation is optimized for single-channel image output.
        - The quality and characteristics of the generated images depend on the model's training.

    References:
        - "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
          by Radford et al. for foundational concepts in DCGANs.
    """

    # Your class and script code follows here...

    def __init__(self, latent_space=100):
        super(Generator, self).__init__()
        self.latent_space = latent_space
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.latent_space,
                out_channels=256,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        if x is not None:
            x = self.model(x)
        else:
            x = "ERROR"
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generator model is defined".capitalize()
    )
    parser.add_argument(
        "--latent_space", type=int, default=100, help="latent space".capitalize()
    )
    parser.add_argument(
        "--generator",
        action="store_true",
        help="Generator model is defined".capitalize(),
    )

    args = parser.parse_args()

    if args.generator:
        if args.latent_space:
            generator = Generator(args.latent_space)

            total_parameters = 0

            for _, params in generator.named_parameters():
                total_parameters += params.numel()

            print("Total number of parameters # {} ".format(total_parameters).upper())
            logging.info("Total parameters of Generator # {} ".format(total_parameters))
        else:
            logging.exception("Latent space is not defined".capitalize())
    else:
        logging.exception("Generator model is not defined".capitalize())
