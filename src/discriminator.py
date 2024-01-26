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
    A deep convolutional generative adversarial network (DCGAN) discriminator module.

    This class implements a discriminator as part of a DCGAN, using a series of convolutional layers to classify images as real or generated (fake). It is designed to be used in generative tasks alongside a generator to produce realistic images.

    The architecture of the discriminator is as follows:
    1. Convolutional Layer: Reduces the spatial dimension and increases the depth of feature maps from the input image.
    2. LeakyReLU Activation: Allows for a small gradient when the unit is not active, preventing dying ReLU problem and helping the gradients to flow through the architecture.
    3. Batch Normalization: Stabilizes learning by normalizing the input to each activation layer.
    4. Further convolutions, leaky ReLUs, and batch normalizations: Continue to process the feature maps.
    5. Fully Connected Layer: Flattens the output and maps it to a single value.
    6. Sigmoid Activation: Outputs a probability indicating how likely the input image is real.

    The discriminator takes a single-channel image (e.g., grayscale) as input and outputs a scalar probability between 0 (fake) and 1 (real).

    Example usage:
        # Initialize the discriminator
        disc = Discriminator()

        # Pass an image (real or generated) to the discriminator
        probability_real = disc(real_image)
        probability_fake = disc(generated_image)

    Note:
    - The discriminator is crucial for the adversarial learning process, guiding the generator to produce more realistic images.
    - The input size to the discriminator must match the output size of the generator.
    - For RGB images, modify the input channel of the first layer to 3.

    This implementation is based on the principles outlined in the DCGAN paper: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" by Radford et al.
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
