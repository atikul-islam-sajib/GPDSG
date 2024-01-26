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
    A deep convolutional generative adversarial network (DCGAN) generator module.

    This class implements a generator as part of a DCGAN, using a series of transposed convolutional layers to generate images from random noise. It is designed to be used in generative tasks where the goal is to produce realistic images.

    Parameters:
    - latent_space (int): Dimensionality of the latent space (random noise vector). Default is 100.

    The architecture of the generator is as follows:
    1. Transposed Convolutional Layer: Expands the input latent vector into a small feature map.
    2. Batch Normalization: Stabilizes learning by normalizing the input to each activation layer.
    3. ReLU Activation: Introduces non-linearity, allowing the model to generate complex patterns.
    4. Further transposed convolutions, batch normalizations, and ReLU activations: Continue to upscale the feature map to the desired output size.
    5. Tanh Activation: Scales the output to a range of [-1, 1], typical for image data.

    The final output is a single-channel image of the same height and width as the kernel in the last transposed convolutional layer. The output size and quality depend on the parameters of each layer and the complexity of the latent space.

    Example usage:
        # Initialize the generator
        gen = Generator(latent_space=100)

        # Generate a random noise vector
        noise = torch.randn((1, 100, 1, 1))

        # Generate an image
        fake_image = gen(noise)

    Note:
    - This implementation assumes a single-channel output (e.g., grayscale images). For generating RGB images, modify the output channel of the last layer to 3.
    - The quality of generated images heavily depends on the training process and the complexity of the model.

    For more information on DCGANs, refer to the original paper: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" by Radford et al.
    """

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
