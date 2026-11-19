"""Flow matching: velocity field.

TODO: complete the class `VelocityField`.

The velocity field is a neural network that gives the velocity v(x, t) of an
image x at the time t. You are free to design its architecture (type and
number of layers, their sizes, the way the time is given to the network),
but keep the interface described in the docstrings: the notebook and the
other files rely on it.

Notation
    B : number of images in a batch
    x : images,     shape (B, 1, 32, 32)
    t : times,      shape (B,), values in [0, 1]
                    t = 0 : noise, x ~ N(0, I)
                    t = 1 : data, images with values in [0, 1]
    v : velocities, shape (B, 1, 32, 32)
"""

import torch
import torch.nn as nn


class VelocityField(nn.Module):
    def __init__(self):
        """Define the layers of the velocity field.

        The notebook creates the model with `VelocityField()`, so every
        argument you add needs a default value.

        The velocity of a path from noise to data contains the noise, which
        is different in every pixel: a network that reduces the image to a
        few numbers (a small hidden layer) cannot give it. Convolutional
        layers are a good choice, and so are skip connections, which give
        the outputs of the first layers to the last layers.
        """
        super().__init__()
        # TODO: define the layers
        raise NotImplementedError("mymodel.py: VelocityField.__init__ is not implemented")

    def forward(self, x, t):
        """Velocity v(x, t) of the images x at the times t.

        The velocity depends on x and on t. A simple way to give the time to
        the network is to join it to the image as a second channel, whose
        pixels all have the value t: the input of the network then has the
        shape (B, 2, 32, 32).

        Args:
            x: images, shape (B, 1, 32, 32).
            t: times, shape (B,), values in [0, 1]: one time per image.

        Returns:
            velocities, shape (B, 1, 32, 32).
        """
        # TODO
        raise NotImplementedError("mymodel.py: VelocityField.forward is not implemented")
