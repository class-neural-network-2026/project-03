"""Generation of new images with a trained velocity field.

TODO: choose NUM_STEPS, and complete `integrate` and `sample`.
"""

import torch

# Used by the notebook. The value below is only a starting point.
NUM_STEPS = 100     # number of steps of the Euler method


def integrate(model, x, num_steps):
    """Move images from t = 0 to t = 1 along the velocity field: solve
    dx/dt = v(x, t) with the Euler method, in `num_steps` steps of size
    1 / num_steps.

        x <- x + v(x, t) / num_steps
        for t = 0, 1 / num_steps, 2 / num_steps, ..., (num_steps - 1) / num_steps

    The B images are at the same time: give it to the model as a tensor t
    of shape (B,).

    The model is not updated: put it in evaluation mode and do not track
    gradients (torch.no_grad).

    Args:
        model:     the velocity field, on the same device as x.
        x:         images at t = 0, shape (B, 1, 32, 32).
        num_steps: number of steps (int, at least 1).

    Returns:
        path: the images at the times 0, 1 / num_steps, ..., 1, a tensor of
        shape (num_steps + 1, B, 1, 32, 32). path[0] is the input x and
        path[-1] holds the images at t = 1.
    """
    # TODO
    raise NotImplementedError("mysample.py: integrate is not implemented")


def sample(model, num_samples, num_steps, device):
    """Generate new images: draw noise x ~ N(0, I) of the shape of the
    images, move it from t = 0 to t = 1 with `integrate`, and clamp the
    images at t = 1 to [0, 1] (the velocity field does not keep their values
    exactly in this range).

    Args:
        model:       the velocity field, already on `device`.
        num_samples: number of images to generate.
        num_steps:   number of steps of the Euler method.
        device:      torch.device to compute on.

    Returns:
        generated images, shape (num_samples, 1, 32, 32), values in [0, 1].
    """
    # TODO
    raise NotImplementedError("mysample.py: sample is not implemented")
