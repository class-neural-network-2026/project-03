"""Training of the velocity field: flow matching loss and one epoch of
training.

TODO: complete `interpolate`, `loss_function` and `train_one_epoch`.

Notation
    x1  : data, images of the training set,  shape (B, 1, 32, 32), values in [0, 1]
    x0  : noise,                             shape (B, 1, 32, 32), x0 ~ N(0, I)
    t   : times,                             shape (B,),           t ~ U[0, 1]
    x_t : points of the straight paths from x0 to x1, at the times t, shape (B, 1, 32, 32)
    v   : velocities given by the model at (x_t, t), shape (B, 1, 32, 32)
"""

import torch


def interpolate(x0, x1, t):
    """Point at the time t of the straight path from x0 (t = 0) to x1 (t = 1).

        x_t = (1 - t) x0 + t x1

    Each of the B pairs (x0, x1) has its own time: t has the shape (B,), and
    it must be reshaped to (B, 1, 1, 1) to multiply images of shape
    (B, 1, 32, 32).

    Args:
        x0: noise, shape (B, 1, 32, 32).
        x1: data,  shape (B, 1, 32, 32).
        t:  times, shape (B,), values in [0, 1].

    Returns:
        x_t, shape (B, 1, 32, 32).
    """
    # TODO
    raise NotImplementedError("mytrain.py: interpolate is not implemented")


def loss_function(v, x0, x1):
    """Flow matching loss of a batch.

        loss = mean of || v - (x1 - x0) ||^2

    x1 - x0 is the velocity of the straight path from x0 to x1: it is the
    target of the velocity v given by the model at a point of this path.
    The squared difference is summed over the pixels of an image and
    averaged over the B images of the batch.

    Args:
        v:  velocities given by the model, shape (B, 1, 32, 32).
        x0: noise, shape (B, 1, 32, 32).
        x1: data,  shape (B, 1, 32, 32).

    Returns:
        loss: a scalar tensor.
    """
    # TODO
    raise NotImplementedError("mytrain.py: loss_function is not implemented")


def train_one_epoch(model, optimizer, dataloader, device):
    """Train `model` for one epoch, i.e. one pass over `dataloader`.

    Put the model in training mode first. Then, for every mini-batch of B
    images:

        1. Move the images to `device`: they are the data x1.
        2. Draw the noise x0 ~ N(0, I), of the same shape as x1, and the
           times t ~ U[0, 1], of shape (B,).
        3. Compute x_t with `interpolate`, the velocities v = model(x_t, t),
           and the loss with `loss_function`.
        4. Back-propagate and update the parameters of the model with
           `optimizer`.

    Args:
        model:      the velocity field, already on `device`.
        optimizer:  the optimizer from myoptim.get_optimizer.
        dataloader: yields batches of images, shape (B, 1, 32, 32), values
                    in [0, 1] (there are no labels).
        device:     torch.device to compute on.

    Returns:
        dict {"loss": ...} of a Python float: the average of the loss per
        image over the whole epoch.
    """
    # TODO
    raise NotImplementedError("mytrain.py: train_one_epoch is not implemented")
