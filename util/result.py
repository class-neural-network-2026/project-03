"""Result of a trained flow matching model (provided: do not modify this
file).

Two functions are used by the notebook:

    evaluate(model, integrate, num_steps)
        measures the trained velocity field with the testing data, the
        5,000 images of `get_dataset("test")` in data.py:
            FID (Frechet inception distance) between 5,000 generated images
                and the 5,000 test images
        Lower is better.

    save(result, info)
        saves the value in the file
            result/result_<student ID>_<name>.csv

Usage (with the `util` and `code` folders on sys.path):
    from myinfo import get_info
    from mysample import NUM_STEPS, integrate
    from result import evaluate, save

    result = evaluate(model, integrate, NUM_STEPS)    # model: the trained velocity field
    save(result, get_info())

The FID is computed with the package `torcheval`:
    pip install torcheval torchvision
It uses the Inception-v3 network of torchvision, pretrained on ImageNet. Its
weights (about 100 MB) are downloaded the first time `evaluate` is called,
which requires an internet connection, and are then read from the cache
folder of PyTorch. `evaluate` takes seconds with a GPU, but can take more
than 10 minutes with a CPU.
"""

import csv
from pathlib import Path

import torch
from torcheval.metrics import FrechetInceptionDistance

from data import get_dataset

# <project root>/result, independent of the directory the notebook is run from
RESULT_DIR = Path(__file__).resolve().parent.parent / "result"

# Number of images processed at a time. It is kept small because the
# Inception-v3 network works on images enlarged to 299 x 299 pixels.
BATCH_SIZE = 50

# Seed of the noise of the generated images: the same noise is used every
# time, so that the FID of a model does not change from one evaluation to
# the next.
SEED = 0


@torch.no_grad()
def evaluate(model, integrate, num_steps):
    """Measure a trained velocity field with the test images.

    FID: as many images as test images (5,000) are generated. Noise drawn
    from N(0, I), of the shape of the images, is moved from t = 0 to t = 1
    along the velocity field by `integrate`, in `num_steps` steps, and the
    images at t = 1 are clamped to [0, 1]. The FID is the Frechet distance
    between the Gaussian distributions fitted to the Inception-v3 features
    of the generated images and of the test images. It is 0 when the two
    Gaussians are the same.

    The model is not updated: gradients are not tracked. The computation
    runs on the device of the model.

    Args:
        model:     the trained velocity field, with
                       model(x, t) -> v   x of shape (B, 1, 32, 32), t of shape (B,),
                                          v of shape (B, 1, 32, 32)
        integrate: the function `integrate` of mysample.py, with
                       integrate(model, x, num_steps) -> path
                                          x of shape (B, 1, 32, 32),
                                          path of shape (num_steps + 1, B, 1, 32, 32)
        num_steps: number of steps of the Euler method.

    Returns:
        dict {"fid": ...} of a Python float.
    """
    device = next(model.parameters()).device
    test_images = get_dataset("test").images

    # noise of the generated images, one per test image
    rng = torch.Generator().manual_seed(SEED)
    noise = torch.randn(test_images.shape, generator=rng)

    fid = FrechetInceptionDistance(device=device)

    for x, x0 in zip(test_images.split(BATCH_SIZE), noise.split(BATCH_SIZE)):
        x = x.to(device)

        # generated images: noise -> images at t = 1 (clamped to [0, 1] by _to_rgb)
        path = integrate(model, x0.to(device), num_steps)
        samples = path[-1]
        if samples.shape != x.shape:
            raise ValueError(
                f"evaluate: the generated images have the shape {tuple(samples.shape)}, "
                f"expected {tuple(x.shape)}"
            )

        fid.update(_to_rgb(x), is_real=True)
        fid.update(_to_rgb(samples), is_real=False)

    return {"fid": fid.compute().item()}


def save(result, info):
    """Save the result of `evaluate` in a CSV file.

    The file is result/result_<student ID>_<name>.csv. It has a header line
    and one line of values:

        student_id,name,fid

    Args:
        result: the dict returned by `evaluate`.
        info:   the dict returned by `get_info` in myinfo.py:
                {"name": ..., "student_id": ...}.

    Returns:
        the path of the saved file.
    """
    name = str(info["name"]).strip()
    student_id = str(info["student_id"]).strip()
    if not name or not student_id:
        raise ValueError("save: fill in your name and student ID in myinfo.py")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"result_{_for_filename(student_id)}_{_for_filename(name)}.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id", "name", "fid"])
        writer.writerow([student_id, name, result["fid"]])

    return path


def _to_rgb(images):
    """Gray images (B, 1, H, W) -> RGB images (B, 3, H, W), float32 with
    values in [0, 1], as expected by the FID of torcheval."""
    return images.float().clamp(0, 1).repeat(1, 3, 1, 1)


def _for_filename(text):
    """Replace the characters that are not letters or digits (spaces, /, ...)
    by "_", so that the text can be part of a file name."""
    return "".join(c if c.isalnum() else "_" for c in text)
