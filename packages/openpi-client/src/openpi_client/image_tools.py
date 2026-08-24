import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


def resize_no_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Direct (aspect-DISTORTING) resize to (height, width) with NO padding.

    Counterpart to resize_with_pad, which preserves aspect by fitting the frame inside the target and
    filling the remainder with black. On a 480x640 wrist frame into 224x224 that fit costs 56 of the
    224 rows (25% of the model's input carries nothing) and samples vertically at 224/640; this
    function spends all 224 rows on the scene, i.e. 224/480 vertically = +33% vertical sampling, at
    the price of a 1.33x anisotropic stretch. Field of view is identical either way -- resize_with_pad
    does not crop.

    Lives here, next to resize_with_pad, because ResizeImages runs inside CPU DataLoader workers on
    numpy arrays. openpi.shared.image_tools has a jax.jit'd resize_no_pad for on-device use, but
    openpi.transforms imports `from openpi_client import image_tools`, so a jax-only definition was
    unreachable: selecting resize_pad=False raised
        AttributeError: module 'openpi_client.image_tools' has no attribute 'resize_no_pad'
    which made every resize_pad=False config unrunnable.
    """
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [np.asarray(Image.fromarray(im).resize((width, height), resample=method)) for im in images]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])
