from typing import List

from PIL import Image

import torch.nn.functional as F


def stack_images_horizontally(images: List, save_path=None):
    widths, heights = list(zip(*(i.size for i in images)))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGBA", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    if save_path is not None:
        new_im.save(save_path)
    return new_im


def images2gif(
    images: List, save_path, optimize=True, duration=None, loop=0, disposal=2
):
    if duration is None:
        duration = int(5000 / len(images))
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        duration=duration,
        loop=loop,
        disposal=disposal,
    )


def stack_images_vertically(images: List, save_path=None):
    widths, heights = list(zip(*(i.size for i in images)))
    max_width = max(widths)
    total_height = sum(heights)
    new_im = Image.new("RGBA", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    if save_path is not None:
        new_im.save(save_path)
    return new_im


def merge_images(images: List):
    if isinstance(images[0], Image.Image):
        return stack_images_horizontally(images)

    images = list(map(stack_images_horizontally, images))
    return stack_images_vertically(images)

def resize(img):
    h, w = img.shape[2:]
    l = min(h, w)
    h = int(h * 512 / l)
    w = int(w * 512 / l)
    img_512 = F.interpolate(img, size=(h, w), mode="bilinear")
    return img_512
