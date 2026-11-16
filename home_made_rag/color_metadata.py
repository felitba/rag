"""Consistent dominant-color estimation for catalog and uploaded images."""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

from PIL import Image, ImageOps


def estimate_garment_color(image_path: str | Path) -> dict[str, str | float]:
    """Return a broad dominant-color label and its pixel share.

    The estimator downsamples the image, masks a flat background connected to
    the borders, maps foreground pixels into a fixed HSV palette, then picks
    the most frequent palette label. Confidence is the share of retained pixels
    assigned to that label; it is not a calibrated probability.
    """
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((128, 128))

    pixels = list(image.getdata())
    foreground = _foreground_pixels(image)
    if len(foreground) < max(20, int(len(pixels) * 0.04)):
        foreground = pixels

    counts = Counter(_color_name(rgb) for rgb in foreground)
    color, count = counts.most_common(1)[0]
    return {
        "color": color,
        "color_source": "image_dominant_hsv",
        "color_confidence": round(count / len(foreground), 3),
    }


def _foreground_pixels(image: Image.Image) -> list[tuple[int, int, int]]:
    width, height = image.size
    pixels = list(image.getdata())
    corners = (
        pixels[0],
        pixels[width - 1],
        pixels[(height - 1) * width],
        pixels[height * width - 1],
    )
    if max(
        sum(abs(a - b) for a, b in zip(first, second))
        for index, first in enumerate(corners)
        for second in corners[index + 1 :]
    ) > 72:
        return pixels
    background = tuple(
        sorted(values)[len(values) // 2]
        for values in zip(*corners)
    )

    def resembles_background(index: int) -> bool:
        return sum(abs(channel - base) for channel, base in zip(pixels[index], background)) <= 48

    visited: set[int] = set()
    queue = deque()
    for x in range(width):
        queue.extend((x, (height - 1) * width + x))
    for y in range(height):
        queue.extend((y * width, y * width + width - 1))

    while queue:
        index = queue.popleft()
        if index in visited or not resembles_background(index):
            continue
        visited.add(index)
        x, y = index % width, index // width
        if x > 0:
            queue.append(index - 1)
        if x + 1 < width:
            queue.append(index + 1)
        if y > 0:
            queue.append(index - width)
        if y + 1 < height:
            queue.append(index + width)

    return [pixel for index, pixel in enumerate(pixels) if index not in visited]


def _color_name(rgb: tuple[int, int, int]) -> str:
    red, green, blue = (channel / 255 for channel in rgb)
    high, low = max(red, green, blue), min(red, green, blue)
    delta = high - low
    saturation = 0 if high == 0 else delta / high

    if high < 0.16:
        return "black"
    if saturation < 0.16:
        if high > 0.9:
            return "white"
        if high > 0.68:
            return "beige"
        if high < 0.48:
            return "brown"
        return "gray"

    if red >= green and red >= blue:
        hue = 60 * ((green - blue) / delta % 6)
    elif green >= red and green >= blue:
        hue = 60 * ((blue - red) / delta + 2)
    else:
        hue = 60 * ((red - green) / delta + 4)

    if 8 <= hue < 40 and high < 0.58:
        return "brown"
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 40:
        return "orange"
    if hue < 68:
        return "yellow"
    if hue < 165:
        return "green"
    if hue < 255:
        return "blue"
    if hue < 290:
        return "purple"
    return "pink"
