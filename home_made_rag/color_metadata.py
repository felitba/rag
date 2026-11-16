"""Consistent dominant-color estimation for catalog and uploaded images."""

from __future__ import annotations

from collections import Counter, deque
from math import exp, sqrt
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, ImageOps


def estimate_garment_color(image_path: Image.Image | str | Path) -> dict[str, str | float]:
    """Return a dominant palette label, confidence, and representative RGB.

    The estimator downsamples the image, masks a flat background connected to
    the borders, maps foreground pixels into a fixed HSV palette, then picks
    the most frequent palette label. Confidence is the share of retained pixels
    assigned to that label; it is not a calibrated probability.
    """
    if isinstance(image_path, Image.Image):
        image = ImageOps.exif_transpose(image_path).convert("RGB")
    else:
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((128, 128))

    pixels = list(image.getdata())
    foreground = _foreground_pixels(image)
    if len(foreground) < max(20, int(len(pixels) * 0.04)):
        foreground = pixels

    labeled_pixels = [(rgb, _color_name(rgb)) for rgb in foreground]
    counts = Counter(label for _, label in labeled_pixels)
    color, count = counts.most_common(1)[0]
    dominant_pixels = [rgb for rgb, label in labeled_pixels if label == color]
    return {
        "color": color,
        "color_source": "image_dominant_hsv",
        "color_confidence": round(count / len(foreground), 3),
        "color_r": round(median(pixel[0] for pixel in dominant_pixels) / 255, 4),
        "color_g": round(median(pixel[1] for pixel in dominant_pixels) / 255, 4),
        "color_b": round(median(pixel[2] for pixel in dominant_pixels) / 255, 4),
    }


_COLOR_ANCHORS = {
    "black": (32, 32, 32),
    "gray": (128, 128, 128),
    "white": (245, 245, 242),
    "beige": (214, 198, 164),
    "brown": (126, 78, 42),
    "red": (193, 47, 51),
    "orange": (225, 126, 35),
    "yellow": (222, 194, 57),
    "green": (68, 125, 70),
    "blue": (65, 105, 160),
    "purple": (126, 79, 151),
    "pink": (215, 137, 166),
}


def color_similarity(
    query: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    *,
    delta_e_scale: float = 40.0,
) -> float | None:
    """Return a gradual 0..1 similarity from CIE Lab color distance.

    Use measured RGB channels when available. Older catalog/index records fall
    back to a representative color per broad palette label.
    """
    if not query or not candidate:
        return None
    query_rgb = _rgb_for_color(query)
    candidate_rgb = _rgb_for_color(candidate)
    if query_rgb is None or candidate_rgb is None:
        return None

    delta_e = _delta_e_76(_rgb_to_lab(query_rgb), _rgb_to_lab(candidate_rgb))
    scale = max(float(delta_e_scale), 1e-6)
    return round(exp(-delta_e / scale), 6)


def color_coverage_similarity(
    query: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> float | None:
    """Compare how much of each image is covered by its dominant color."""
    if not query or not candidate:
        return None
    query_share = query.get("color_confidence")
    candidate_share = candidate.get("color_confidence")
    if not isinstance(query_share, (int, float)) or not isinstance(candidate_share, (int, float)):
        return None
    query_share = max(0.0, min(1.0, float(query_share)))
    candidate_share = max(0.0, min(1.0, float(candidate_share)))
    return round(1.0 - abs(query_share - candidate_share), 6)


def combined_color_similarity(
    query: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    *,
    delta_e_scale: float = 40.0,
    tone_weight: float = 0.75,
    coverage_weight: float = 0.25,
) -> dict[str, float | None]:
    tone = color_similarity(query, candidate, delta_e_scale=delta_e_scale)
    coverage = color_coverage_similarity(query, candidate)
    parts = ((tone, tone_weight), (coverage, coverage_weight))
    available_weight = sum(weight for value, weight in parts if value is not None)
    score = None
    if tone is not None and available_weight > 0:
        score = round(
            sum(value * weight for value, weight in parts if value is not None)
            / available_weight,
            6,
        )
    return {"score": score, "tone": tone, "coverage": coverage}


def _rgb_for_color(color: dict[str, Any]) -> tuple[float, float, float] | None:
    channels = tuple(color.get(f"color_{channel}") for channel in "rgb")
    if all(isinstance(value, (int, float)) for value in channels):
        return tuple(max(0.0, min(1.0, float(value))) for value in channels)

    label = str(color.get("color") or "").strip().lower()
    aliases = {"grey": "gray", "cream": "beige", "ivory": "white", "tan": "beige"}
    anchor = _COLOR_ANCHORS.get(aliases.get(label, label))
    if anchor is None:
        return None
    return tuple(channel / 255 for channel in anchor)


def _rgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    def linearize(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linearize(channel) for channel in rgb)
    x = (red * 0.4124564 + green * 0.3575761 + blue * 0.1804375) / 0.95047
    y = red * 0.2126729 + green * 0.7151522 + blue * 0.0721750
    z = (red * 0.0193339 + green * 0.1191920 + blue * 0.9503041) / 1.08883

    def transform(value: float) -> float:
        return value ** (1 / 3) if value > 0.008856 else 7.787 * value + 16 / 116

    fx, fy, fz = transform(x), transform(y), transform(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def _delta_e_76(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> float:
    return sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


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
