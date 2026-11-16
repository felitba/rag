from __future__ import annotations

import logging
import threading
from pathlib import Path

from PIL import Image, ImageOps


logger = logging.getLogger(__name__)


class ForegroundPreprocessor:
    """Remove image backgrounds and prepare a neutral, garment-focused view."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        model_name: str = "u2net",
        max_side: int = 1024,
        background_color: str = "#f1f0ec",
        padding_ratio: float = 0.10,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.max_side = max(256, int(max_side))
        self.background_color = self._parse_hex_color(background_color)
        self.padding_ratio = max(0.0, min(0.4, float(padding_ratio)))
        self._session = None
        self._remove = None
        self._failed_to_initialize = False
        self._session_lock = threading.Lock()

    def process(self, source: Image.Image | str | Path) -> Image.Image:
        image = self._load_rgb(source)
        if not self.enabled:
            return image

        image.thumbnail((self.max_side, self.max_side), Image.Resampling.LANCZOS)
        session = self._get_session()
        if session is None:
            return image

        try:
            cutout = self._remove(image, session=session, decontaminate=True)
            if not isinstance(cutout, Image.Image):
                raise TypeError("El removedor de fondo no devolvió una imagen PIL")
            return self._compose_cutout(cutout)
        except Exception:
            logger.exception("Background removal failed; using the original input image")
            return image

    def _get_session(self):
        if self._session is not None:
            return self._session
        if self._failed_to_initialize:
            return None
        with self._session_lock:
            if self._session is not None:
                return self._session
            if self._failed_to_initialize:
                return None
            try:
                from rembg import new_session, remove

                self._remove = remove
                self._session = new_session(self.model_name)
            except Exception:
                self._failed_to_initialize = True
                logger.exception(
                    "Could not initialize background-removal model %s; using original images",
                    self.model_name,
                )
                return None
        return self._session

    @staticmethod
    def _load_rgb(source: Image.Image | str | Path) -> Image.Image:
        if isinstance(source, Image.Image):
            return ImageOps.exif_transpose(source).convert("RGB")
        with Image.open(source) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")

    def _compose_cutout(self, cutout: Image.Image) -> Image.Image:
        rgba = cutout.convert("RGBA")
        alpha = rgba.getchannel("A")
        visible = alpha.point(lambda value: 255 if value > 16 else 0)
        bounds = visible.getbbox()
        if bounds is None:
            raise ValueError("Background-removal mask contains no foreground")

        left, top, right, bottom = bounds
        width, height = rgba.size
        horizontal_padding = round((right - left) * self.padding_ratio)
        vertical_padding = round((bottom - top) * self.padding_ratio)
        rgba = rgba.crop((
            max(0, left - horizontal_padding),
            max(0, top - vertical_padding),
            min(width, right + horizontal_padding),
            min(height, bottom + vertical_padding),
        ))

        crop_width, crop_height = rgba.size
        padding = max(1, round(max(crop_width, crop_height) * self.padding_ratio))
        side = max(crop_width, crop_height) + 2 * padding
        background = Image.new("RGBA", (side, side), (*self.background_color, 255))
        background.alpha_composite(
            rgba,
            ((side - crop_width) // 2, (side - crop_height) // 2),
        )
        return background.convert("RGB")

    @staticmethod
    def _parse_hex_color(value: str) -> tuple[int, int, int]:
        normalized = value.strip().removeprefix("#")
        if len(normalized) != 6:
            raise ValueError("background_color must use six-digit hex, e.g. #f1f0ec")
        try:
            return tuple(int(normalized[index:index + 2], 16) for index in (0, 2, 4))
        except ValueError as error:
            raise ValueError("background_color must use six-digit hex, e.g. #f1f0ec") from error
