from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CatalogService:
    """Loads the canonical garment and outfit JSON files and checks references."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        data_dir = self.project_root / "app" / "data"

        self.garments_list: list[dict[str, Any]] = self._read_json(
            data_dir / "garments.json"
        )
        self.outfits: dict[str, list[str]] = self._read_json(
            data_dir / "outfits.json"
        )

        self.garments: dict[str, dict[str, Any]] = {}
        for garment in self.garments_list:
            garment_id = garment.get("garment_id")
            if not isinstance(garment_id, str) or not garment_id.strip():
                raise ValueError("Cada prenda debe tener un garment_id no vacío.")
            if garment_id in self.garments:
                raise ValueError(f"garment_id duplicado: {garment_id}")
            self.garments[garment_id] = garment

        for outfit_id, garment_ids in self.outfits.items():
            if not isinstance(garment_ids, list) or not garment_ids:
                raise ValueError(f"El outfit {outfit_id} debe contener prendas.")
            for garment_id in garment_ids:
                if garment_id not in self.garments:
                    raise ValueError(
                        f"El outfit {outfit_id} referencia una prenda inexistente: "
                        f"{garment_id}"
                    )

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.is_file():
            raise FileNotFoundError(f"No se encontró el archivo de catálogo: {path}")
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def resolve_image_path(self, relative_path: str) -> Path:
        """Resolve image paths but never allow them to escape catalog/."""
        raw_path = Path(relative_path)
        if raw_path.is_absolute():
            raise ValueError(f"image_path debe ser relativo: {relative_path}")

        absolute_path = (self.project_root / raw_path).resolve()
        catalog_root = (self.project_root / "catalog").resolve()
        if not absolute_path.is_relative_to(catalog_root):
            raise ValueError(
                f"La imagen debe estar dentro de catalog/: {relative_path}"
            )
        return absolute_path

    def image_url(self, relative_path: str) -> str:
        """Convert catalog/garments/item.jpg to /catalog/garments/item.jpg."""
        safe_path = Path(relative_path).as_posix()
        if not safe_path.startswith("catalog/"):
            raise ValueError(f"Ruta de imagen no válida: {relative_path}")
        return "/" + safe_path

    def validate_image_files(self) -> None:
        for garment in self.garments_list:
            image_path = self.resolve_image_path(garment["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Falta la imagen de {garment['garment_id']}: {image_path}\n"
                    "Colocá las imágenes del catálogo en catalog/garments/ y "
                    "actualizá image_path en garments.json."
                )
