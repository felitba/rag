from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI


class LLM:
    """Uses a vision-capable OpenAI model through LangChain to explain retrieved looks."""

    def __init__(
        self,
        model_name: str,
        max_images_per_outfit: int = 2,
    ) -> None:
        self.max_images_per_outfit = max(0, max_images_per_outfit)
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        # LCEL pipeline: payload -> multimodal messages -> ChatOpenAI -> text.
        self.chain = RunnableLambda(self._build_messages) | self.llm | StrOutputParser()

    @staticmethod
    def _image_data_url(path: Path, max_side: int = 768) -> str:
        """Resize/encode local image so it can be sent to the OpenAI API."""
        buffer = io.BytesIO()
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((max_side, max_side))
            image.save(buffer, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _public_item_context(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "garment_id": item["garment_id"],
            "category": item.get("category", "unknown"),
            "color": item.get("color", "unknown"),
            "style": item.get("style", "unknown"),
            "price": item.get("price"),
        }

    def _build_messages(self, payload: dict[str, Any]) -> list[Any]:
        user_image = Path(payload["user_image_path"])
        outfits = payload["outfits"]
        descriptions = []

        human_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Esta es la prenda de referencia que envió el usuario. "
                    "Usala para comparar visualmente los looks del catálogo."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": self._image_data_url(user_image),
                    "detail": "high",
                },
            },
        ]

        for outfit in outfits:
            descriptions.append({
                "outfit_id": outfit["outfit_id"],
                "items": [
                    self._public_item_context(item)
                    for item in outfit["items"]
                ],
                "matched_garment_ids": outfit["matched_garment_ids"],
            })

            matched_ids = set(outfit["matched_garment_ids"])
            ordered_items = sorted(
                outfit["items"],
                key=lambda item: item["garment_id"] not in matched_ids,
            )
            selected_items = ordered_items[: self.max_images_per_outfit]

            for item in selected_items:
                image_path = Path(item["_image_abs_path"])
                human_content.append({
                    "type": "text",
                    "text": (
                        f"Imagen de una prenda del {outfit['outfit_id']}: "
                        f"{self._public_item_context(item)}."
                    ),
                })
                human_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": self._image_data_url(image_path),
                        "detail": "low",
                    },
                })

        human_content.append({
            "type": "text",
            "text": (
                "Estos son los outfits autorizados y los metadatos de sus prendas. "
                "La lista es la fuente de verdad:\n"
                + json.dumps(descriptions, ensure_ascii=False, indent=2)
                + "\nExplicá en español qué similitudes visuales encontrás y por qué "
                  "podría interesarle al usuario cada outfit. Diferenciá similitud "
                  "de compatibilidad de estilo. No inventes atributos, precios, "
                  "prendas ni IDs. No agregues outfits que no estén en la lista. "
                  "Mencioná cada outfit usando exactamente su outfit_id. Si la "
                  "evidencia visual es insuficiente, decilo. Sé breve."
            ),
        })

        return [
            SystemMessage(content=(
                "Sos un asistente de recomendaciones de moda para una tienda. "
                "Debés basarte exclusivamente en la foto de referencia, las imágenes "
                "y los datos suministrados. No afirmes que conocés un material o una "
                "textura con certeza si no puede observarse en la imagen. La selección "
                "de productos ya fue realizada por el sistema; tu tarea es explicarla, "
                "no reemplazarla."
            )),
            HumanMessage(content=human_content),
        ]

    def explain(
        self,
        user_image_path: Path,
        recommendations: list[dict[str, Any]],
    ) -> str:
        return self.chain.invoke({
            "user_image_path": str(user_image_path),
            "outfits": recommendations,
        })
