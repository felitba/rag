from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from home_made_rag.llm import image_data_url
from home_made_rag.llm_cost import log_llm_call, usage_from_message


logger = logging.getLogger(__name__)


class ExpansionError(RuntimeError):
    """Raised when the vision model cannot produce a usable expansion."""


class ObservedGarment(BaseModel):
    category: str = Field(description="Broad garment category, e.g. t-shirt, jacket, dress.")
    color: str = Field(description="Dominant observable color.")
    style: str = Field(description="Observable style, e.g. streetwear, formal, casual.")
    pattern: str | None = Field(default=None, description="Observable pattern, if any.")
    fit: str | None = Field(default=None, description="Observable fit, e.g. oversized, slim.")
    visible_details: list[str] = Field(
        default_factory=list,
        description="Short list of visible details such as zippers, distressing or prints.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the visual analysis, from 0 to 1.",
    )
    anchor_description_en: str | None = Field(
        default="",
        description="Concise English description of the user's garment for CLIP retrieval.",
    )


class HypotheticalGarment(BaseModel):
    category: str = Field(description="Broad category of the complementary garment.")
    description_en: str = Field(
        description="Concise self-contained English visual description for CLIP.",
    )
    description_es: str | None = Field(
        default="",
        description="Spanish translation of the description.",
    )
    reason_es: str | None = Field(
        default="",
        description="Short Spanish explanation of why it complements.",
    )


class ExpansionResult(BaseModel):
    observed: ObservedGarment
    complementary: list[HypotheticalGarment] = Field(
        default_factory=list,
        description="Two or three garments that complete an outfit with the user's garment.",
    )
    stylist_summary_es: str | None = Field(
        default="",
        description="Brief Spanish styling summary for the user.",
    )


SYSTEM_PROMPT = (
    "Sos un asistente de moda para una tienda de ropa. Analizá la prenda del "
    "usuario usando exclusivamente lo que se ve en la imagen y/o lo que dice el "
    "texto. No inventes marcas, materiales ni detalles que no puedan observarse. "
    "Los metadatos del catálogo son incompletos, así que priorizá el análisis "
    "visual. Devolvé únicamente el JSON estructurado solicitado."
)

USER_PROMPT = (
    "Analizá la prenda del usuario y proponé prendas complementarias.\n"
    "1. Identificá los atributos observables de la prenda del usuario: categoría, "
    "color, estilo, patrón, calce, detalles visibles y un nivel de confianza entre "
    "0 y 1. Escribí también una descripción ancla breve en inglés (máximo 30 "
    "palabras) que sirva para recuperar imágenes.\n"
    "2. Proponé exactamente {num_complements} prendas complementarias que armen un "
    "look con la prenda del usuario. No repitas la prenda del usuario. Para cada "
    "una, escribí una descripción visual breve y autocontenida en inglés (máximo "
    "30 palabras) apta para CLIP, su traducción al español y una justificación "
    "breve en español.\n"
    "3. Escribí un resumen de 2 o 3 oraciones en español para el usuario. Presentá "
    "las prendas complementarias como sugerencias de estilo, no como productos "
    "garantizados en la tienda.\n"
    "Categorías disponibles en el catálogo: {categories}."
)


class QueryExpander:
    """Vision-first HyDE-style query expansion using a low-cost model."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.2,
        num_complements: int = 3,
        image_detail: str = "low",
        image_max_side: int = 768,
        max_output_tokens: int = 800,
    ) -> None:
        self.model_name = model_name
        self.num_complements = max(1, num_complements)
        self.image_detail = image_detail
        self.image_max_side = image_max_side
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        self.structured_llm = self.llm.with_structured_output(
            ExpansionResult,
            include_raw=True,
        )

    def expand(
        self,
        image_path: str | Path | None = None,
        text: str | None = None,
        catalog_categories: list[str] | None = None,
    ) -> ExpansionResult:
        if image_path is None and not (text or "").strip():
            raise ValueError("Se necesita una imagen o un texto para expandir la consulta")

        content: list[dict[str, Any]] = []

        if text and text.strip():
            content.append({
                "type": "text",
                "text": f"Descripción del usuario: {text.strip()}",
            })

        if image_path is not None:
            content.append({
                "type": "text",
                "text": "Imagen de la prenda del usuario:",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_data_url(Path(image_path), max_side=self.image_max_side),
                    "detail": self.image_detail,
                },
            })

        categories = ", ".join(catalog_categories or []) or "sin información"
        content.append({
            "type": "text",
            "text": USER_PROMPT.format(
                num_complements=self.num_complements,
                categories=categories,
            ),
        })

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]

        logger.info(
            "[LLM] call=query_expansion model=%s image=%s text=%s",
            self.model_name,
            image_path is not None,
            bool(text and text.strip()),
        )

        try:
            result = self.structured_llm.invoke(messages)
        except Exception as exc:
            logger.error(
                "[LLM] call=query_expansion model=%s failed: %s "
                "estimated_cost_usd=0.000000",
                self.model_name,
                exc,
            )
            raise

        self._log_usage(result)

        if isinstance(result, ExpansionResult):
            parsed = result
        else:
            parsing_error = result.get("parsing_error")
            if parsing_error is not None:
                raise ExpansionError(f"No se pudo interpretar la expansión: {parsing_error}")
            parsed = result.get("parsed")

        if parsed is None:
            raise ExpansionError("El modelo no devolvió una expansión válida")

        return self._validate(parsed)

    def _log_usage(self, result: Any) -> None:
        raw = result.get("raw") if isinstance(result, dict) else None
        usage = usage_from_message(raw)
        log_llm_call(
            self.model_name,
            "query_expansion",
            usage["input_tokens"],
            usage["output_tokens"],
            usage["cached_tokens"],
        )

    def _validate(self, result: ExpansionResult) -> ExpansionResult:
        complementary = [
            garment
            for garment in result.complementary
            if garment.description_en and garment.description_en.strip()
        ]

        if len(complementary) > self.num_complements:
            complementary = complementary[: self.num_complements]

        if not complementary:
            raise ExpansionError("La expansión no devolvió prendas complementarias utilizables")

        if len(complementary) < 2:
            logger.warning(
                "[LLM] query_expansion returned only %d complementary garment(s)",
                len(complementary),
            )

        if not (result.observed.anchor_description_en or "").strip():
            result.observed.anchor_description_en = " ".join(
                part
                for part in (
                    result.observed.color,
                    result.observed.style,
                    result.observed.category,
                )
                if part
            ).strip()

        if not (result.stylist_summary_es or "").strip():
            result.stylist_summary_es = (
                "Estos looks del catálogo son una buena base para combinar con tu prenda."
            )

        result.complementary = complementary
        return result
