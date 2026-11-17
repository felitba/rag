from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class MonitoringService:
    """Persistent local request, retrieval, reranking, and LLM-usage monitor."""

    def __init__(self, database_path: str | Path, retention_days: int = 30) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.images_directory = self.database_path.parent / "monitoring_inputs"
        self.images_directory.mkdir(parents=True, exist_ok=True)
        self.retention_days = max(1, int(retention_days))
        self._initialize()
        self.prune_old_requests()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    method TEXT NOT NULL,
                    path TEXT NOT NULL,
                    input_type TEXT,
                    query_attributes_json TEXT,
                    status_code INTEGER,
                    duration_ms REAL,
                    generation_status TEXT,
                    llm_model TEXT,
                    llm_credential_source TEXT,
                    llm_credential_fingerprint TEXT,
                    retrieval_count INTEGER NOT NULL DEFAULT 0,
                    reranked_count INTEGER NOT NULL DEFAULT 0,
                    retrieved_context_precision REAL,
                    reranked_context_precision REAL,
                    context_precision_k INTEGER,
                    context_precision_relevant_count INTEGER,
                    context_precision_criteria_json TEXT,
                    context_precision_color_threshold REAL,
                    context_precision_available INTEGER NOT NULL DEFAULT 0,
                    llm_calls INTEGER NOT NULL DEFAULT 0,
                    llm_failures INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0,
                    error_type TEXT
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    request_id TEXT NOT NULL REFERENCES requests(request_id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    garment_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (request_id, stage, rank)
                );
                CREATE TABLE IF NOT EXISTS generator_outfits (
                    request_id TEXT NOT NULL REFERENCES requests(request_id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    outfit_id TEXT,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (request_id, stage, rank)
                );
                CREATE INDEX IF NOT EXISTS requests_started_at_idx ON requests(started_at);
                """
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(requests)").fetchall()}
            if "query_attributes_json" not in columns:
                db.execute("ALTER TABLE requests ADD COLUMN query_attributes_json TEXT")
            if "llm_model" not in columns:
                db.execute("ALTER TABLE requests ADD COLUMN llm_model TEXT")
            if "llm_credential_source" not in columns:
                db.execute("ALTER TABLE requests ADD COLUMN llm_credential_source TEXT")
            if "llm_credential_fingerprint" not in columns:
                db.execute("ALTER TABLE requests ADD COLUMN llm_credential_fingerprint TEXT")
            for column, column_type in (
                ("retrieved_context_precision", "REAL"),
                ("reranked_context_precision", "REAL"),
                ("context_precision_k", "INTEGER"),
                ("context_precision_relevant_count", "INTEGER"),
                ("context_precision_criteria_json", "TEXT"),
                ("context_precision_color_threshold", "REAL"),
                ("context_precision_available", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if column not in columns:
                    db.execute(f"ALTER TABLE requests ADD COLUMN {column} {column_type}")

    def begin_request(self, request_id: str, method: str, path: str) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO requests(request_id, started_at, method, path) VALUES (?, ?, ?, ?)",
                (request_id, utc_now(), method, path),
            )

    def set_request_input(self, request_id: str, input_type: str) -> None:
        if not request_id:
            return
        with self._connect() as db:
            db.execute(
                "UPDATE requests SET input_type = ? WHERE request_id = ?",
                (input_type, request_id),
            )

    def record_query_images(self, request_id: str | None, original_image, processed_image) -> None:
        """Persist the uploaded image and the exact foreground-processed image used by RAG."""
        if not request_id:
            return
        if not request_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid request id for monitoring image storage")

        from PIL import Image

        request_directory = self.images_directory / request_id
        request_directory.mkdir(parents=True, exist_ok=True)

        # Preserve the uploaded bytes and use an extension the browser can render.
        format_extensions = {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
            "GIF": ".gif",
            "BMP": ".bmp",
            "TIFF": ".tiff",
        }
        with Image.open(original_image) as opened:
            extension = format_extensions.get(opened.format, ".img")
        shutil.copyfile(original_image, request_directory / f"input{extension}")

        if isinstance(processed_image, Image.Image):
            processed_image.convert("RGB").save(
                request_directory / "processed.jpg", format="JPEG", quality=90, optimize=True
            )
        else:
            with Image.open(processed_image) as opened:
                opened.convert("RGB").save(
                    request_directory / "processed.jpg", format="JPEG", quality=90, optimize=True
                )

    def query_image_path(self, request_id: str, variant: str) -> Path | None:
        if variant not in {"input", "processed"}:
            return None
        if not request_id.replace("-", "").replace("_", "").isalnum():
            return None
        request_directory = self.images_directory / request_id
        if variant == "processed":
            candidates = [request_directory / "processed.jpg"]
        else:
            candidates = [request_directory / f"input{extension}" for extension in (
                ".jpg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".img"
            )]
        return next((path for path in candidates if path.is_file()), None)

    def set_query_attributes(self, request_id: str, attributes: dict[str, Any] | None) -> None:
        if not request_id or not attributes:
            return
        with self._connect() as db:
            db.execute(
                "UPDATE requests SET query_attributes_json = ? WHERE request_id = ?",
                (json.dumps(attributes, ensure_ascii=False), request_id),
            )

    def set_generation_status(self, request_id: str, status: str) -> None:
        if not request_id:
            return
        with self._connect() as db:
            db.execute(
                "UPDATE requests SET generation_status = ? WHERE request_id = ?",
                (status, request_id),
            )

    def set_context_precision(self, request_id: str | None, metrics: dict[str, Any]) -> None:
        if not request_id:
            return
        with self._connect() as db:
            db.execute(
                """UPDATE requests SET retrieved_context_precision = ?,
                   reranked_context_precision = ?, context_precision_k = ?,
                   context_precision_relevant_count = ?, context_precision_criteria_json = ?,
                   context_precision_color_threshold = ?, context_precision_available = ?
                   WHERE request_id = ?""",
                (
                    metrics.get("retrieved"),
                    metrics.get("reranked"),
                    metrics.get("k"),
                    metrics.get("relevant_count"),
                    json.dumps(metrics.get("criteria", []), ensure_ascii=False),
                    metrics.get("color_similarity_threshold"),
                    int(bool(metrics.get("available"))),
                    request_id,
                ),
            )

    def record_candidates(
        self,
        request_id: str | None,
        stage: str,
        matches: list[dict[str, Any]],
        catalog,
    ) -> None:
        if not request_id:
            return
        records = []
        for rank, match in enumerate(matches, start=1):
            item_id = str(match["item_id"])
            item = catalog.items.get(item_id, {})
            payload = {
                "garment_id": item_id,
                "image_url": catalog.image_url(item_id),
                "title": item.get("title") or None,
                "description": item.get("description") or None,
                "category": match.get("category") or item.get("category"),
                "color": match.get("color") or item.get("color"),
                "color_confidence": match.get("color_confidence") or item.get("color_confidence"),
                "categories": self._as_list(item.get("categories")),
                "related": self._as_list(item.get("related")),
                "metadata": {
                    key: value for key, value in item.items() if value not in (None, "")
                },
                "cosine_distance": float(match.get("distance", 0)),
                "cosine_similarity": match.get("rerank_components", {}).get(
                    "cosine_similarity", max(0.0, min(1.0, 1 - float(match.get("distance", 0))))
                ),
                "rerank_score": match.get("rerank_score"),
                "rerank_components": match.get("rerank_components"),
                "context_relevance": match.get("context_relevance"),
            }
            records.append((request_id, stage, rank, item_id, json.dumps(payload, ensure_ascii=False)))

        count_column = "retrieval_count" if stage == "retrieved" else "reranked_count"
        with self._connect() as db:
            db.execute("DELETE FROM candidates WHERE request_id = ? AND stage = ?", (request_id, stage))
            db.executemany(
                "INSERT INTO candidates(request_id, stage, rank, garment_id, payload_json) VALUES (?, ?, ?, ?, ?)",
                records,
            )
            db.execute(
                f"UPDATE requests SET {count_column} = ? WHERE request_id = ?",
                (len(records), request_id),
            )

    def record_generator_outfits(
        self,
        request_id: str | None,
        stage: str,
        outfits: list[dict[str, Any]],
        catalog,
    ) -> None:
        """Persist the outfit candidates sent to generation or its final output."""
        if not request_id:
            return
        if stage not in {"generator_input", "generator_output"}:
            raise ValueError(f"Unsupported generator monitoring stage: {stage}")

        records = []
        for rank, outfit in enumerate(outfits, start=1):
            monitored_items = []
            for view in outfit.get("items", []):
                garment_id = str(view.get("garment_id", ""))
                if not garment_id:
                    continue
                item = catalog.items.get(garment_id, {})
                generator_metadata = {
                    key: value
                    for key in ("category", "title", "description", "color", "style", "related")
                    if (value := item.get(key)) not in (None, "")
                }
                categories = item.get("catgeories") or item.get("categories")
                if categories not in (None, ""):
                    generator_metadata["categories"] = categories
                generator_metadata["garment_id"] = garment_id
                monitored_items.append({
                    "garment_id": garment_id,
                    "image_url": view.get("image_url") or catalog.image_url(garment_id),
                    "title": item.get("title") or view.get("title"),
                    "category": item.get("category") or view.get("category"),
                    "color": item.get("color") or view.get("color"),
                    "metadata": {
                        key: value for key, value in item.items() if value not in (None, "")
                    },
                    "generator_metadata": generator_metadata,
                })
            payload = {
                "outfit_id": outfit.get("outfit_id"),
                "name": outfit.get("name"),
                "description": outfit.get("description"),
                "explanation": outfit.get("explanation"),
                "style_tags": outfit.get("style_tags", []),
                "source_outfits": outfit.get("source_outfits", []),
                "matched_garment_ids": outfit.get("matched_garment_ids", []),
                "generated": bool(outfit.get("generated", False)),
                "score": outfit.get("score"),
                "items": monitored_items,
            }
            records.append((
                request_id,
                stage,
                rank,
                str(outfit.get("outfit_id") or ""),
                json.dumps(payload, ensure_ascii=False),
            ))

        with self._connect() as db:
            db.execute(
                "DELETE FROM generator_outfits WHERE request_id = ? AND stage = ?",
                (request_id, stage),
            )
            db.executemany(
                """INSERT INTO generator_outfits(request_id, stage, rank, outfit_id, payload_json)
                   VALUES (?, ?, ?, ?, ?)""",
                records,
            )

    def record_llm_call_start(
        self,
        request_id: str | None,
        model: str,
        credential_source: str | None,
        credential_fingerprint: str | None,
    ) -> None:
        if not request_id:
            return
        with self._connect() as db:
            db.execute(
                """UPDATE requests SET llm_calls = llm_calls + 1, llm_model = ?,
                   llm_credential_source = ?, llm_credential_fingerprint = ? WHERE request_id = ?""",
                (model, credential_source, credential_fingerprint, request_id),
            )

    def record_llm_call_error(self, request_id: str | None) -> None:
        if not request_id:
            return
        with self._connect() as db:
            db.execute(
                "UPDATE requests SET llm_failures = llm_failures + 1 WHERE request_id = ?",
                (request_id,),
            )

    def record_llm_usage(
        self,
        request_id: str | None,
        *,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
    ) -> None:
        if not request_id:
            return
        with self._connect() as db:
            db.execute(
                """UPDATE requests SET
                    input_tokens = input_tokens + ?,
                    cached_input_tokens = cached_input_tokens + ?,
                    output_tokens = output_tokens + ?,
                    estimated_cost_usd = estimated_cost_usd + ?
                    WHERE request_id = ?""",
                (input_tokens, cached_input_tokens, output_tokens, estimated_cost_usd, request_id),
            )

    def finish_request(
        self,
        request_id: str,
        *,
        status_code: int,
        duration_ms: float,
        error_type: str | None = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """UPDATE requests SET completed_at = ?, status_code = ?, duration_ms = ?, error_type = ?
                   WHERE request_id = ?""",
                (utc_now(), status_code, round(duration_ms, 2), error_type, request_id),
            )

    def summary(self) -> dict[str, Any]:
        since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        with self._connect() as db:
            row = db.execute(
                """SELECT
                    COUNT(*) AS requests_24h,
                    SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) AS errors_24h,
                    AVG(duration_ms) AS avg_latency_ms_24h,
                    SUM(llm_calls) AS llm_calls_24h,
                    SUM(llm_failures) AS llm_failures_24h,
                    SUM(input_tokens) AS input_tokens_24h,
                    SUM(cached_input_tokens) AS cached_input_tokens_24h,
                    SUM(output_tokens) AS output_tokens_24h,
                    SUM(estimated_cost_usd) AS estimated_cost_usd_24h
                   FROM requests WHERE started_at >= ?""",
                (since,),
            ).fetchone()
            lifetime = db.execute(
                """SELECT COUNT(*) AS total_requests, SUM(llm_calls) AS total_llm_calls,
                          SUM(input_tokens + output_tokens) AS total_tokens,
                          SUM(estimated_cost_usd) AS total_estimated_cost_usd
                   FROM requests"""
            ).fetchone()
        return {
            **dict(row),
            **dict(lifetime),
            "retention_days": self.retention_days,
            "errors_24h": row["errors_24h"] or 0,
            "llm_calls_24h": row["llm_calls_24h"] or 0,
            "llm_failures_24h": row["llm_failures_24h"] or 0,
            "input_tokens_24h": row["input_tokens_24h"] or 0,
            "cached_input_tokens_24h": row["cached_input_tokens_24h"] or 0,
            "output_tokens_24h": row["output_tokens_24h"] or 0,
            "estimated_cost_usd_24h": row["estimated_cost_usd_24h"] or 0,
            "total_llm_calls": lifetime["total_llm_calls"] or 0,
            "total_tokens": lifetime["total_tokens"] or 0,
            "total_estimated_cost_usd": lifetime["total_estimated_cost_usd"] or 0,
            "credential_usage_24h": self._credential_usage(since),
        }

    def _credential_usage(self, since: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT llm_model, llm_credential_source, llm_credential_fingerprint,
                          COUNT(*) AS request_count, SUM(llm_calls) AS llm_calls,
                          SUM(llm_failures) AS llm_failures,
                          SUM(input_tokens) AS input_tokens,
                          SUM(cached_input_tokens) AS cached_input_tokens,
                          SUM(output_tokens) AS output_tokens,
                          SUM(estimated_cost_usd) AS estimated_cost_usd
                   FROM requests
                   WHERE started_at >= ? AND llm_calls > 0
                   GROUP BY llm_model, llm_credential_source, llm_credential_fingerprint
                   ORDER BY SUM(llm_calls) DESC""",
                (since,),
            ).fetchall()
        return [
            {
                **dict(row),
                "llm_calls": row["llm_calls"] or 0,
                "llm_failures": row["llm_failures"] or 0,
                "input_tokens": row["input_tokens"] or 0,
                "cached_input_tokens": row["cached_input_tokens"] or 0,
                "output_tokens": row["output_tokens"] or 0,
                "estimated_cost_usd": row["estimated_cost_usd"] or 0,
            }
            for row in rows
        ]

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None or value == "":
            return []
        return value if isinstance(value, list) else [value]

    def list_requests(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM requests ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            request = db.execute("SELECT * FROM requests WHERE request_id = ?", (request_id,)).fetchone()
            if request is None:
                return None
            rows = db.execute(
                "SELECT stage, rank, payload_json FROM candidates WHERE request_id = ? ORDER BY stage, rank",
                (request_id,),
            ).fetchall()
            outfit_rows = db.execute(
                """SELECT stage, rank, payload_json FROM generator_outfits
                   WHERE request_id = ? ORDER BY stage, rank""",
                (request_id,),
            ).fetchall()
        result: dict[str, Any] = dict(request)
        raw_attributes = result.pop("query_attributes_json", None)
        raw_precision_criteria = result.pop("context_precision_criteria_json", None)
        result["query_attributes"] = json.loads(raw_attributes) if raw_attributes else None
        result["context_precision_criteria"] = json.loads(raw_precision_criteria) if raw_precision_criteria else []
        result["input_image_url"] = (
            f"/api/monitor/requests/{request_id}/image/input"
            if self.query_image_path(request_id, "input") else None
        )
        result["processed_image_url"] = (
            f"/api/monitor/requests/{request_id}/image/processed"
            if self.query_image_path(request_id, "processed") else None
        )
        result["retrieved"] = []
        result["reranked"] = []
        result["generator_input"] = []
        result["generator_output"] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["rank"] = row["rank"]
            result[row["stage"]].append(payload)
        for row in outfit_rows:
            payload = json.loads(row["payload_json"])
            payload["rank"] = row["rank"]
            result[row["stage"]].append(payload)
        return result

    def prune_old_requests(self) -> None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        with self._connect() as db:
            expired_ids = [
                row[0] for row in db.execute(
                    "SELECT request_id FROM requests WHERE started_at < ?", (cutoff,)
                ).fetchall()
            ]
            db.execute("DELETE FROM requests WHERE started_at < ?", (cutoff,))
        for request_id in expired_ids:
            if request_id.replace("-", "").replace("_", "").isalnum():
                shutil.rmtree(self.images_directory / request_id, ignore_errors=True)
