import React, { useCallback, useEffect, useState } from "react";
import { API_BASE, PageLayout, useCatalog } from "./shared.jsx";

function formatNumber(value, digits = 0) {
  return new Intl.NumberFormat("es-AR", { maximumFractionDigits: digits }).format(value || 0);
}

function formatMoney(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 4,
  }).format(value || 0);
}

function formatPercent(value) {
  return value === null || value === undefined
    ? "—"
    : `${formatNumber(value * 100, 1)}%`;
}

function formatDate(value) {
  return value ? new Date(value).toLocaleString("es-AR") : "En curso";
}

function formatSeconds(milliseconds) {
  return milliseconds === null || milliseconds === undefined
    ? "—"
    : `${formatNumber(milliseconds / 1000, 1)} s`;
}

// Orden en que corren las etapas; el color acompaña al nombre, nunca lo reemplaza.
const STAGES = [
  { key: "preprocessing", label: "Quitar fondo", color: "#7a8f6a" },
  { key: "embedding", label: "Embedding CLIP", color: "#c9a45c" },
  { key: "retrieval", label: "Retrieval", color: "#5b7f95" },
  { key: "attributes", label: "Atributos de la foto", color: "#b0765a" },
  { key: "rewriting", label: "Query rewriting", color: "#8a6f9e" },
  { key: "rerank", label: "Reranking", color: "#4f8a7c" },
  { key: "outfits", label: "Orden de outfits", color: "#a8a295" },
  { key: "generation", label: "Generación", color: "#2d382d" },
  { key: "other", label: "Otros (monitor, validación, red)", color: "#d8d2c6" },
];

const STATUS_STYLES = {
  generated: { label: "Generado", color: "#50634f" },
  fallback: { label: "Fallback al catálogo", color: "#b0765a" },
  out_of_scope: { label: "Fuera de alcance", color: "#a8a295" },
};

function SegmentBar({ segments, format }) {
  const total = segments.reduce((sum, segment) => sum + segment.value, 0);
  if (!total) return null;
  return (
    <>
      <div className="monitor-bar" role="img" aria-label={segments.map((segment) => `${segment.label}: ${format(segment.value, total)}`).join(", ")}>
        {segments.map((segment) => (
          <span
            key={segment.label}
            title={`${segment.label}: ${format(segment.value, total)}`}
            style={{ width: `${(segment.value / total) * 100}%`, background: segment.color }}
          />
        ))}
      </div>
      <div className="monitor-legend">
        {segments.map((segment) => (
          <span key={segment.label}>
            <i style={{ background: segment.color }} />
            {segment.label} <strong>{format(segment.value, total)}</strong>
          </span>
        ))}
      </div>
    </>
  );
}

function stageSegments(timings, totalMs) {
  const segments = STAGES
    .filter((stage) => timings[stage.key] > 0)
    .map((stage) => ({ label: stage.label, color: stage.color, value: timings[stage.key] }));
  const measured = segments.reduce((sum, segment) => sum + segment.value, 0);
  // Lo que no está dentro de ninguna etapa medida: escrituras del monitor, validación, HTTP.
  if (totalMs && totalMs - measured > 1) {
    const other = STAGES[STAGES.length - 1];
    segments.push({ label: other.label, color: other.color, value: totalMs - measured });
  }
  return segments;
}

// Las etapas rápidas (retrieval, reranking) tardan milisegundos: en segundos se verían como "0 s".
const formatStage = (value, total) =>
  `${value < 1000 ? `${formatNumber(value, 0)} ms` : formatSeconds(value)} · ${formatNumber((value / total) * 100, 0)}%`;
const formatShare = (value, total) => `${formatNumber(value)} · ${formatNumber((value / total) * 100, 1)}%`;

function Aggregates({ summary }) {
  const statuses = (summary.generation_status_24h || []).map((row) => ({
    label: STATUS_STYLES[row.status]?.label || row.status,
    color: STATUS_STYLES[row.status]?.color || "#c9a45c",
    value: row.requests,
  }));
  const averageTimings = summary.avg_stage_timings_ms_24h || {};
  const averageSegments = stageSegments(averageTimings, null);
  return (
    <div className="monitor-aggregates">
      <section>
        <h2>Resultado de la generación · 24 h</h2>
        <p className="monitor-muted">Sobre {formatNumber(summary.ok_requests_24h)} requests con respuesta 200.</p>
        {statuses.length ? <SegmentBar segments={statuses} format={formatShare} /> : <p className="monitor-muted">Sin requests.</p>}
      </section>
      <section>
        <h2>Latencia media por etapa · 24 h</h2>
        <p className="monitor-muted">
          {summary.stage_timing_requests_24h
            ? `Promedio de ${formatNumber(summary.stage_timing_requests_24h)} requests con tiempos por etapa.`
            : "Todavía no hay requests con tiempos por etapa (se registran desde esta versión)."}
        </p>
        <SegmentBar segments={averageSegments} format={formatStage} />
      </section>
      <section>
        <h2>Por tipo de entrada · 24 h</h2>
        <table className="monitor-table">
          <thead>
            <tr><th>Entrada</th><th>Requests</th><th>Latencia p50</th><th>Latencia p95</th><th>Costo medio</th><th>Fallbacks</th></tr>
          </thead>
          <tbody>
            {(summary.by_input_type_24h || []).map((row) => (
              <tr key={row.input_type}>
                <td>{INPUT_LABELS[row.input_type] || row.input_type}</td>
                <td>{formatNumber(row.requests)}</td>
                <td>{formatSeconds(row.p50_latency_ms)}</td>
                <td>{formatSeconds(row.p95_latency_ms)}</td>
                <td>{formatMoney(row.avg_cost_usd)}</td>
                <td>{formatNumber(row.fallbacks)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section>
        <h2>Context precision media · 24 h</h2>
        <p className="monitor-muted">
          {summary.context_precision_requests_24h
            ? `Antes del reranking ${formatPercent(summary.avg_retrieved_context_precision_24h)} → después ${formatPercent(summary.avg_reranked_context_precision_24h)}, en ${formatNumber(summary.context_precision_requests_24h)} requests con foto.`
            : "Sin requests con atributos de foto."}
        </p>
        <p className="monitor-muted">
          Es una métrica proxy: llama relevante a una prenda cuando coinciden categoría y color, las mismas señales que usa el reranker para ordenar. La suba es esperable y no mide calidad percibida.
        </p>
      </section>
    </div>
  );
}

function StageTimings({ detail }) {
  if (!detail.stage_timings) return null;
  return (
    <section className="monitor-timings">
      <h3>Tiempo por etapa</h3>
      <p className="monitor-muted">Total {formatSeconds(detail.duration_ms)}.</p>
      <SegmentBar segments={stageSegments(detail.stage_timings, detail.duration_ms)} format={formatStage} />
    </section>
  );
}

function PipelineSteps({ detail }) {
  const rewriting = detail.query_rewriting;
  const attributes = detail.query_attributes;
  const trace = detail.pipeline_trace;
  const config = trace?.config;
  const hop = trace?.outfit_hop;
  const attempts = trace?.generation?.attempts || [];
  const drafts = attempts.flatMap((attempt) => attempt.drafts || []);
  const hasPhoto = Boolean(detail.input_image_url || detail.processed_image_url);
  const steps = [
    {
      name: "Entrada",
      what: "Lo que mandó la persona: foto, texto o ambos.",
      text: `${INPUT_LABELS[detail.input_type] || detail.input_type || "procesando"}${rewriting?.original ? ` · “${rewriting.original}”` : ""}`,
    },
    {
      name: "Preprocesamiento de la foto",
      what: "rembg quita el fondo, recorta la prenda y la centra sobre un fondo neutro, para parecerse a las fotos del catálogo.",
      text: !hasPhoto ? null : config && !config.image_preprocessing ? "desactivado: se usó la foto original" : "fondo removido y prenda centrada",
    },
    {
      name: "Embedding",
      what: "CLIP ViT-B/32 convierte la foto (o cada query de texto) en un vector normalizado; el catálogo está indexado con el mismo modelo.",
      text: hasPhoto ? "1 vector de imagen" : rewriting?.queries?.length ? `${rewriting.queries.length} ${rewriting.queries.length === 1 ? "vector de texto" : "vectores de texto"}` : null,
    },
    {
      name: "Atributos de la foto",
      what: "FashionCLIP estima la categoría (zero-shot) y un análisis HSV el color dominante. Alimentan el reranking; no filtran.",
      text: attributes
        ? `categoría ${attributes.category?.label || `dudosa (${attributes.category?.best_label || "s/d"})`} · color ${attributes.color?.color || "s/d"}`
        : null,
    },
    {
      name: "Query rewriting",
      what: "Un LLM reescribe el texto en queries cortas en inglés para CLIP y decide si el pedido es de ropa.",
      text: !rewriting ? null
        : !rewriting.in_scope ? "fuera de alcance: se respondió con el mensaje fijo"
        : rewriting.source !== "rewritten" ? "sin reescritura: se usó el texto original"
        : `modo ${rewriting.mode || "s/d"} · ${rewriting.queries.length} ${rewriting.queries.length === 1 ? "query" : "queries"}`,
    },
    {
      name: "Retrieval",
      what: `ChromaDB devuelve las ${config?.top_k || "top_k"} prendas más cercanas por similitud coseno${hasPhoto ? " a la foto" : " a cada query"}. Sin filtros${hasPhoto ? "" : ", salvo categoría por query"}.`,
      text: detail.retrieval_count
        ? `${formatNumber(detail.retrieval_count)} prendas candidatas`
        : null,
    },
    {
      name: "Reranking híbrido",
      what: "Reordena las mismas prendas combinando coseno, similitud de color y coincidencia de categoría. No agrega ni quita prendas.",
      text: !detail.reranked_count ? null
        : detail.context_precision_available
          ? `context precision (proxy) ${formatPercent(detail.retrieved_context_precision)} → ${formatPercent(detail.reranked_context_precision)}`
          : "sin atributos de foto: queda el orden por coseno",
    },
    {
      name: "De prendas a outfits",
      what: "Cada prenda trae los outfits reales que la contienen. Se ordenan y se conservan los primeros.",
      text: hop
        ? `${formatNumber(hop.outfits_found)} outfits encontrados · pasan ${formatNumber(hop.outfits_kept)} · regla: ${HOP_RULES[hop.rule]?.short || hop.rule}`
        : (detail.generator_input || []).length ? `${detail.generator_input.length} outfits` : null,
    },
    {
      name: "Generación",
      what: "El LLM recibe esos outfits como contexto (solo metadata, sin imágenes salvo que se active) y propone looks en JSON estructurado.",
      text: detail.generation_status
        ? `${STATUS_STYLES[detail.generation_status]?.label || detail.generation_status} · ${formatNumber(detail.llm_calls)} llamadas al LLM · ${formatMoney(detail.estimated_cost_usd)}${attempts.length > 1 ? ` · ${attempts.length} proveedores probados` : ""}`
        : null,
    },
    {
      name: "Validación de la salida",
      what: "Guardrail determinístico: cada look debe usar solo IDs del contexto y respetar las reglas de combinación. Si ninguno pasa, se devuelven los outfits recuperados.",
      text: drafts.length
        ? `${drafts.filter((draft) => draft.accepted).length} de ${drafts.length} looks propuestos aceptados`
        : detail.generation_status === "fallback" ? "ningún look válido: fallback al catálogo" : null,
    },
  ];
  return (
    <section className="monitor-pipeline">
      <h3>Pasos del pipeline</h3>
      <ol>
        {steps.map((step) => (
          <li key={step.name} className={step.text ? "" : "skipped"}>
            <strong>{step.name}</strong>
            <span className="monitor-step-result">{step.text || "no aplica en esta request"}</span>
            <span className="monitor-step-what">{step.what}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}

const HOP_RULES = {
  first_match: {
    short: "mejor prenda",
    text: "Solo foto: el outfit toma el puntaje de su prenda mejor rankeada (la primera que lo trajo). Tener varias prendas parecidas no suma.",
  },
  text_coverage: {
    short: "cobertura del texto",
    text: "Solo texto: puntaje = promedio, sobre las queries, de qué tan bien responden las prendas del outfit (mejor prenda de la categoría, o promedio del outfit si la query es general).",
  },
  photo_and_text: {
    short: "foto + texto",
    text: "Foto y texto: el puntaje de la foto y el del texto se reescalan a 0–1 (min-max entre los outfits candidatos) y se combinan con los pesos configurados.",
  },
};

const LOOK_KINDS = {
  existing: "outfit del catálogo, completo",
  trimmed: "outfit del catálogo, recortado",
  mix: "combinación nueva",
};

const REJECTION_LABELS = {
  garment_ids_not_list: "garment_ids no es una lista",
  fewer_than_two_garments: "menos de dos prendas",
  duplicate_garment_within_look: "prenda repetida dentro del look",
  source_outfit_ids_not_list: "source_outfit_ids no es una lista",
  missing_source_outfits: "no cita outfits de origen",
  duplicate_source_outfit_id: "outfit de origen repetido",
  garment_id_outside_candidates: "usa una prenda que no estaba en el contexto",
  source_outfit_id_outside_candidates: "cita un outfit que no estaba en el contexto",
  garment_not_in_cited_source_outfit: "una prenda no pertenece a los outfits citados",
  fewer_than_two_source_outfits_for_new_combination: "combinación nueva con un solo outfit de origen",
  no_contributing_source_outfits: "ningún outfit citado aporta prendas",
  fewer_than_two_contributing_source_outfits: "no mezcla prendas de al menos dos outfits",
};

function ConfigUsed({ config }) {
  if (!config) return null;
  const weights = config.rerank_weights || {};
  const onOff = (value) => (value ? "sí" : "no");
  const rows = [
    ["Quitar fondo", onOff(config.image_preprocessing)],
    ["Clasificador de categoría", onOff(config.category_classifier)],
    ["Query rewriting", onOff(config.query_rewriting)],
    ["top_k (prendas recuperadas)", config.top_k],
    ["Pesos del reranking", `coseno ${weights.cosine_similarity} · color ${weights.color} · categoría ${weights.category}`],
    ["Color: tono / cobertura", `${config.color_tone_weight} / ${config.color_coverage_weight}`],
    ["Pesos foto / texto", `${config.photo_weight} / ${config.text_weight}`],
    ["Generador", config.generator ? config.generator_model : "desactivado"],
    ["Outfits al generador (top_n)", config.generator ? config.top_n_outfits : config.max_outfits],
    ["Looks pedidos", config.max_generated_outfits],
    ["Generador ve imágenes", onOff(config.generator_uses_images)],
  ];
  return (
    <section className="monitor-pipeline">
      <h3>Configuración con la que corrió</h3>
      <p className="monitor-muted">Guardada por request: sirve para comparar versiones (ablation) sin depender del config actual.</p>
      <dl className="monitor-config">
        {rows.map(([label, value]) => (
          <div key={label}><dt>{label}</dt><dd>{value ?? "—"}</dd></div>
        ))}
      </dl>
    </section>
  );
}

// Qué componentes del reranking existían para esta consulta y con qué peso real.
function rerankWeights(detail) {
  const configured = detail.pipeline_trace?.config?.rerank_weights;
  if (!configured) return null;
  const attributes = detail.query_attributes;
  const available = {
    cosine_similarity: true,
    color: Boolean(attributes?.color?.color),
    category: Boolean(attributes?.category?.label),
  };
  const total = Object.keys(configured).reduce((sum, key) => sum + (available[key] ? configured[key] : 0), 0);
  const effective = Object.fromEntries(
    Object.keys(configured).map((key) => [key, available[key] && total ? configured[key] / total : 0]),
  );
  return { configured, available, effective };
}

function RerankExplain({ detail }) {
  const weights = rerankWeights(detail);
  const config = detail.pipeline_trace?.config;
  const before = detail.retrieved || [];
  const after = detail.reranked || [];
  const previous = Object.fromEntries(before.map((item) => [item.garment_id, item.rank]));
  const moved = after.filter((item) => previous[item.garment_id] !== item.rank).length;
  const topBefore = new Set(before.slice(0, 5).map((item) => item.garment_id));
  const newInTop = after.slice(0, 5).filter((item) => !topBefore.has(item.garment_id)).length;
  return (
    <section className="monitor-pipeline">
      <h3>Cómo se calcula el reranking</h3>
      {weights ? (
        <>
          <p className="monitor-formula">
            puntaje = ( {formatNumber(weights.configured.cosine_similarity, 2)} × coseno
            {" + "}{formatNumber(weights.configured.color, 2)} × color
            {" + "}{formatNumber(weights.configured.category, 2)} × categoría ) ÷ suma de los pesos disponibles
          </p>
          <ul className="monitor-explain">
            <li><strong>coseno</strong>: similitud CLIP entre la consulta y la prenda (1 − distancia), entre 0 y 1.</li>
            <li><strong>color</strong>: {config?.color_tone_weight} × tono (distancia CIE Lab ΔE76, escala {config?.color_delta_e_scale}) + {config?.color_coverage_weight} × cobertura (qué parte de la prenda ocupa ese color). {weights.available.color ? "Disponible en esta consulta." : "No disponible: la consulta no tiene color."}</li>
            <li><strong>categoría</strong>: 1 si la categoría de la prenda coincide con la detectada en la foto, 0 si no. {weights.available.category ? "Disponible en esta consulta." : "No disponible: sin foto o categoría dudosa."}</li>
            <li>Un componente que falta se descarta y los pesos se renormalizan. Pesos reales en esta consulta: coseno {formatPercent(weights.effective.cosine_similarity)}, color {formatPercent(weights.effective.color)}, categoría {formatPercent(weights.effective.category)}.</li>
          </ul>
        </>
      ) : (
        <p className="monitor-muted">Esta request es anterior al registro de los pesos; los componentes de cada prenda igual figuran abajo.</p>
      )}
      <p className="monitor-muted">
        Efecto en esta request: {formatNumber(moved)} de {formatNumber(after.length)} prendas cambiaron de posición; {formatNumber(newInTop)} de las 5 primeras son nuevas en el top 5.
      </p>
    </section>
  );
}

function OutfitHop({ detail }) {
  const hop = detail.pipeline_trace?.outfit_hop;
  const config = detail.pipeline_trace?.config;
  if (!hop) {
    return <p className="monitor-muted">Esta request es anterior al registro de este paso. Enviá una consulta nueva para verlo.</p>;
  }
  const withText = hop.rule !== "first_match";
  return (
    <section className="monitor-pipeline">
      <h3>De prendas a outfits</h3>
      <p className="monitor-muted">
        Las {formatNumber(detail.reranked_count)} prendas rankeadas pertenecen a {formatNumber(hop.outfits_found)} outfits reales del catálogo.
        {" "}Pasan los primeros {formatNumber(hop.outfits_kept)}{hop.limit ? ` (límite ${hop.limit})` : ""}: son el contexto del generador, o el resultado final si el generador está apagado o falla.
      </p>
      <p className="monitor-explain-rule"><strong>Regla de orden — {HOP_RULES[hop.rule]?.short || hop.rule}.</strong> {HOP_RULES[hop.rule]?.text}
        {hop.rule === "photo_and_text" && config ? ` Pesos: foto ${config.photo_weight}, texto ${config.text_weight}.` : ""}
      </p>
      <table className="monitor-table">
        <thead>
          <tr>
            <th>#</th><th>Outfit</th><th>Puntaje</th>
            {withText && <><th>Foto</th><th>Texto</th></>}
            <th>Prendas</th><th>Prendas del ranking que lo trajeron (posición · puntaje)</th><th></th>
          </tr>
        </thead>
        <tbody>
          {hop.outfits.map((outfit, index) => (
            <tr key={outfit.outfit_id} className={outfit.kept ? "" : "monitor-row-dropped"}>
              <td>{outfit.kept ? index + 1 : "—"}</td>
              <td>{outfit.outfit_id}</td>
              <td>{formatNumber(outfit.score, 3)}</td>
              {withText && <><td>{outfit.photo_score == null ? "—" : formatNumber(outfit.photo_score, 3)}</td><td>{outfit.text_score == null ? "—" : formatNumber(outfit.text_score, 3)}</td></>}
              <td>{formatNumber(outfit.garment_count)}</td>
              <td>
                <span className="monitor-hop-matches">
                  {(outfit.matches || []).map((match) => (
                    <span key={match.garment_id} title={`${match.garment_id} · ${match.category || ""} · coseno ${formatNumber(match.cosine_similarity, 3)}`}>
                      <img src={`${API_BASE}${match.image_url}`} alt="" loading="lazy" />
                      #{match.rank} · {formatNumber(match.score, 3)}
                    </span>
                  ))}
                </span>
              </td>
              <td><span className={`monitor-chip ${outfit.kept ? "status-generated" : ""}`}>{outfit.kept ? "pasa" : "queda afuera"}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
      {hop.outfits_found > hop.outfits.length && (
        <p className="monitor-muted">Se muestran los que pasan y los 10 siguientes; hay {formatNumber(hop.outfits_found - hop.outfits.length)} más.</p>
      )}
    </section>
  );
}

function GenerationTrace({ detail }) {
  const attempts = detail.pipeline_trace?.generation?.attempts;
  if (!attempts) {
    return detail.pipeline_trace
      ? <p className="monitor-muted">En esta request no se llamó al generador.</p>
      : <p className="monitor-muted">Esta request es anterior al registro de los intentos del generador.</p>;
  }
  return (
    <section className="monitor-pipeline">
      <h3>Qué propuso el modelo y qué aceptó el guardrail</h3>
      <p className="monitor-muted">
        Un intento por proveedor, en orden. Si un intento falla o ningún look pasa la validación, se prueba el siguiente; si todos fallan, se devuelven los outfits recuperados.
      </p>
      {attempts.map((attempt, index) => (
        <div className="monitor-attempt" key={index}>
          <p>
            <strong>Intento {index + 1} · {attempt.provider} · {attempt.model}</strong>{" "}
            <span className={`monitor-chip ${attempt.status === "ok" ? "status-generated" : "status-error"}`}>{attempt.status === "ok" ? "usado" : "descartado"}</span>
            {attempt.drafts_proposed !== undefined && <span className="monitor-muted"> {formatNumber(attempt.drafts_proposed)} looks propuestos</span>}
          </p>
          {attempt.error && <p className="monitor-muted">Motivo: {attempt.error}</p>}
          {attempt.drafts?.length > 0 && (
            <table className="monitor-table">
              <thead><tr><th>#</th><th>Look propuesto</th><th>Prendas</th><th>Outfits citados</th><th>Veredicto</th></tr></thead>
              <tbody>
                {attempt.drafts.map((draft, position) => (
                  <tr key={position}>
                    <td>{position + 1}</td>
                    <td>{draft.name || "sin nombre"}</td>
                    <td>{formatNumber(draft.garment_ids?.length)}</td>
                    <td>{(draft.source_outfits || draft.cited_source_outfits || []).join(", ") || "—"}</td>
                    <td>
                      <span className={`monitor-chip ${draft.accepted ? "status-generated" : "status-error"}`}>{draft.accepted ? "aceptado" : "rechazado"}</span>
                      {" "}{draft.accepted ? (LOOK_KINDS[draft.look_kind] || "") : (REJECTION_LABELS[draft.reason] || draft.reason || "")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      ))}
    </section>
  );
}

function RewritingTrace({ rewriting }) {
  const queries = rewriting.queries || [];
  const ranking = rewriting.ranking || [];
  const searched = queries.some((query) => query.match_count !== undefined);
  return (
    <section className="monitor-rewriting">
      <h3>Query rewriting</h3>
      <div className="monitor-rewriting-facts">
        <span>texto original: <strong>"{rewriting.original}"</strong></span>
        {rewriting.mode && (
          <span>modo: <strong>{rewriting.mode}</strong> ({rewriting.has_photo ? "hay foto: no se agregan prendas" : "solo texto: se descompone en prendas"})</span>
        )}
        {rewriting.owned_category && <span>categoría de la foto, excluida de las queries: <strong>{rewriting.owned_category}</strong></span>}
        {rewriting.model && <span>modelo: {rewriting.model}</span>}
        {rewriting.input_tokens !== undefined && (
          <span>{formatNumber(rewriting.input_tokens)} input / {formatNumber(rewriting.output_tokens)} output tokens · {formatMoney(rewriting.estimated_cost_usd)} · {formatNumber(rewriting.duration_ms)} ms</span>
        )}
      </div>
      {!rewriting.in_scope && <p className="monitor-rewriting-note">Fuera de alcance: el modelo marcó que el texto no es un pedido de ropa. No se buscó en el catálogo.</p>}
      {rewriting.source === "original_text" && (
        <p className="monitor-rewriting-note">Sin reescritura: se buscó con el texto original.{rewriting.error ? ` Error: ${rewriting.error}` : ""}</p>
      )}
      {queries.length > 0 && (
        <table className="monitor-table">
          <thead>
            <tr>
              <th>#</th><th>Query para CLIP</th><th>Categoría</th>
              {searched && <><th>Filtro</th><th>Prendas</th><th>Más parecidas (similitud)</th></>}
            </tr>
          </thead>
          <tbody>
            {queries.map((query, index) => (
              <tr key={index}>
                <td>{index + 1}</td>
                <td>{query.text}</td>
                <td>{query.category || "cualquiera"}</td>
                {searched && (
                  <>
                    <td>{query.category_filter ? "por categoría" : "sin filtro"}</td>
                    <td>{formatNumber(query.match_count)}</td>
                    <td>{(query.top_matches || []).map((match) => `${match.title || match.category} (${formatNumber(match.similarity, 3)})`).join(" · ")}</td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {queries.length > 0 && !searched && rewriting.has_photo && (
        <p className="monitor-rewriting-note">Con foto, las queries no buscan prendas nuevas: solo ordenan los outfits que trajo la foto.</p>
      )}
      {ranking.length > 0 && (
        <>
          <h4>Cómo quedó el orden de los outfits</h4>
          <table className="monitor-table">
            <thead>
              <tr>
                <th>#</th><th>Outfit</th><th>Score final</th>
                {rewriting.has_photo && <th>Foto</th>}
                <th>Texto</th><th>Prenda que mejor responde a cada query (similitud)</th>
              </tr>
            </thead>
            <tbody>
              {ranking.map((outfit, index) => (
                <tr key={outfit.outfit_id}>
                  <td>{index + 1}</td>
                  <td>{outfit.outfit_id}</td>
                  <td>{formatNumber(outfit.score, 3)}</td>
                  {rewriting.has_photo && <td>{formatNumber(outfit.photo_score, 3)}</td>}
                  <td>{formatNumber(outfit.text_score, 3)}</td>
                  <td>
                    {(outfit.matches || []).map((match, position) => (
                      <div key={position}>
                        {position + 1}. {match.garment_id ? `${match.title || match.category} (${formatNumber(match.best_similarity, 3)})` : "ninguna prenda de esa categoría"}
                        {match.rule === "promedio de todas las prendas" && ` · promedio del outfit ${formatNumber(match.score, 3)}`}
                      </div>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="monitor-rewriting-note">
            Texto = promedio sobre las queries. Una query con categoría toma la mejor prenda de esa categoría; una query general promedia todas las prendas del outfit.
            {rewriting.has_photo && " Score final = foto y texto reescalados a 0–1 y combinados con los pesos configurados."}
          </p>
        </>
      )}
      {rewriting.system_message && (
        <details className="monitor-rewriting-prompt">
          <summary>Ver el prompt enviado al modelo</summary>
          <pre>{rewriting.system_message}</pre>
        </details>
      )}
    </section>
  );
}

// La cuenta completa del puntaje de una prenda, con los pesos de esa request.
function rerankArithmetic(item, weights) {
  const parts = [
    ["cosine_similarity", item.rerank_components.cosine_similarity],
    ["color", item.rerank_components.color],
    ["category", item.rerank_components.category],
  ].filter(([, value]) => value !== null && value !== undefined);
  const total = parts.reduce((sum, [key]) => sum + weights[key], 0);
  const terms = parts.map(([key, value]) => `${formatNumber(weights[key], 2)} × ${formatNumber(value, 3)}`).join(" + ");
  return `(${terms}) ÷ ${formatNumber(total, 2)} = ${formatNumber(item.rerank_score, 3)}`;
}

function CandidateList({ title, candidates, stage, previousRanks, weights }) {
  return (
    <section className="monitor-stage">
      <h3>{title}<span>{candidates.length} prendas</span></h3>
      {!candidates.length ? <p className="monitor-muted">Sin candidatos registrados.</p> : (
        <div className="monitor-candidates">
          {candidates.map((item, index) => (
            <article className="monitor-candidate" key={`${stage}-${item.garment_id}-${index}`}>
              <img src={`${API_BASE}${item.image_url}`} alt={item.title || item.garment_id} loading="lazy" />
              <div className="monitor-candidate-content">
                <div className="monitor-candidate-title">
                  <strong>
                    #{item.rank || index + 1} · {item.title || item.garment_id}
                    {previousRanks && previousRanks[item.garment_id] !== undefined && previousRanks[item.garment_id] !== item.rank && (
                      <span className={`monitor-rank-change ${previousRanks[item.garment_id] > item.rank ? "up" : "down"}`}>
                        {previousRanks[item.garment_id] > item.rank ? "▲" : "▼"} {Math.abs(previousRanks[item.garment_id] - item.rank)} (antes #{previousRanks[item.garment_id]})
                      </span>
                    )}
                  </strong>
                  <span>{item.category || "Categoría desconocida"} · {item.color || "Color desconocido"}</span>
                </div>
                <div className="monitor-scores">
                  {item.cosine_similarity !== null && item.cosine_similarity !== undefined && (
                    <span>coseno {formatNumber(item.cosine_similarity, 4)}</span>
                  )}
                  {item.rerank_score !== null && item.rerank_score !== undefined && (
                    <span>rerank {formatNumber(item.rerank_score, 4)}</span>
                  )}
                  {item.color_confidence !== null && item.color_confidence !== undefined && (
                    <span>color píxeles {formatNumber(item.color_confidence * 100, 1)}%</span>
                  )}
                </div>
                {item.context_relevance && item.context_relevance.relevant !== null && (
                  <div className={`monitor-relevance ${item.context_relevance.relevant ? "relevant" : "not-relevant"}`}>
                    {item.context_relevance.relevant ? "Relevante" : "No relevante"} según atributos
                    {item.context_relevance.category_match !== null && ` · categoría ${item.context_relevance.category_match ? "coincide" : "no coincide"}`}
                    {item.context_relevance.color_similarity !== null && ` · color ${formatPercent(item.context_relevance.color_similarity)}`}
                  </div>
                )}
                {item.rerank_components && (
                  <div className="monitor-components">
                    Componentes: coseno {formatNumber(item.rerank_components.cosine_similarity, 3)},
                    similitud de color {item.rerank_components.color === null || item.rerank_components.color === undefined ? "s/d" : formatNumber(item.rerank_components.color, 3)},
                    (tono {item.rerank_components.color_tone_similarity === null || item.rerank_components.color_tone_similarity === undefined ? "s/d" : formatNumber(item.rerank_components.color_tone_similarity, 3)},
                    similitud de cobertura {item.rerank_components.color_coverage_similarity === null || item.rerank_components.color_coverage_similarity === undefined ? "s/d" : formatNumber(item.rerank_components.color_coverage_similarity, 3)}),
                    categoría {item.rerank_components.category ?? "s/d"}
                  </div>
                )}
                {weights && item.rerank_components && (
                  <div className="monitor-arithmetic">{rerankArithmetic(item, weights.configured)}</div>
                )}
                {item.description && <p className="monitor-description">{item.description}</p>}
                {(item.categories?.length > 0 || item.related?.length > 0) && (
                  <p className="monitor-tags">
                    {[...(item.categories || []), ...(item.related || [])].join(" · ")}
                  </p>
                )}
                <details>
                  <summary>Metadata completa</summary>
                  <pre>{JSON.stringify(item.metadata, null, 2)}</pre>
                </details>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function GeneratorOutfitList({ title, outfits, stage }) {
  return (
    <section className="monitor-generator-stage">
      <h3>{title}<span>{outfits.length} outfits</span></h3>
      {!outfits.length ? <p className="monitor-muted">Sin outfits en esta etapa. Las requests anteriores a esta actualización no tienen este registro.</p> : (
        <div className="monitor-generator-outfits">
          {outfits.map((outfit, index) => (
            <article className="monitor-generator-outfit" key={`${stage}-${outfit.outfit_id}-${index}`}>
              <header>
                <strong>#{outfit.rank || index + 1} · {outfit.name || outfit.outfit_id}</strong>
                <span>{outfit.generated ? "Generado" : "Catálogo"}</span>
              </header>
              <p className="monitor-generator-ids">
                ID: {outfit.outfit_id || "s/d"}
                {outfit.score !== null && outfit.score !== undefined && ` · score ${formatNumber(outfit.score, 4)}`}
              </p>
              {outfit.description && <p className="monitor-description">{outfit.description}</p>}
              {outfit.explanation && <p className="monitor-generator-explanation">{outfit.explanation}</p>}
              {outfit.look_kind && <p className="monitor-generator-ids">Tipo de look: {outfit.look_kind}{outfit.omitted_garment_ids?.length > 0 && ` · prendas omitidas: ${outfit.omitted_garment_ids.join(", ")}`}</p>}
              {outfit.changes_from_source && <p className="monitor-generator-explanation">Motivo según el modelo: {outfit.changes_from_source}</p>}
              {outfit.matched_garment_ids?.length > 0 && (
                <p className="monitor-generator-ids">Prendas que coincidieron con la consulta: {outfit.matched_garment_ids.join(", ")}</p>
              )}
              {outfit.source_outfits?.length > 0 && (
                <p className="monitor-generator-ids">Outfits fuente: {outfit.source_outfits.join(", ")}</p>
              )}
              <div className="monitor-generator-items">
                {(outfit.items || []).map((item) => (
                  <article className="monitor-generator-item" key={item.garment_id}>
                    <img src={`${API_BASE}${item.image_url}`} alt={item.title || item.garment_id} loading="lazy" />
                    <div>
                      <strong>{item.title || item.garment_id}</strong>
                      <span>{item.garment_id} · {item.category || "Categoría desconocida"} · {item.color || "Color desconocido"}</span>
                      <details>
                        <summary>Metadata completa del catálogo</summary>
                        <pre>{JSON.stringify(item.metadata || {}, null, 2)}</pre>
                      </details>
                      <details>
                        <summary>Campos enviados al generador</summary>
                        <pre>{JSON.stringify(item.generator_metadata || {}, null, 2)}</pre>
                      </details>
                    </div>
                  </article>
                ))}
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

const INPUT_LABELS = { image: "Foto", text: "Texto", "image+text": "Foto + texto" };

const FILTERS = [
  { key: "all", label: "Todas", test: () => true },
  { key: "image", label: "Foto", test: (request) => request.input_type === "image" },
  { key: "text", label: "Texto", test: (request) => request.input_type === "text" },
  { key: "image+text", label: "Foto + texto", test: (request) => request.input_type === "image+text" },
  { key: "fallback", label: "Fallback", test: (request) => request.generation_status === "fallback" },
  { key: "error", label: "Errores", test: (request) => request.status_code >= 400 },
];

function requestText(request) {
  // La lista trae el JSON crudo; el detalle ya lo trae parseado.
  if (request.query_rewriting?.original) return request.query_rewriting.original;
  if (!request.query_rewriting_json) return null;
  try {
    return JSON.parse(request.query_rewriting_json).original || null;
  } catch {
    return null;
  }
}

function requestTitle(request) {
  const text = requestText(request);
  if (text) return `“${text}”`;
  if (request.input_type === "image") return "Solo foto";
  return request.input_type ? "Texto sin registrar" : "Procesando…";
}

function StatusChip({ request }) {
  const failed = request.status_code >= 400;
  const status = failed ? "error" : request.generation_status || "pending";
  const label = failed
    ? `Error ${request.status_code}`
    : STATUS_STYLES[request.generation_status]?.label || request.generation_status || "En curso";
  return <span className={`monitor-chip status-${status}`}>{label}</span>;
}

function formatTime(value) {
  return value ? new Date(value).toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit" }) : "";
}

export default function MonitoringDashboard() {
  const [summary, setSummary] = useState(null);
  const [requests, setRequests] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");
  const [tab, setTab] = useState("summary");
  const catalog = useCatalog();

  const refresh = useCallback(async () => {
    try {
      const [summaryResponse, requestsResponse] = await Promise.all([
        fetch(`${API_BASE}/api/monitor/summary`),
        fetch(`${API_BASE}/api/monitor/requests?limit=100`),
      ]);
      if (!summaryResponse.ok || !requestsResponse.ok) throw new Error("No se pudo leer el monitor.");
      const [summaryData, requestsData] = await Promise.all([
        summaryResponse.json(),
        requestsResponse.json(),
      ]);
      setSummary(summaryData);
      setRequests(requestsData.requests || []);
      // La primera vez, abrir la request más reciente para que el detalle no quede vacío.
      setSelectedId((current) => current || requestsData.requests?.[0]?.request_id || null);
      setError("");
    } catch (err) {
      setError(err.message || "No se pudo conectar con el monitor.");
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return undefined;
    }
    let cancelled = false;
    const loadDetail = () => fetch(`${API_BASE}/api/monitor/requests/${selectedId}`)
      .then((response) => {
        if (!response.ok) throw new Error("No se encontró el detalle de esta request.");
        return response.json();
      })
      .then((data) => { if (!cancelled) setDetail(data); })
      .catch((err) => { if (!cancelled) setError(err.message); });
    loadDetail();
    const timer = window.setInterval(loadDetail, 5000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [selectedId]);

  const activeFilter = FILTERS.find((item) => item.key === filter) || FILTERS[0];
  const visibleRequests = requests.filter(activeFilter.test);
  // En el orden en que corre el pipeline.
  const tabs = detail ? [
    { key: "summary", label: "Resumen" },
    { key: "input", label: "1 · Entrada y consulta" },
    { key: "retrieval", label: "2 · Retrieval y reranking", count: detail.reranked?.length },
    { key: "outfits", label: "3 · Outfits", count: detail.pipeline_trace?.outfit_hop?.outfits_kept ?? detail.generator_input?.length },
    { key: "generator", label: "4 · Generador", count: detail.generator_output?.length },
  ] : [];
  const previousRanks = detail ? Object.fromEntries((detail.retrieved || []).map((item) => [item.garment_id, item.rank])) : {};
  const activeTab = tabs.some((item) => item.key === tab) ? tab : "summary";

  return (
    <PageLayout current="monitor" catalog={catalog}>
    <div className="monitor-page">
      <header className="monitor-header">
        <div>
          <h1>Monitor del pipeline</h1>
          <p>Qué pasó en cada búsqueda: tiempos, retrieval, generación y costo. Últimas 24 h.</p>
        </div>
      </header>

      {error && <div className="monitor-error">{error}</div>}

      <section className="monitor-metrics">
        <article><span>Requests</span><strong>{formatNumber(summary?.requests_24h)}</strong><small>{formatNumber(summary?.errors_24h)} con error</small></article>
        <article><span>Latencia p50</span><strong>{formatSeconds(summary?.p50_latency_ms_24h)}</strong><small>p95 {formatSeconds(summary?.p95_latency_ms_24h)} · media {formatSeconds(summary?.avg_latency_ms_24h)}</small></article>
        <article><span>Costo por request</span><strong>{formatMoney(summary?.avg_cost_per_request_usd_24h)}</strong><small>Total {formatMoney(summary?.estimated_cost_usd_24h)}</small></article>
        <article><span>Tokens</span><strong>{formatNumber((summary?.input_tokens_24h || 0) + (summary?.output_tokens_24h || 0))}</strong><small>{formatNumber(summary?.input_tokens_24h)} input · {formatNumber(summary?.output_tokens_24h)} output</small></article>
        <article><span>Llamadas al LLM</span><strong>{formatNumber(summary?.llm_calls_24h)}</strong><small>{formatNumber(summary?.llm_failures_24h)} fallidas</small></article>
      </section>

      {summary && <Aggregates summary={summary} />}

      <div className="monitor-layout">
        <section className="monitor-request-panel">
          <div className="monitor-section-heading">
            <h2>Requests</h2>
            <button type="button" onClick={refresh}>Actualizar</button>
          </div>
          <div className="monitor-filters" role="group" aria-label="Filtrar requests">
            {FILTERS.map((item) => (
              <button
                type="button"
                key={item.key}
                className={item.key === filter ? "active" : ""}
                aria-pressed={item.key === filter}
                onClick={() => setFilter(item.key)}
              >
                {item.label} <span>{requests.filter(item.test).length}</span>
              </button>
            ))}
          </div>
          {!visibleRequests.length ? (
            <p className="monitor-muted">
              {requests.length ? "Ninguna request coincide con este filtro." : "Todavía no hay requests. Enviá una consulta desde el asistente."}
            </p>
          ) : (
            <div className="monitor-requests">
              {visibleRequests.map((request) => (
                <button
                  type="button"
                  className={`monitor-request-row ${selectedId === request.request_id ? "selected" : ""}`}
                  key={request.request_id}
                  onClick={() => { setSelectedId(request.request_id); setTab("summary"); }}
                >
                  {request.input_type?.startsWith("image") ? (
                    <img
                      className="monitor-request-thumb"
                      src={`${API_BASE}/api/monitor/requests/${request.request_id}/image/input`}
                      alt=""
                      loading="lazy"
                      onError={(event) => { event.currentTarget.style.visibility = "hidden"; }}
                    />
                  ) : <span className="monitor-request-thumb text" aria-hidden="true">Aa</span>}
                  <span className="monitor-request-body">
                    <strong>{requestTitle(request)}</strong>
                    <span className="monitor-request-meta">
                      {formatTime(request.started_at)} · {INPUT_LABELS[request.input_type] || "—"} · {formatSeconds(request.duration_ms)} · {formatMoney(request.estimated_cost_usd)}
                    </span>
                    <StatusChip request={request} />
                  </span>
                </button>
              ))}
            </div>
          )}
        </section>

        <section className="monitor-detail-panel">
          {!detail ? <div className="monitor-empty">Seleccioná una request para ver su recorrido por el pipeline.</div> : (
            <>
              <div className="monitor-detail-heading">
                <div>
                  <p className="eyebrow">{(INPUT_LABELS[detail.input_type] || "Request").toUpperCase()} · {formatDate(detail.started_at)}</p>
                  <h2>{requestTitle(detail)}</h2>
                  <div className="monitor-detail-chips">
                    <StatusChip request={detail} />
                    <span className="monitor-chip">{formatSeconds(detail.duration_ms)}</span>
                    <span className="monitor-chip">{formatMoney(detail.estimated_cost_usd)}</span>
                    <span className="monitor-chip">{formatNumber(detail.input_tokens)} + {formatNumber(detail.output_tokens)} tokens</span>
                    {detail.llm_model && <span className="monitor-chip">{formatNumber(detail.llm_calls)} × {detail.llm_model}</span>}
                    {detail.context_precision_available ? (
                      <span className="monitor-chip">CP proxy {formatPercent(detail.retrieved_context_precision)} → {formatPercent(detail.reranked_context_precision)}</span>
                    ) : null}
                  </div>
                </div>
                <code className="monitor-request-id" title="ID de la request">{detail.request_id.slice(0, 12)}</code>
              </div>

              <div className="monitor-tabs" role="tablist">
                {tabs.map((item) => (
                  <button
                    type="button"
                    role="tab"
                    key={item.key}
                    aria-selected={item.key === activeTab}
                    className={item.key === activeTab ? "active" : ""}
                    onClick={() => setTab(item.key)}
                  >
                    {item.label}{item.count ? <span>{item.count}</span> : null}
                  </button>
                ))}
              </div>

              {activeTab === "summary" && (
                <>
                  <StageTimings detail={detail} />
                  <PipelineSteps detail={detail} />
                  <ConfigUsed config={detail.pipeline_trace?.config} />
                  {(detail.llm_credential_source || detail.llm_credential_fingerprint) && (
                    <p className="monitor-muted monitor-credential-line">
                      Credencial: {detail.llm_credential_source || "s/d"}{detail.llm_credential_fingerprint ? ` · huella ${detail.llm_credential_fingerprint}` : ""}
                    </p>
                  )}
                </>
              )}

              {activeTab === "input" && (
                <>
                  {(detail.input_image_url || detail.processed_image_url) ? (
                    <section className="monitor-query-images">
                      <h3>Foto enviada y foto que se usó para buscar</h3>
                      <div className="monitor-query-image-grid">
                        <figure>
                          <figcaption>Input del usuario</figcaption>
                          {detail.input_image_url
                            ? <img src={`${API_BASE}${detail.input_image_url}`} alt="Imagen original enviada por el usuario" />
                            : <div className="monitor-query-image-empty">Imagen no disponible</div>}
                        </figure>
                        <figure>
                          <figcaption>Después de quitar el fondo (esta es la que ve CLIP)</figcaption>
                          {detail.processed_image_url
                            ? <img src={`${API_BASE}${detail.processed_image_url}`} alt="Imagen usada por el retrieval, después del procesamiento de fondo" />
                            : <div className="monitor-query-image-empty">Imagen procesada no disponible</div>}
                        </figure>
                      </div>
                    </section>
                  ) : <p className="monitor-muted">Consulta sin foto: no hay preprocesamiento ni atributos de imagen.</p>}
                  {detail.query_attributes && (
                    <section className="monitor-pipeline">
                      <h3>Atributos estimados de la foto</h3>
                      <div className="monitor-query-attributes">
                        <span>categoría <strong>{detail.query_attributes.category?.label || (detail.query_attributes.category?.uncertain ? `dudosa (${detail.query_attributes.category.best_label || "sin candidata"})` : "s/d")}</strong></span>
                        {detail.query_attributes.category?.alternatives?.length > 0 && (
                          <span>{detail.query_attributes.category.uncertain ? "candidatas" : "alternativas"} {detail.query_attributes.category.alternatives.map((item) => `${item.label} (${formatNumber(item.similarity, 3)})`).join(" · ")}</span>
                        )}
                        {detail.query_attributes.category?.similarity !== undefined && (
                          <span>similitud {formatNumber(detail.query_attributes.category.similarity, 3)} · margen {formatNumber(detail.query_attributes.category.margin, 3)} (mínimo {formatNumber(detail.query_attributes.category.minimum_margin, 3)})</span>
                        )}
                        <span>color <strong>{detail.query_attributes.color?.color || "s/d"}</strong></span>
                        <span>cobertura del color {formatNumber((detail.query_attributes.color?.color_confidence || 0) * 100, 1)}%</span>
                      </div>
                      <p className="monitor-muted monitor-credential-line">
                        La categoría es la de mayor similitud FashionCLIP; si el margen sobre la segunda es menor al mínimo se marca dudosa y no se usa. El color es el dominante en HSV sobre los píxeles de la prenda.
                      </p>
                    </section>
                  )}
                  {detail.query_rewriting
                    ? <RewritingTrace rewriting={detail.query_rewriting} />
                    : <p className="monitor-muted">Sin texto: no hubo query rewriting.</p>}
                </>
              )}

              {activeTab === "retrieval" && (
                <>
                  <RerankExplain detail={detail} />
                  <section className="monitor-context-precision">
                    <strong>Context Precision (AP@k proxy)</strong>
                    {detail.context_precision_available ? (
                      <>
                        <span>Antes: {formatPercent(detail.retrieved_context_precision)} · después: {formatPercent(detail.reranked_context_precision)}</span>
                        <span>{formatNumber(detail.context_precision_relevant_count)} prendas relevantes de {formatNumber(detail.context_precision_k)} · criterio: {(detail.context_precision_criteria || []).join(" + ")}</span>
                        {(detail.context_precision_criteria || []).includes("color") && (
                          <span>Umbral de similitud de color: {formatPercent(detail.context_precision_color_threshold)}</span>
                        )}
                        <span>AP@k = promedio, sobre las prendas relevantes, de la precisión en la posición de cada una. Sube cuando las relevantes quedan más arriba.</span>
                      </>
                    ) : <span>No calculable: faltan atributos de consulta para evaluar relevancia.</span>}
                    <small>Relevancia proxy: categoría coincide y similitud de color sobre el umbral. Son las mismas señales que usa el reranker, así que la suba es esperable; no es una evaluación humana.</small>
                  </section>
                  <div className="monitor-stage-grid">
                    <CandidateList title="Antes del reranking (orden por coseno)" candidates={detail.retrieved || []} stage="retrieved" />
                    <CandidateList title="Después del reranking" candidates={detail.reranked || []} stage="reranked" previousRanks={previousRanks} weights={rerankWeights(detail)} />
                  </div>
                </>
              )}

              {activeTab === "outfits" && <OutfitHop detail={detail} />}

              {activeTab === "generator" && (
                <section className="monitor-generator-section">
                  <GenerationTrace detail={detail} />
                  <p className="monitor-generator-note">
                    {detail.llm_calls > 0
                      ? "Antes: los outfits del paso 3, enviados como contexto al modelo."
                      : "Antes: los outfits del paso 3; esta request no registró una llamada al modelo."}
                    {" "}Después: lo que devolvió la API. Si la generación falla o está desactivada, son los mismos outfits.
                  </p>
                  <div className="monitor-generator-grid">
                    <GeneratorOutfitList title="Antes del generador" outfits={detail.generator_input || []} stage="generator-input" />
                    <GeneratorOutfitList title="Después del generador" outfits={detail.generator_output || []} stage="generator-output" />
                  </div>
                </section>
              )}
            </>
          )}
        </section>
      </div>

      <details className="monitor-credentials">
        <summary>Uso de credenciales y notas sobre el costo</summary>
        {!summary?.credential_usage_24h?.length ? <p className="monitor-muted">Sin llamadas al generador en las últimas 24 horas.</p> : (
          <div className="monitor-credential-grid">
            {summary.credential_usage_24h.map((usage) => (
              <article className="monitor-credential-card" key={`${usage.llm_credential_fingerprint}-${usage.llm_model}-${usage.llm_credential_source}`}>
                <strong>{usage.llm_credential_source || "Credencial"} · {usage.llm_credential_fingerprint || "sin huella"}</strong>
                <span>Modelo: {usage.llm_model || "desconocido"}</span>
                <span>{formatNumber(usage.llm_calls)} llamadas LLM · {formatNumber(usage.request_count)} requests de app</span>
                <span>{formatNumber(usage.input_tokens)} input / {formatNumber(usage.output_tokens)} output tokens</span>
                <span>{formatMoney(usage.estimated_cost_usd)} estimado · {formatNumber(usage.llm_failures)} fallos</span>
              </article>
            ))}
          </div>
        )}
        <p className="monitor-footnote">El costo se estima con las tarifas configuradas. Las llamadas son invocaciones registradas por esta aplicación; no equivalen necesariamente a requests de red ni al consumo facturado del proveedor. El monitor no consulta saldo ni límites de la API.</p>
      </details>
    </div>
    </PageLayout>
  );
}
