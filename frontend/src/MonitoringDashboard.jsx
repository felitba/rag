import React, { useCallback, useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

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

function CandidateList({ title, candidates, stage }) {
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
                  <strong>#{item.rank || index + 1} · {item.title || item.garment_id}</strong>
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

export default function MonitoringDashboard() {
  const [summary, setSummary] = useState(null);
  const [requests, setRequests] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");

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

  return (
    <main className="monitor-page">
      <header className="monitor-header">
        <div>
          <p className="eyebrow">OBSERVABILIDAD LOCAL</p>
          <h1>Monitor de retrieval</h1>
          <p>Requests, candidatos antes y después del reranking, y uso estimado del generador.</p>
        </div>
        <a href="/">Volver a recomendaciones</a>
      </header>

      {error && <div className="monitor-error">{error}</div>}

      <section className="monitor-metrics">
        <article><span>Requests · 24 h</span><strong>{formatNumber(summary?.requests_24h)}</strong><small>Últimos {summary?.retention_days || 30} días: {formatNumber(summary?.total_requests)}</small></article>
        <article><span>Errores · 24 h</span><strong>{formatNumber(summary?.errors_24h)}</strong></article>
        <article><span>Latencia media · 24 h</span><strong>{formatNumber(summary?.avg_latency_ms_24h, 0)} ms</strong></article>
        <article><span>Llamadas al LLM · 24 h</span><strong>{formatNumber(summary?.llm_calls_24h)}</strong><small>Últimos {summary?.retention_days || 30} días: {formatNumber(summary?.total_llm_calls)}</small></article>
        <article><span>Errores LLM · 24 h</span><strong>{formatNumber(summary?.llm_failures_24h)}</strong></article>
        <article><span>Tokens · 24 h</span><strong>{formatNumber((summary?.input_tokens_24h || 0) + (summary?.output_tokens_24h || 0))}</strong><small>Últimos {summary?.retention_days || 30} días: {formatNumber(summary?.total_tokens)}</small></article>
        <article><span>Costo estimado · 24 h</span><strong>{formatMoney(summary?.estimated_cost_usd_24h)}</strong><small>Últimos {summary?.retention_days || 30} días: {formatMoney(summary?.total_estimated_cost_usd)}</small></article>
      </section>

      <section className="monitor-request-panel monitor-credential-usage">
        <div className="monitor-section-heading"><h2>Uso de credenciales · 24 h</h2></div>
        {!summary?.credential_usage_24h?.length ? <p className="monitor-muted">Sin llamadas al generador en las últimas 24 horas.</p> : (
          <div className="monitor-credential-grid">
            {summary.credential_usage_24h.map((usage) => (
              <article className="monitor-credential-card" key={`${usage.llm_credential_fingerprint}-${usage.llm_model}`}>
                <strong>{usage.llm_credential_source || "Credencial"} · {usage.llm_credential_fingerprint || "sin huella"}</strong>
                <span>Modelo: {usage.llm_model || "desconocido"}</span>
                <span>{formatNumber(usage.llm_calls)} llamadas LLM · {formatNumber(usage.request_count)} requests de app</span>
                <span>{formatNumber(usage.input_tokens)} input / {formatNumber(usage.output_tokens)} output tokens</span>
                <span>{formatMoney(usage.estimated_cost_usd)} estimado · {formatNumber(usage.llm_failures)} fallos</span>
              </article>
            ))}
          </div>
        )}
      </section>

      <div className="monitor-layout">
        <section className="monitor-request-panel">
          <div className="monitor-section-heading">
            <h2>Requests recientes</h2>
            <button type="button" onClick={refresh}>Actualizar</button>
          </div>
          {!requests.length ? <p className="monitor-muted">Todavía no hay requests. Enviá una consulta desde la app.</p> : (
            <div className="monitor-requests">
              {requests.map((request) => (
                <button
                  type="button"
                  className={`monitor-request-row ${selectedId === request.request_id ? "selected" : ""}`}
                  key={request.request_id}
                  onClick={() => setSelectedId(request.request_id)}
                >
                  <span className="monitor-request-date">{formatDate(request.started_at)}</span>
                  <strong>{request.input_type || "procesando"} · {request.status_code || "en curso"} · {request.generation_status || "generando"}</strong>
                  <span>{formatNumber(request.duration_ms, 0)} ms · {request.retrieval_count}/{request.reranked_count} prendas</span>
                  <span>{formatNumber(request.input_tokens + request.output_tokens)} tokens · {formatMoney(request.estimated_cost_usd)}</span>
                  {request.context_precision_available ? <span>Context Precision {formatPercent(request.reranked_context_precision)}</span> : null}
                  <code>{request.request_id.slice(0, 12)}</code>
                </button>
              ))}
            </div>
          )}
        </section>

        <section className="monitor-detail-panel">
          {!detail ? <div className="monitor-empty">Seleccioná una request para ver sus candidatos.</div> : (
            <>
              <div className="monitor-detail-heading">
                <div>
                  <p className="eyebrow">DETALLE DE REQUEST</p>
                  <h2>{detail.request_id}</h2>
                  <span>{detail.input_type || "procesando"} · HTTP {detail.status_code || "en curso"} · {formatNumber(detail.duration_ms, 0)} ms · generación: {detail.generation_status || "en curso"}</span>
                </div>
                <div className="monitor-usage">
                  <strong>{formatNumber(detail.llm_calls)} llamadas · {detail.llm_model || "sin modelo"}</strong>
                  {detail.llm_credential_source && <span>Variable: {detail.llm_credential_source}</span>}
                  {detail.llm_credential_fingerprint && <span>Huella de clave: {detail.llm_credential_fingerprint}</span>}
                  <span>{formatNumber(detail.input_tokens)} input / {formatNumber(detail.output_tokens)} output tokens</span>
                  <span>{formatMoney(detail.estimated_cost_usd)} estimado</span>
                </div>
              </div>
              {(detail.input_image_url || detail.processed_image_url) && (
                <section className="monitor-query-images">
                  <h3>Imagen enviada y resultado de quitar el fondo</h3>
                  <div className="monitor-query-image-grid">
                    <figure>
                      <figcaption>Input del usuario</figcaption>
                      {detail.input_image_url
                        ? <img src={`${API_BASE}${detail.input_image_url}`} alt="Imagen original enviada por el usuario" />
                        : <div className="monitor-query-image-empty">Imagen no disponible</div>}
                    </figure>
                    <figure>
                      <figcaption>Después de quitar el fondo</figcaption>
                      {detail.processed_image_url
                        ? <img src={`${API_BASE}${detail.processed_image_url}`} alt="Imagen usada por el retrieval, después del procesamiento de fondo" />
                        : <div className="monitor-query-image-empty">Imagen procesada no disponible</div>}
                    </figure>
                  </div>
                </section>
              )}
              {detail.query_attributes && (
                <div className="monitor-query-attributes">
                  <strong>Consulta estimada:</strong>
                  <span>categoría {detail.query_attributes.category?.label || (detail.query_attributes.category?.uncertain ? `dudosa (${detail.query_attributes.category.best_label || "sin candidata"})` : "s/d")}</span>
                  {detail.query_attributes.category?.alternatives?.length > 0 && (
                    <span>{detail.query_attributes.category.uncertain ? "candidatas" : "alternativas"} {detail.query_attributes.category.alternatives.map((item) => `${item.label} (${formatNumber(item.similarity, 3)})`).join(" · ")}</span>
                  )}
                  {detail.query_attributes.category?.similarity !== undefined && (
                    <span>similitud {formatNumber(detail.query_attributes.category.similarity, 3)} · margen {formatNumber(detail.query_attributes.category.margin, 3)}</span>
                  )}
                  <span>color {detail.query_attributes.color?.color || "s/d"}</span>
                  <span>color pixels {formatNumber((detail.query_attributes.color?.color_confidence || 0) * 100, 1)}%</span>
                </div>
              )}
              <section className="monitor-context-precision">
                <strong>Context Precision (AP@k proxy)</strong>
                {detail.context_precision_available ? (
                  <>
                    <span>Antes: {formatPercent(detail.retrieved_context_precision)} · después: {formatPercent(detail.reranked_context_precision)}</span>
                    <span>{formatNumber(detail.context_precision_relevant_count)} prendas relevantes de {formatNumber(detail.context_precision_k)} · criterio: {(detail.context_precision_criteria || []).join(" + ")}</span>
                    {(detail.context_precision_criteria || []).includes("color") && (
                      <span>Umbral de similitud de color: {formatPercent(detail.context_precision_color_threshold)}</span>
                    )}
                  </>
                ) : <span>No calculable: faltan atributos de consulta para evaluar relevancia.</span>}
                <small>Relevancia proxy basada en categoría confiable y similitud de color; no es una evaluación humana.</small>
              </section>
              <div className="monitor-stage-grid">
                <CandidateList title="Antes del reranking" candidates={detail.retrieved || []} stage="retrieved" />
                <CandidateList title="Después del reranking" candidates={detail.reranked || []} stage="reranked" />
              </div>
              <section className="monitor-generator-section">
                <div className="monitor-generator-heading">
                  <div>
                    <p className="eyebrow">ETAPA DEL GENERADOR</p>
                    <h2>Outfits antes y después del generador</h2>
                  </div>
                  <span>Estado: {detail.generation_status || "en curso"}</span>
                </div>
                <p className="monitor-generator-note">
                  {detail.llm_calls > 0
                    ? "Antes: outfits recuperados después del reranking y enviados como contexto al modelo."
                    : "Antes: outfits recuperados después del reranking; esta request no registró una llamada al modelo."}
                  {" "}Después: resultado final que devolvió la API; si la generación falla o está desactivada, se muestran los outfits recuperados.
                </p>
                <div className="monitor-generator-grid">
                  <GeneratorOutfitList title="Antes del generador" outfits={detail.generator_input || []} stage="generator-input" />
                  <GeneratorOutfitList title="Después del generador" outfits={detail.generator_output || []} stage="generator-output" />
                </div>
              </section>
            </>
          )}
        </section>
      </div>
      <p className="monitor-footnote">El costo se estima con las tarifas configuradas. Las llamadas son invocaciones registradas por esta aplicación; no equivalen necesariamente a requests de red ni al consumo facturado del proveedor. El monitor no consulta saldo ni límites de la API.</p>
    </main>
  );
}
