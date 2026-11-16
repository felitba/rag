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
              <div className="monitor-stage-grid">
                <CandidateList title="Antes del reranking" candidates={detail.retrieved || []} stage="retrieved" />
                <CandidateList title="Después del reranking" candidates={detail.reranked || []} stage="reranked" />
              </div>
            </>
          )}
        </section>
      </div>
      <p className="monitor-footnote">El costo se estima con las tarifas configuradas. Las llamadas son invocaciones registradas por esta aplicación; no equivalen necesariamente a requests de red ni al consumo facturado del proveedor. El monitor no consulta saldo ni límites de la API.</p>
    </main>
  );
}
