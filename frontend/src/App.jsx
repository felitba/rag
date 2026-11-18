import React, { useEffect, useRef, useState } from "react";
import { API_BASE, FromStore, PageLayout, categoryLabel, colorLabel, formatNumber, useCatalog } from "./shared.jsx";

const LOOK_KIND_LABELS = {
  existing: "Outfit del catálogo",
  trimmed: "Outfit del catálogo, recortado",
  mix: "Nueva combinación",
};

// Entradas de ejemplo: un clic completa el formulario.
const SAMPLE_TEXTS = [
  "Disfraz de pirata",
  "Algo para salir de noche",
  "Look de oficina para el invierno",
  "Casamiento de día en la playa",
];

// Prendas del catálogo local; si la imagen no existe, la miniatura se oculta sola.
const SAMPLE_GARMENT_IDS = ["48678871", "111273640", "100263578", "105301005"];

const COMPATIBILITY_STATUS_LABELS = {
  scored: "evaluado",
  partial: "parcial",
  unavailable: "no disponible",
  unscorable: "sin pares",
  error: "error",
};

function TypeAwareCompatibility({ report }) {
  if (!report) return null;
  const hasScore = Number.isFinite(report.compatibility_similarity);
  const status = COMPATIBILITY_STATUS_LABELS[report.status] || report.status || "sin dato";
  const missing = report.missing_garment_ids?.length || 0;

  return (
    <div className={`outfit-compatibility status-${report.status || "unknown"}`}>
      <div className="outfit-compatibility-heading">
        <strong>Compatibilidad type-aware</strong>
        <span>{status}</span>
      </div>
      {hasScore ? (
        <p>
          <strong>{formatNumber(report.compatibility_similarity, 4)}</strong>
          {" · similitud normalizada · "}
          {formatNumber(report.pair_count)} pares
          {report.fallback_pair_count ? ` · ${formatNumber(report.fallback_pair_count)} con espacio general` : ""}
          {missing ? ` · ${formatNumber(missing)} prendas sin imagen` : ""}
        </p>
      ) : (
        <p>{report.reason || "No se pudo calcular para este outfit."}</p>
      )}
      {hasScore && <small>Escala 0–1; no es una probabilidad. El score original del paper queda disponible como compatibility_score en la respuesta.</small>}
    </div>
  );
}

function OutfitCards({ outfits, matchLabel }) {
  return (
    <div className="outfit-list">
      {outfits.map((outfit, index) => {
        const matched = new Set(outfit.matched_garment_ids || []);
        const notes = [
          outfit.look_kind === "mix" && `Combina prendas de ${outfit.source_outfits?.length || 2} outfits del catálogo.`,
          outfit.omitted_items?.length > 0 && `Quitamos del outfit original: ${outfit.omitted_items
            .map((item) => [categoryLabel(item.category), colorLabel(item.color).toLowerCase()].filter(Boolean).join(" ").toLowerCase())
            .join(", ")}.`,
          outfit.changes_from_source && `Según el modelo: ${outfit.changes_from_source}`,
        ].filter(Boolean);
        return (
          <article className="outfit-card" key={outfit.outfit_id}>
            <div className="outfit-card-heading">
              <div>
                <p className="eyebrow">LOOK {String(index + 1).padStart(2, "0")}</p>
                <h3>{outfit.name || `Outfit ${outfit.outfit_id}`}</h3>
              </div>
              <span className="outfit-tag">
                {LOOK_KIND_LABELS[outfit.look_kind] || (outfit.generated ? "Armado por IA" : "Outfit del catálogo")}
              </span>
            </div>
            {outfit.description && <p className="outfit-description">{outfit.description}</p>}
            {outfit.style_tags?.length > 0 && (
              <div className="style-tags">
                {outfit.style_tags.map((tag) => <span key={tag}>{tag}</span>)}
              </div>
            )}
            <TypeAwareCompatibility report={outfit.type_aware_compatibility} />
            <div className="garment-grid">
              {(outfit.items || []).map((item) => (
                <div className="garment-card" key={item.garment_id}>
                  <div className="garment-image-wrap">
                    <img
                      src={`${API_BASE}${item.image_url}`}
                      alt={`${categoryLabel(item.category)} ${colorLabel(item.color).toLowerCase()}`.trim()}
                      loading="lazy"
                    />
                    {matchLabel && matched.has(item.garment_id) && <span className="match-badge">{matchLabel}</span>}
                  </div>
                  <div className="garment-info">
                    <span className="garment-category">{categoryLabel(item.category)}</span>
                    {item.title && <p className="garment-title">{item.title}</p>}
                    {item.color && <p className="garment-color">{colorLabel(item.color)}</p>}
                  </div>
                </div>
              ))}
            </div>
            {(outfit.explanation || notes.length > 0) && (
              <div className="outfit-why">
                <span className="outfit-why-label">Por qué funciona</span>
                {outfit.explanation && <p>{outfit.explanation}</p>}
                {notes.map((note) => <p className="outfit-note" key={note}>{note}</p>)}
              </div>
            )}
          </article>
        );
      })}
    </div>
  );
}

function LoadingLooks() {
  return (
    <div className="outfit-list" aria-hidden="true">
      {[0, 1].map((card) => (
        <div className="outfit-card skeleton-card" key={card}>
          <div className="skeleton-line short" />
          <div className="skeleton-line" />
          <div className="garment-grid">
            {[0, 1, 2, 3, 4].map((garment) => <div className="skeleton-image" key={garment} />)}
          </div>
        </div>
      ))}
    </div>
  );
}

function isImage(candidate) {
  return Boolean(candidate) && ["image/jpeg", "image/png", "image/webp"].includes(candidate.type);
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [dragging, setDragging] = useState(false);
  // Qué se envió en la última consulta: se muestra arriba de los resultados.
  const [submitted, setSubmitted] = useState(null);
  const resultsRef = useRef(null);
  const fileInputRef = useRef(null);
  const catalog = useCatalog();

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Liberar la miniatura de la consulta anterior cuando se reemplaza.
  useEffect(() => () => {
    if (submitted?.previewUrl) URL.revokeObjectURL(submitted.previewUrl);
  }, [submitted]);

  // Al empezar a cargar y al llegar la respuesta, llevar la vista a los resultados.
  useEffect(() => {
    if (loading || results) {
      resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [loading, results]);

  function attach(candidate) {
    if (!isImage(candidate)) {
      setError("La foto tiene que ser JPG, PNG o WEBP.");
      return;
    }
    setFile(candidate);
    setError("");
  }

  function removeFile() {
    setFile(null);
    // Sin esto, volver a elegir el mismo archivo no dispara onChange.
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function pickSampleGarment(garmentId) {
    try {
      const response = await fetch(`${API_BASE}/images/${garmentId}.jpg`);
      if (!response.ok) throw new Error();
      const blob = await response.blob();
      attach(new File([blob], `ejemplo-${garmentId}.jpg`, { type: blob.type || "image/jpeg" }));
    } catch {
      setError("No se pudo cargar la prenda de ejemplo.");
    }
  }

  function onDrop(event) {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) attach(dropped);
  }

  function onPaste(event) {
    const pasted = event.clipboardData.files?.[0];
    if (pasted) {
      event.preventDefault();
      attach(pasted);
    }
  }

  function onKeyDown(event) {
    // Enter envía; Shift + Enter agrega un salto de línea.
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      event.currentTarget.form.requestSubmit();
    }
  }

  async function onSubmit(event) {
    event.preventDefault();
    const text = message.trim();
    if (loading) return;
    if (!file && !text) {
      setError("Elegí una foto de una prenda o escribí un mensaje para empezar.");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);
    setSubmitted({ hadPhoto: Boolean(file), text, previewUrl: file ? URL.createObjectURL(file) : "" });

    try {
      const body = new FormData();
      if (file) body.append("file", file);
      if (text) body.append("message", text);

      const response = await fetch(`${API_BASE}/api/recommend`, {
        method: "POST",
        body,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "No se pudieron generar recomendaciones.");
      }
      setResults(data);
    } catch (err) {
      setError(err.message || "No se pudo conectar con el backend.");
    } finally {
      setLoading(false);
    }
  }

  const outOfScope = results?.generation_status === "out_of_scope";
  const recommendations = results?.recommendations || [];
  // Solo con foto: en una búsqueda de texto casi todas las prendas coinciden y la etiqueta no informa.
  const matchLabel = !submitted?.hadPhoto ? null : submitted.text ? "Coincide" : "Parecida a tu foto";
  const detected = [
    categoryLabel(results?.query_attributes?.category?.label || ""),
    colorLabel(results?.query_attributes?.color?.color),
  ].filter((value) => value && value !== "Prenda");

  return (
    <PageLayout current="assistant" catalog={catalog}>
        <section className="composer-section" aria-labelledby="composer-heading">
          <h1 id="composer-heading">¿Qué me pongo?</h1>
          <p className="composer-lead">
            Mostranos una prenda o contanos la ocasión.
            Te armamos looks con prendas reales <FromStore catalog={catalog} />.
          </p>

          <form
            className={`composer ${dragging ? "dragging" : ""}`}
            onSubmit={onSubmit}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            {previewUrl && (
              <div className="composer-attachment">
                <img src={previewUrl} alt="Prenda elegida" />
                <button type="button" onClick={removeFile} aria-label="Quitar foto">×</button>
              </div>
            )}
            <textarea
              className="composer-input"
              rows={2}
              maxLength={300}
              placeholder={file ? "Opcional: ¿para qué ocasión?" : "Una ocasión, un estilo, una idea…"}
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              onKeyDown={onKeyDown}
              onPaste={onPaste}
              aria-label="Qué estás buscando"
            />
            <div className="composer-toolbar">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                hidden
                onChange={(event) => { const selected = event.target.files?.[0]; if (selected) attach(selected); }}
              />
              <button type="button" className="composer-attach" onClick={() => fileInputRef.current?.click()}>
                <span aria-hidden="true">＋</span> {file ? "Cambiar foto" : "Agregar foto"}
              </button>
              <span className="composer-hint">JPG, PNG o WEBP · o arrastrala acá</span>
              <button type="submit" className="submit-button" disabled={(!file && !message.trim()) || loading}>
                {loading ? <><span className="spinner" /> Armando…</> : "Buscar looks"}
              </button>
            </div>
          </form>
          {error && <p className="error-message" role="alert">{error}</p>}

          <div className="samples">
            <span className="samples-label">Probá con</span>
            {SAMPLE_GARMENT_IDS.map((garmentId) => (
              <button
                type="button"
                className="sample-garment"
                key={garmentId}
                onClick={() => pickSampleGarment(garmentId)}
                aria-label="Usar esta prenda de ejemplo"
              >
                <img
                  src={`${API_BASE}/images/${garmentId}.jpg`}
                  alt=""
                  onError={(event) => { event.currentTarget.parentElement.hidden = true; }}
                />
              </button>
            ))}
            {SAMPLE_TEXTS.map((sample) => (
              <button type="button" className="sample-text" key={sample} onClick={() => setMessage(sample)}>
                {sample}
              </button>
            ))}
          </div>
        </section>

        {(loading || results) && (
          <section className="results-section" aria-live="polite" ref={resultsRef}>
            <div className="query-strip">
              {submitted?.previewUrl && <img src={submitted.previewUrl} alt="Tu foto" />}
              <div>
                <span className="query-strip-label">Tu búsqueda</span>
                <p>{submitted?.text ? `“${submitted.text}”` : "Tu foto"}</p>
                {detected.length > 0 && <span className="query-strip-detected">En la foto vemos: {detected.join(" · ").toLowerCase()}</span>}
              </div>
              {results && !outOfScope && (
                results.generation_status === "generated" ? (
                  <span className="generation-status success">Armados por IA</span>
                ) : (
                  <span className="generation-status fallback">Outfits originales del catálogo</span>
                )
              )}
            </div>

            {loading && (
              <>
                <p className="loading-note">
                  <span className="spinner dark" />
                  Buscando prendas parecidas y armando los looks. Tarda unos 10 segundos.
                </p>
                <LoadingLooks />
              </>
            )}

            {results?.explanation && (
              <div className="explanation">
                <span className="explanation-mark">✳</span>
                <p>{results.explanation}</p>
              </div>
            )}

            {results && !outOfScope && (
              recommendations.length
                ? <OutfitCards outfits={recommendations} matchLabel={matchLabel} />
                : <div className="empty-state">No encontramos outfits para esta búsqueda.</div>
            )}
          </section>
        )}
    </PageLayout>
  );
}
