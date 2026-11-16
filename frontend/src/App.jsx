import React, { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function money(value) {
  if (value === null || value === undefined) return "Precio no disponible";
  return new Intl.NumberFormat("es-AR", {
    style: "currency",
    currency: "ARS",
    maximumFractionDigits: 2,
  }).format(value);
}

function alignmentLabel(alignment) {
  return {
    match: "Coincide en categoría y color",
    category_only: "Coincide en categoría; falta dato de color",
    mismatch: "No coincide con los atributos detectados",
    unknown: "Sin datos suficientes para comparar",
  }[alignment] || "Sin datos suficientes para comparar";
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function onSelectFile(event) {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setResults(null);
    setError("");
  }

  async function onSubmit(event) {
    event.preventDefault();
    const text = message.trim();
    if (!file && !text) {
      setError("Elegí una foto de una prenda o escribi un mensaje para empezar.");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);

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

  return (
    <div className="page-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Fashion RAG inicio">
          <span className="brand-mark">F</span>
          <span>FORMA <em>STUDIO</em></span>
        </a>
        <span className="header-note">Tu estilo, nuevas combinaciones</span>
      </header>

      <main id="top">
        <section className="hero">
          <p className="eyebrow">RECOMENDADOR VISUAL</p>
          <h1>Una prenda.<br /><i>Muchas posibilidades.</i></h1>
          <p className="hero-copy">
            Subí la foto de una prenda y encontrá outfits seleccionados de nuestro catálogo,
            inspirados en su estilo y apariencia.
          </p>
        </section>

        <section className="upload-panel" aria-labelledby="upload-heading">
          <div className="upload-copy">
            <p className="eyebrow">PASO 01</p>
            <h2 id="upload-heading">Elegí tu prenda de referencia</h2>
            <p>Funciona mejor con una foto clara, donde la prenda sea el centro de atención.</p>
            <form onSubmit={onSubmit}>
              <label className="file-picker">
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={onSelectFile}
                />
                <span className="file-picker-icon">＋</span>
                <span>{file ? file.name : "Seleccionar una foto"}</span>
              </label>
              <textarea
                className="message-input"
                rows={3}
                maxLength={300}
                placeholder="Opcional: contanos qué estás buscando"
                value={message}
                onChange={(event) => setMessage(event.target.value)}
              />
              <button
                type="submit"
                className="submit-button"
                disabled={(!file && !message.trim()) || loading}
              >Enviar</button>
              <p className="file-hint">JPG, PNG o WEBP · Máximo 10 MB</p>
            </form>
            {error && <p className="error-message" role="alert">{error}</p>}
          </div>

          <div className={`preview ${previewUrl ? "has-image" : ""}`}>
            {previewUrl ? (
              <img src={previewUrl} alt="Vista previa de la prenda elegida" />
            ) : (
              <div className="preview-placeholder">
                <span className="hanger">⌁</span>
                <span>Tu foto aparece acá</span>
              </div>
            )}
          </div>
        </section>

        {results && (
          <section className="results-section" aria-live="polite">
            <div className="section-heading">
              <div>
                <p className="eyebrow">PASO 02</p>
                <h2>Outfits para explorar</h2>
              </div>
              <span className="result-count">
                {results.recommendations.length} {results.recommendations.length === 1 ? "look" : "looks"}
              </span>
            </div>

            {results.explanation && (
              <div className="explanation">
                <span className="explanation-mark">✳</span>
                <p>{results.explanation}</p>
              </div>
            )}

            {results.query_attributes && (
              <div className="attribute-analysis">
                <p className="eyebrow">LECTURA VISUAL DE LA FOTO</p>
                <div className="attribute-values">
                  <span>Prenda: <strong>{results.query_attributes.category?.label || "No clasificada"}</strong></span>
                  <span>Color: <strong>{results.query_attributes.color?.color || "Sin estimación"}</strong></span>
                </div>
                <p className="attribute-note">Estimación zero-shot de FashionCLIP; los scores son similitudes relativas, no probabilidades.</p>
                {results.retrieved_metadata_matches?.length > 0 && (
                  <div className="metadata-check">
                    <strong>Chequeo orientativo contra metadatos recuperados</strong>
                    {results.retrieved_metadata_matches.map((match) => (
                      <div className="metadata-match-row" key={match.garment_id}>
                        <span>{match.title || match.garment_id}</span>
                        <span>{alignmentLabel(match.alignment)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {results.recommendations.length === 0 ? (
              <div className="empty-state">No encontramos outfits disponibles para esta foto. Probá con otra toma de la prenda.</div>
            ) : (
              <div className="outfit-list">
                {results.recommendations.map((outfit, index) => (
                  <article className="outfit-card" key={outfit.outfit_id}>
                    <div className="outfit-card-heading">
                      <div>
                        <p className="eyebrow">LOOK {String(index + 1).padStart(2, "0")}</p>
                        <h3>{outfit.name || outfit.outfit_id.replaceAll("_", " ")}</h3>
                      </div>
                      <span className="outfit-tag">{outfit.generated ? "Nueva combinación" : "Del catálogo"}</span>
                    </div>
                    {outfit.description && <p className="outfit-description">{outfit.description}</p>}
                    {outfit.explanation && <p className="outfit-explanation">{outfit.explanation}</p>}
                    <div className="garment-grid">
                      {outfit.items.map((item) => (
                        <div className="garment-card" key={item.garment_id}>
                          <div className="garment-image-wrap">
                            <img
                              src={`${API_BASE}${item.image_url}`}
                              alt={`${item.category}, ${item.color}`}
                              loading="lazy"
                            />
                          </div>
                          <div className="garment-info">
                            <span className="garment-category">{item.category}</span>
                            <p>{item.color}</p>
                            <strong>{money(item.price)}</strong>
                          </div>
                        </div>
                      ))}
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        )}
      </main>

      <footer>
        <span>FORMA STUDIO</span>
        <span>Recomendaciones basadas en el catálogo disponible.</span>
      </footer>
    </div>
  );
}
