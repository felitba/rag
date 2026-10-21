import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function money(value) {
  if (value === null || value === undefined) return "Precio no disponible";
  return new Intl.NumberFormat("es-AR", {
    style: "currency",
    currency: "ARS",
    maximumFractionDigits: 2,
  }).format(value);
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

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
    if (!file) {
      setError("Elegí una foto de una prenda para empezar.");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);

    try {
      const body = new FormData();
      body.append("file", file);

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
              <button type="submit" className="submit-button" disabled={!file || loading}>
                {loading ? <><span className="spinner" /> Buscando outfits…</> : "Encontrar outfits →"}
              </button>
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

            {results.recommendations.length === 0 ? (
              <div className="empty-state">No encontramos outfits disponibles para esta foto. Probá con otra toma de la prenda.</div>
            ) : (
              <div className="outfit-list">
                {results.recommendations.map((outfit, index) => (
                  <article className="outfit-card" key={outfit.outfit_id}>
                    <div className="outfit-card-heading">
                      <div>
                        <p className="eyebrow">LOOK {String(index + 1).padStart(2, "0")}</p>
                        <h3>{outfit.outfit_id.replaceAll("_", " ")}</h3>
                      </div>
                      <span className="outfit-tag">Del catálogo</span>
                    </div>
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
