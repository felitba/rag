import React from "react";
import {
  API_BASE, FromStore, PageLayout, categoryLabel, colorLabel, formatNumber, useCatalog,
} from "./shared.jsx";

function percent(share) {
  return `${formatNumber(share * 100, 1)}%`;
}

// Lista de barras: una fila por valor, con cantidad y porcentaje escritos.
function BarList({ rows }) {
  const largest = Math.max(...rows.map((row) => row.share), 0.0001);
  return (
    <ul className="bar-list">
      {rows.map((row) => (
        <li key={row.label}>
          <span className="bar-list-label">{row.label}</span>
          <span className="bar-list-track">
            <span className={row.share === 0 ? "empty" : ""} style={{ width: `${(row.share / largest) * 100}%` }} />
          </span>
          <span className="bar-list-value">{formatNumber(row.count)} · {percent(row.share)}</span>
        </li>
      ))}
    </ul>
  );
}

function biasNotes(catalog) {
  const categories = catalog.categories;
  const top = categories.slice(0, 3);
  const last = categories[categories.length - 1];
  const coverage = Object.fromEntries(catalog.metadata_coverage.map((row) => [row.field, row]));
  const audience = catalog.audience;
  const topColors = catalog.colors.slice(0, 3);
  const estimated = catalog.color_sources.find((row) => row.value !== "sin dato");
  const lower = (row) => categoryLabel(row.value).toLowerCase();
  const notes = [
    {
      title: "Pocas categorías dominan",
      text: `${top.map(lower).join(", ")} son el ${percent(top.reduce((sum, row) => sum + row.share, 0))}. De ${lower(last)} hay ${formatNumber(last.count)} (${percent(last.share)}): vas a ver pocas opciones.`,
    },
    coverage.title && {
      title: "La mayoría no tiene texto",
      text: `Solo el ${percent(coverage.title.share)} tiene título y el ${percent(coverage.description?.share || 0)} descripción. Al resto lo conocemos por foto, categoría y color, y lo explicamos peor.`,
    },
    audience.garments_with_text > 0 && {
      title: "Casi todo es ropa de mujer",
      text: `De ${formatNumber(audience.garments_with_text)} prendas con texto, ${formatNumber(audience.mention_women)} mencionan mujer y ${formatNumber(audience.mention_men)} hombre. Para hombre o niños hay muy poco.`,
    },
    estimated && {
      title: "El color es estimado",
      text: `El catálogo no trae color: lo calculamos desde la foto y puede fallar. Los más frecuentes: ${topColors.map((row) => `${colorLabel(row.value).toLowerCase()} (${percent(row.share)})`).join(", ")}.`,
    },
    coverage.price && coverage.price.share === 0 && {
      title: "Sin precio ni stock",
      text: "Ninguna prenda tiene precio y todas figuran disponibles. No filtramos por presupuesto ni sabemos si se sigue vendiendo.",
    },
    {
      title: "Lo que no existe",
      text: "No hay talle, género, marca, temporada ni material. Los textos están en inglés y reflejan el gusto de quienes armaron los outfits.",
    },
  ];
  return notes.filter(Boolean);
}

export default function CatalogPage() {
  const catalog = useCatalog();

  return (
    <PageLayout current="catalog" catalog={catalog}>
      <div className="catalog-page">
        <header className="catalog-header">
          <p className="eyebrow">TRANSPARENCIA</p>
          <h1>Qué hay en el catálogo</h1>
          <p className="composer-lead">
            El asistente solo recomienda lo que existe en el catálogo <FromStore catalog={catalog} />.
            Acá ves qué tiene, qué le falta y cómo eso sesga los resultados.
          </p>
        </header>

        {!catalog ? (
          <div className="empty-state">Cargando el catálogo… Si no aparece, reiniciá el backend.</div>
        ) : (
          <>
            <section className="catalog-kpis">
              <article><span>Prendas buscables</span><strong>{formatNumber(catalog.garments)}</strong>
                {catalog.garments_in_catalog_file !== catalog.garments && (
                  <small>de {formatNumber(catalog.garments_in_catalog_file)} en el archivo; el resto no está indexado</small>
                )}
              </article>
              <article><span>Outfits de referencia</span><strong>{formatNumber(catalog.outfits)}</strong>
                <small>armados por personas; son el punto de partida</small>
              </article>
              <article><span>Categorías</span><strong>{formatNumber(catalog.categories.length)}</strong>
                <small>la más chica tiene {formatNumber(catalog.categories[catalog.categories.length - 1].count)} prendas</small>
              </article>
              <article><span>Con título</span><strong>{percent(catalog.metadata_coverage.find((row) => row.field === "title")?.share || 0)}</strong>
                <small>el resto: solo foto, categoría y color</small>
              </article>
            </section>

            {catalog.store.about && (
              <p className="catalog-about">
                <strong>De dónde sale.</strong> {catalog.store.about}
                {catalog.store.url && <> <a href={catalog.store.url} target="_blank" rel="noreferrer">Ver la fuente</a>.</>}
              </p>
            )}

            <section className="catalog-section">
              <h2>Sesgos y límites</h2>
              <div className="bias-grid">
                {biasNotes(catalog).map((note) => (
                  <article key={note.title}>
                    <h3>{note.title}</h3>
                    <p>{note.text}</p>
                  </article>
                ))}
              </div>
            </section>

            <div className="catalog-columns">
              <section className="catalog-section">
                <h2>Prendas por categoría</h2>
                <BarList rows={catalog.categories.map((row) => ({ ...row, label: categoryLabel(row.value) }))} />
              </section>
              <section className="catalog-section">
                <h2>Prendas por color estimado</h2>
                <BarList rows={catalog.colors.map((row) => ({ ...row, label: colorLabel(row.value) }))} />
              </section>
            </div>

            <div className="catalog-columns">
              <section className="catalog-section">
                <h2>Qué datos tiene cada prenda</h2>
                <p className="catalog-note">Prendas que traen cada campo.</p>
                <BarList rows={catalog.metadata_coverage.map((row) => ({ ...row, label: row.label }))} />
              </section>
              <section className="catalog-section">
                <h2>Prendas por outfit</h2>
                <p className="catalog-note">Tamaño de los outfits de referencia.</p>
                <BarList
                  rows={catalog.outfit_sizes.map((row) => ({
                    label: `${row.size} prendas`,
                    count: row.count,
                    share: row.count / Math.max(catalog.outfits, 1),
                  }))}
                />
              </section>
            </div>

            <section className="catalog-section">
              <h2>Cómo se ve una prenda en el catálogo</h2>
              <p className="catalog-note">
                Dos registros reales: uno completo y uno sin texto. Es todo lo que el asistente sabe de la prenda, además de la foto.
              </p>
              <div className="sample-records">
                {catalog.samples.map((sample, index) => (
                  <article key={sample.garment_id}>
                    <img src={`${API_BASE}${sample.image_url}`} alt="" loading="lazy" />
                    <div>
                      <span className="monitor-chip">{index === 0 ? "Registro completo" : "Registro sin texto"}</span>
                      <pre>{JSON.stringify(sample.record, null, 2)}</pre>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          </>
        )}
      </div>
    </PageLayout>
  );
}
