import React, { useEffect, useState } from "react";

export const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// El catálogo está en inglés; la interfaz, en español.
export const CATEGORY_LABELS = {
  tops: "Parte de arriba",
  bottoms: "Parte de abajo",
  "all-body": "Vestido o enterito",
  outerwear: "Abrigo",
  shoes: "Calzado",
  bags: "Bolso",
  jewellery: "Joyería",
  sunglasses: "Anteojos de sol",
  hats: "Sombrero",
  scarves: "Pañuelo o bufanda",
  accessories: "Accesorio",
};

export const COLOR_LABELS = {
  black: "Negro",
  gray: "Gris",
  white: "Blanco",
  beige: "Beige",
  brown: "Marrón",
  red: "Rojo",
  orange: "Naranja",
  yellow: "Amarillo",
  green: "Verde",
  blue: "Azul",
  purple: "Violeta",
  pink: "Rosa",
};

export function categoryLabel(category) {
  return CATEGORY_LABELS[category] || category || "Prenda";
}

export function colorLabel(color) {
  return COLOR_LABELS[color] || color || "";
}

export function formatNumber(value, digits = 0) {
  return new Intl.NumberFormat("es-AR", { maximumFractionDigits: digits }).format(value || 0);
}

// Nombre, link y estadísticas de la tienda. Todo sale de config.json ("store") y de
// los archivos del catálogo en el backend: cambiar de tienda no toca el frontend.
export function useCatalog() {
  const [catalog, setCatalog] = useState(null);
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/catalog`)
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => { if (!cancelled) setCatalog(data); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);
  return catalog;
}

// "de Polyvore" con link si la tienda está configurada; si no, "del catálogo".
export function FromStore({ catalog }) {
  const store = catalog?.store;
  if (!store?.name) return "del catálogo";
  return (
    <>de {store.url ? <a href={store.url} target="_blank" rel="noreferrer">{store.name}</a> : store.name}</>
  );
}

const PAGES = [
  { key: "assistant", label: "Asistente", href: "/" },
  { key: "catalog", label: "Catálogo", href: "/catalogo" },
  { key: "monitor", label: "Monitor", href: "/monitor" },
];

export function TopNav({ current, catalog }) {
  return (
    <nav className="top-nav" aria-label="Secciones">
      <a className="top-nav-store" href="/">
        {catalog?.store?.name ? `${catalog.store.name} · ` : ""}Asistente de outfits
      </a>
      <div className="top-nav-links">
        {PAGES.map((page) => (
          page.key === current
            ? <span className="top-nav-current" aria-current="page" key={page.key}>{page.label}</span>
            : <a href={page.href} key={page.key}>{page.label}</a>
        ))}
      </div>
    </nav>
  );
}

export function Footer({ catalog }) {
  const retention = catalog?.photo_retention_days;
  return (
    <footer className="site-footer">
      <div>
        <strong>Resultados</strong>
        <p>
          {catalog
            ? <>Buscamos en {formatNumber(catalog.garments)} prendas y {formatNumber(catalog.outfits)} outfits <FromStore catalog={catalog} />. </>
            : "Buscamos solo en las prendas del catálogo. "}
          Los looks los arma una IA y puede equivocarse. <a href="/catalogo">Qué tiene el catálogo</a>.
        </p>
      </div>
      <div>
        <strong>Tu foto</strong>
        <p>
          {retention
            ? `La guardamos hasta ${retention} días para monitorear el sistema. Evitá fotos con caras.`
            : "La usamos solo para esta búsqueda."}
        </p>
      </div>
      <div>
        <strong>Proyecto</strong>
        <p>Fashion RAG · Grupo 04 · TP3 de LLM (73.69), ITBA. Prototipo académico.</p>
      </div>
    </footer>
  );
}

// El mismo marco para todas las páginas: navegación arriba, contenido, pie.
export function PageLayout({ current, catalog, children }) {
  return (
    <div className="page-shell">
      <TopNav current={current} catalog={catalog} />
      <main id="top">{children}</main>
      <Footer catalog={catalog} />
    </div>
  );
}
