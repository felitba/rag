# Fashion RAG — CLIP + ChromaDB + LangChain + OpenAI

Proyecto de referencia para recibir una foto de una prenda, encontrar prendas visualmente similares con CLIP, recuperar los outfits prearmados relacionados y generar una explicación en español con un modelo multimodal de OpenAI.

## Arquitectura

1. **Indexación offline:** CLIP genera un embedding por imagen del catálogo. LangChain almacena los vectores en una colección persistente de ChromaDB junto con metadatos escalares. Los IDs de outfits se guardan como JSON dentro de los metadatos.
2. **Expansión antes de recuperar (HyDE visual):** el usuario sube una imagen y/o escribe una descripción. Un modelo de visión económico (`gpt-4o-mini` por defecto) analiza la prenda y genera 2 o 3 descripciones de prendas complementarias. No se consulta la base todavía.
3. **Embeddings hipotéticos:** las descripciones complementarias en inglés se pasan por `ClipEmbedding.embed_text()` para obtener los vectores objetivo.
4. **Recuperación amplia:** con el embedding de la imagen original (o de la descripción ancla generada, en consultas solo texto) se piden `candidate_pool_k` prendas a ChromaDB y se traducen a outfits candidatos.
5. **Rerank:** cada outfit candidato se puntúa con la similitud coseno exacta entre sus prendas y (a) la consulta original y (b) cada embedding hipotético. Los puntajes se normalizan y se combinan con pesos configurables.
6. **Explicación condicionada:** el resumen generado por el modelo de visión se muestra solo si suficientes hipótesis fueron encontradas en los outfits seleccionados. Si no, se devuelve un mensaje genérico. No se envían imágenes del catálogo al LLM por defecto.
7. **Frontend:** React muestra los outfits recuperados, las fotografías del catálogo y la explicación.

> El LLM no decide qué productos existen ni puede agregar artículos a los resultados de la tienda. La selección de IDs y los datos comerciales quedan controlados por el backend.

## Estructura

```text
fashion-rag/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── data/
│   │   ├── garments.json
│   │   └── outfits.json
│   └── services/
│       ├── catalog_service.py
│       ├── clip_service.py
│       ├── chroma_service.py
│       ├── outfit_service.py
│       └── openai_service.py
├── catalog/garments/       # fotografías originales del catálogo
├── scripts/index_catalog.py
├── storage/chroma/          # persistencia local (se crea automáticamente)
├── frontend/                # React + Vite
├── .env.example
└── requirements.txt
```

## 1. Requisitos

- Python 3.11 recomendado.
- Node.js 20 o superior para el frontend.
- Una API key de OpenAI.
- CPU o GPU compatible con PyTorch. En CPU funciona, pero indexar muchas imágenes puede tardar más.

## 2. Instalar backend en Windows (PowerShell)

Desde la carpeta raíz `fashion-rag`:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Instalá PyTorch siguiendo la opción CPU/CUDA apropiada para tu computadora en el selector oficial de PyTorch. Luego:

```powershell
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
Copy-Item .env.example .env
```

Editá `.env` y reemplazá `OPENAI_API_KEY=replace_me` por tu clave. No la guardes en React ni la subas a Git.

La instalación de CLIP se hace desde el repositorio oficial para no confundirlo con otros paquetes de Python que también se llaman `clip`.

## 3. Agregar las imágenes

Copiá tus fotos a `catalog/garments/`. Después editá `app/data/garments.json` con una entrada por prenda:

```json
{
  "garment_id": "jacket_104",
  "image_path": "catalog/garments/jacket_104.jpg",
  "category": "jacket",
  "color": "black",
  "style": "streetwear",
  "price": 89.99,
  "available": true
}
```

`garment_id` debe ser único y `image_path` debe señalar una imagen que esté dentro de la carpeta `catalog/`. Los campos de categoría, color, estilo y precio son tus metadatos del catálogo; adaptalos a tu esquema.

En `app/data/outfits.json`, definí cada outfit con los IDs de todas sus prendas:

```json
{
  "outfit_1": ["jacket_104", "shirt_025", "jeans_083", "shoes_011"],
  "outfit_2": ["jacket_108", "shirt_041", "jeans_092", "shoes_036"]
}
```

Cada ID referenciado en `outfits.json` debe existir en `garments.json`. Cada prenda puede pertenecer a cero, uno o varios outfits. El proceso de indexación construye la relación inversa automáticamente; no hace falta mantener a mano los `outfit_ids` dentro de `garments.json`.

El JSON incluido trae datos ilustrativos y rutas de imágenes de ejemplo. Antes de indexar, reemplazá o completá esos datos y agregá los archivos correspondientes.

## 4. Indexar / reconstruir ChromaDB

```powershell
python scripts/index_catalog.py
```

El script valida las referencias y la existencia de todas las imágenes antes de modificar el índice. Después **reconstruye la colección completa** y calcula los embeddings con CLIP. Si agregás, quitás o modificás fotos, volvé a ejecutarlo.

ChromaDB persiste los vectores en `storage/chroma/`. No subas esa carpeta a Git; podés reconstruirla usando el catálogo.

## 5. Iniciar FastAPI

```powershell
uvicorn app.main:app --reload --port 8000
```

- Swagger: <http://localhost:8000/docs>
- Estado: <http://localhost:8000/health>

Probá `POST /api/recommend` desde Swagger usando una fotografía JPEG, PNG o WEBP. Si ChromaDB está vacío, ejecutá primero la indexación.

## 6. Iniciar React

En una segunda terminal:

```powershell
cd frontend
npm install
npm run dev
```

Abrí la URL que muestre Vite (por defecto <http://localhost:5173>). El frontend envía la foto a FastAPI y muestra los resultados que devuelve el backend.

Opcionalmente, creá `frontend/.env` si la API no está en localhost:

```dotenv
VITE_API_BASE_URL=http://localhost:8000
```

## Variables de entorno

| Variable | Para qué sirve | Valor inicial |
|---|---|---|
| `OPENAI_API_KEY` | Autenticación para el LLM | Obligatoria para explicaciones |
| `OPENAI_MODEL` | Modelo multimodal disponible en tu cuenta | `gpt-4.1-mini` |
| `CLIP_MODEL` | Modelo de CLIP local | `ViT-B/32` |
| `CHROMA_PATH` | Directorio persistente | `storage/chroma` |
| `COLLECTION_NAME` | Colección vectorial | `fashion_garments` |
| `TOP_K_GARMENTS` | Cantidad máxima de prendas candidatas | `30` |
| `MAX_OUTFITS` | Outfits devueltos | `5` |
| `MAX_IMAGES_PER_OUTFIT_FOR_LLM` | Imágenes del catálogo enviadas al LLM por outfit | `2` |
| `MAX_UPLOAD_MB` | Tamaño máximo de foto subida | `10` |

Si la API key falta o OpenAI falla, el backend todavía puede devolver outfits recuperados y una explicación de fallback. Para el flujo completo, configurá una clave válida y un modelo con soporte de imágenes.

## Configuración de la expansión y el rerank

Los parámetros principales viven en `home_made_rag/config.json`:

| Clave | Para qué sirve | Valor inicial |
|---|---|---|
| `query_expansion.enabled` | Activa la expansión HyDE visual | `true` |
| `query_expansion.model_name` | Modelo de visión económico | `gpt-4o-mini` |
| `query_expansion.num_complements` | Prendas complementarias a generar | `3` |
| `query_expansion.image_detail` | Nivel de detalle de la imagen enviada | `low` |
| `query_expansion.max_output_tokens` | Límite de tokens de salida de la expansión | `800` |
| `retrieval.candidate_pool_k` | Prendas recuperadas antes del rerank | `100` |
| `retrieval.rerank_visual_weight` | Peso de la similitud visual | `0.35` |
| `retrieval.rerank_hypothesis_weight` | Peso de la similitud con las hipótesis | `0.65` |
| `explanation.min_hypothesis_similarity` | Umbral de cobertura por hipótesis | `0.24` |
| `explanation.min_covered_hypotheses` | Hipótesis mínimas encontradas para mostrar el resumen | `2` |

Si `query_expansion.enabled` es `false`, o si la llamada al modelo falla, se usa el flujo anterior de recuperación visual directa.

## Cómo se calcula el ranking

Para cada outfit candidato `o`:

```text
visual(o)      = max_i cos(query_embedding, item_embedding_i)
hypothesis(o)  = mean_j max_i cos(hypothesis_embedding_j, item_embedding_i)
```

Ambos puntajes se normalizan con min-max sobre todos los outfits candidatos y se combinan:

```text
score(o) = rerank_visual_weight * norm(visual(o))
         + rerank_hypothesis_weight * norm(hypothesis(o))
```

Los pesos por defecto son `0.35` y `0.65`. La similitud coseno de CLIP no está calibrada, por eso la normalización es necesaria. **El valor no es una probabilidad de compatibilidad.**

### Explicación condicionada por cobertura

Después del rerank, para cada hipótesis se calcula la mejor similitud contra las prendas de los outfits seleccionados. Si al menos `min_covered_hypotheses` superan `min_hypothesis_similarity`, se muestra el resumen generado por el modelo de visión. En caso contrario se devuelve `generic_message_es`. La cobertura usa prendas distintas para evitar que una sola prenda satisfaga varias hipótesis.

El umbral `0.24` es solo un punto de partida. Conviene calibrarlo con consultas reales; si resulta inestable, se puede reemplazar por un z-score relativo a la distribución de similitudes de las prendas candidatas.

### Logs de costo

Cada llamada a OpenAI registra en consola y en `Orag.log` una línea con el nombre de la llamada, modelo, tokens de entrada/salida, tokens cacheados y costo estimado en USD:

```text
[LLM] call=query_expansion model=gpt-4o-mini input_tokens=512 cached_tokens=0 output_tokens=280 total_tokens=792 input_cost_usd=0.000077 output_cost_usd=0.000168 estimated_cost_usd=0.000245
```

Los precios se configuran en `llm_pricing_usd_per_million_tokens` dentro de `config.json`. Si un modelo no tiene precio configurado, el costo se informa como `0` y se emite una advertencia.

## Consideraciones antes de producción

- El endpoint comprueba tamaño, formato y dimensiones de la imagen; las imágenes subidas se normalizan y se borran al finalizar la solicitud.
- El backend sirve las fotos del catálogo por `/catalog/...`. No coloques allí archivos privados.
- La clave de OpenAI se mantiene en el backend.
- En producción, agregá autenticación/rate limiting, logging estructurado, límites de concurrencia para CLIP, monitoreo de costos de OpenAI y almacenamiento de imágenes adecuado.
- El ranking visual debe validarse con fotos reales de usuarios. CLIP puede verse influido por fondo, pose, iluminación o por varias prendas visibles en una sola foto.
- El frontend formatea el precio como ARS solo a modo de demostración. Cambiá `money()` en `frontend/src/App.jsx` a la moneda real de tu tienda.
