# Fashion RAG

Aplicación académica de recomendación de outfits basada en recuperación
aumentada (RAG). El sistema combina un backend FastAPI, una interfaz React y
un catálogo local derivado de Polyvore.

El asistente acepta una foto, texto o ambos. Recupera prendas similares del
catálogo, encuentra outfits de referencia y, cuando hay un proveedor LLM
configurado, puede crear nuevas combinaciones usando únicamente prendas
recuperadas. Sin credenciales LLM, sigue funcionando como buscador y devuelve
outfits existentes.

## Arquitectura

1. `scripts/prepare_polyvore.py` descarga una muestra del dataset
   `mvasil/polyvore-outfits` y genera el catálogo local con imágenes.
2. `scripts/index_catalog.py` calcula embeddings CLIP, completa metadatos de
   color y crea la colección `fashion_items` en ChromaDB.
3. Para una foto, el backend puede quitar el fondo, calcular su embedding CLIP,
   clasificar la categoría con FashionCLIP y estimar el color. Para texto,
   genera consultas de búsqueda y usa embeddings de texto CLIP.
4. El recuperador busca prendas candidatas y el reranker combina similitud
   CLIP, categoría y color. Después las relaciona con outfits del catálogo.
5. El generador LLM, si está disponible, propone combinaciones validadas contra
   los IDs recuperados. No puede inventar prendas fuera del catálogo.
6. El modelo type-aware calcula una evaluación de compatibilidad por outfit.
   Esta métrica se muestra como información adicional; no reordena las
   recomendaciones.

`context_precision` es una señal automática basada en categoría y similitud de
color. No es una evaluación humana ni ground truth.

## Estructura

```text
home_made_rag/       Backend FastAPI y pipeline RAG
frontend/            Interfaz React + Vite
scripts/             Preparación, indexación y evaluación
models/type_aware/   Checkpoint y espacios type-aware
reports/             Resultados de evaluación
data/                Catálogo e imágenes locales (regenerable, ignorado por Git)
chroma_db/           Índice ChromaDB local (regenerable, ignorado por Git)
storage/             SQLite del monitor e imágenes de consultas
demos/               Notebooks de demostración
docs/                Material de referencia y trabajos prácticos
```

## Requisitos

- Python 3.11 recomendado.
- Node.js 20 o superior.
- `curl` para descargar los artefactos type-aware si no están presentes.
- Internet durante la primera instalación, para el dataset y los modelos que
  todavía no estén en caché.
- Una credencial LLM solo para reescritura de consultas y generación de nuevos
  outfits.

## Instalación

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Editá `.env` y agregá al menos una credencial si querés usar generación:

```dotenv
LLM_API_KEY=replace_me
```

También se admite `OPENAI_API_KEY` para usar un endpoint compatible con OpenAI.
Los fallbacks se configuran en `home_made_rag/config.json`; actualmente
OpenRouter está habilitado y Hugging Face está deshabilitado. Las claves son
opcionales para la recuperación.

## Preparar el catálogo

Los archivos de `data/` no forman parte del repositorio y deben generarse antes
de indexar:

```bash
python scripts/prepare_polyvore.py --n 200
```

`--n` es la cantidad de outfits fuente que se inspeccionan y vale `200` por
defecto. El script genera:

- `data/catalog_items.jsonl`
- `data/outfits.json`
- `data/images/`

El checkpoint type-aware y sus espacios ya están en `models/type_aware/`. Si
faltan, descargalos y verificá sus hashes con:

```bash
python scripts/download_type_aware.py
```

Después construí el índice:

```bash
python scripts/index_catalog.py
```

El comando valida las imágenes, calcula colores faltantes y **borra y recrea**
la colección ChromaDB configurada en `home_made_rag/config.json`. Para
recalcular todos los colores:

```bash
python scripts/index_catalog.py --recolor
```

Si cambiás `--n`, el catálogo debe prepararse e indexarse nuevamente.

## Ejecutar la aplicación

En una terminal, desde la raíz:

```bash
source .venv/bin/activate
uvicorn home_made_rag.main:app --reload --host 127.0.0.1 --port 8000
```

En otra terminal:

```bash
cd frontend
npm ci
npm run dev
```

Abrí <http://localhost:5173>. La interfaz tiene estas páginas:

- `/`: asistente de recomendaciones.
- `/catalogo`: estadísticas, cobertura y límites del catálogo.
- `/monitor`: requests, tiempos, candidatos y uso estimado del LLM.

El frontend usa `http://localhost:8000` por defecto. Para cambiarlo, copiá
`frontend/.env.example` como `frontend/.env` y modificá
`VITE_API_BASE_URL`.

El backend permite CORS desde `http://localhost:5173`. Si cambiás el origen,
actualizá `allow_origins` en `home_made_rag/main.py`.

## Configuración

La configuración funcional está en
[home_made_rag/config.json](home_made_rag/config.json):

- `dataset`: dataset, split y paths locales.
- `embeddings` y `vector_store`: CLIP, dispositivo y ChromaDB.
- `image_preprocessing`: eliminación de fondo y normalización de imágenes.
- `category_classifier` y `hybrid_reranker`: categoría, color y pesos del
  ranking.
- `retrieval` y `query_rewriting`: filtros, límites y reescritura de texto.
- `generator`: proveedor, modelos, límites, timeouts y tarifas estimadas.
- `type_aware_compatibility`: checkpoint, type spaces y dispositivo.
- `monitoring`: base SQLite y retención de datos.

Variables de entorno disponibles:

| Variable | Uso |
|---|---|
| `LLM_API_KEY` | Credencial principal del proxy configurado. |
| `OPENAI_API_KEY` | Credencial alternativa para un endpoint compatible con OpenAI. |
| `OPENAI_MODEL` | Sobrescribe el modelo principal. |
| `OPENAI_BASE_URL` | Sobrescribe el endpoint principal. |
| `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` | Fallback OpenRouter, si está habilitado. |
| `HF_TOKEN`, `HF_MODEL` | Fallback Hugging Face, si está habilitado. |

El backend escribe el log de ejecución en `Orag.log`.

## API

Con el backend activo, la documentación interactiva está en
<http://localhost:8000/docs>.

| Método y ruta | Función |
|---|---|
| `GET /health` | Estado del servicio y cantidad de prendas indexadas. |
| `GET /api/catalog` | Identidad y estadísticas del catálogo. |
| `POST /api/recommend` | Recomendación con `file`, `message` o ambos como multipart form. |
| `GET /api/monitor/summary` | Resumen del monitor. |
| `GET /api/monitor/requests?limit=50` | Requests recientes. |
| `GET /api/monitor/requests/{request_id}` | Detalle de un request. |
| `GET /api/monitor/requests/{request_id}/image/{variant}` | Imagen `input` o `processed` de un request. |

Ejemplo:

```bash
curl -X POST http://localhost:8000/api/recommend \
  -F 'file=@/ruta/a/prenda.jpg' \
  -F 'message=busco un look informal'
```

La respuesta incluye `recommendations`, `retrieved_outfits`,
`query_attributes`, `context_precision`, `generation_status` y `request_id`.

## Compatibilidad type-aware

El evaluador implementa *Learning Type-Aware Embeddings for Fashion
Compatibility* (ECCV 2018). Codifica cada imagen y calcula distancias para los
pares de prendas en el espacio correspondiente a sus categorías.

La respuesta contiene:

- `compatibility_score`: `1 - distancia media`; puede ser negativo y no es una
  probabilidad.
- `compatibility_similarity`: valor normalizado entre 0 y 1 mostrado en la
  interfaz; tampoco es una probabilidad.

El checkpoint incluido es una variante pública entrenada con el split disjoint
de Polyvore, con embeddings de 64 dimensiones y 66 espacios. No contiene la
cabeza de métrica aprendida, por lo que la inferencia usa distancia euclidiana.

Para repetir el benchmark:

```bash
python scripts/evaluate_type_aware_metrics.py
```

El resultado se guarda en `reports/type_aware_metrics.json` y requiere acceso
al dataset de imágenes Polyvore. La corrida registrada obtuvo AUC **0.8425**
sobre 30.290 outfits y FITB accuracy **56.31 %** sobre 15.145 preguntas. Esos
valores corresponden a este checkpoint y a distancia euclidiana; no son
directamente comparables con checkpoints nondisjoint de los autores.

## Monitor, datos y privacidad

El monitor guarda requests y metadatos en
`storage/monitoring.sqlite3`. Para consultas con foto, también puede conservar
la imagen original y la procesada en `storage/monitoring_inputs/`, durante
`monitoring.retention_days` (30 días por defecto). Desactivá `monitoring.enabled`
en `home_made_rag/config.json` si no querés guardar estos datos.

El costo del LLM que muestra el monitor es una estimación basada en las tarifas
de `generator`; no consulta saldo ni facturación real del proveedor.

No subas al repositorio `.env`, `data/`, `chroma_db/`,
`storage/monitoring.sqlite3` ni `storage/monitoring_inputs/`. El endpoint
elimina el archivo temporal de cada upload, pero el monitor puede conservar
copias según la retención configurada. Las imágenes del catálogo se sirven
localmente mediante `/images/{item_id}.jpg`.
