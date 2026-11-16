# Fashion RAG — CLIP + ChromaDB + OpenAI

Proyecto de referencia para recibir una foto de una prenda, encontrar prendas visualmente similares con CLIP, recuperar los outfits prearmados relacionados y generar una explicación en español con un modelo multimodal de OpenAI.

## Arquitectura

1. **Indexación offline:** CLIP genera un embedding por imagen del catálogo y una rutina visual estima su color dominante. La categoría existente del dataset se conserva. ChromaDB guarda embeddings y los metadatos `category` y `color`.
2. **Consulta visual:** el usuario sube una imagen. CLIP recupera candidatos por embedding; el sistema estima el color de la consulta y clasifica su categoría dentro de las categorías presentes en el catálogo.
3. **Recuperación de outfits:** Python aplica el re-ranking híbrido a las prendas similares y las traduce a IDs de outfit. `outfits.json` es la fuente canónica para la composición de los looks.
4. **Generación opcional:** `OutfitGenerator` recibe los outfits candidatos y metadatos permitidos del catálogo, y propone nuevas combinaciones con IDs estructurados.
5. **Validación y frontend:** `RAG` valida los IDs y resuelve las prendas desde el catálogo original; React muestra las recomendaciones y explicaciones.

### Re-ranking híbrido

La primera etapa recupera hasta `vector_store.top_k` candidatos por similitud
coseno del embedding CLIP. Después se ordenan con una suma ponderada de
similitud coseno, coincidencia de color y coincidencia de categoría. Los pesos
iniciales están en `hybrid_reranker.weights` en `home_made_rag/config.json` y
son configurables. Si un atributo no está disponible, se omite y se
renormalizan los pesos restantes.

La categoría de cada producto viene del dataset y se conserva sin inferirla al
indexar. Para una foto de consulta, FashionCLIP elige la categoría más cercana
entre las categorías del catálogo. El color del catálogo y de la consulta se
estima con el mismo proceso: se reduce la imagen, se intenta excluir el fondo
uniforme conectado a los bordes, se asigna cada píxel restante a una paleta
HSV y se selecciona el color más frecuente. `color_confidence` es la fracción
de píxeles retenidos que cae en esa clase; no es una probabilidad calibrada.

FashionCLIP se carga de forma diferida cuando llega la primera consulta visual.
La respuesta incluye las estimaciones de categoría/color, las componentes del
ranking y una comparación orientativa con los metadatos de los garments
recuperados. Las similitudes del clasificador son relativas a las categorías
del catálogo.

### Clasificación visual de la consulta

Con `category_classifier.enabled`, FashionCLIP clasifica la foto de entrada
entre las categorías existentes en el catálogo. El color de la consulta se
calcula con el estimador de píxeles descrito arriba. La respuesta de
`/api/recommend` incluye `query_attributes`; las similitudes del clasificador
son relativas a las etiquetas del catálogo, no probabilidades calibradas.

El backend también devuelve `retrieved_metadata_matches`: compara la categoría
del dataset y el color guardado como metadata de cada garment recuperado.
Si falta alguno de esos campos, su componente no participa del score.
Al regenerar el catálogo con `prepare_polyvore.py`, se conservan categorías
taxonómicas, términos relacionados y `url_name` para facilitar esa comparación.

> El LLM propone combinaciones de IDs, pero no decide qué productos existen ni puede agregar artículos a los resultados de la tienda. El backend solo devuelve IDs pertenecientes a los outfits recuperados y conserva sus datos originales.

### Generador de outfits

Después del retrieval y del reranking, `OutfitGenerator` recibe los outfits
candidatos y metadatos permitidos del catálogo. Devuelve IDs estructurados;
`RAG` valida cada ID y resuelve las prendas desde los objetos originales del
catálogo. La configuración `generator` de `home_made_rag/config.json` controla
el modelo, la cantidad de outfits de contexto y el máximo de sugerencias.

La credencial del proxy se lee primero desde `LLM_API_KEY` (`OPENAI_API_KEY`
queda como fallback para uso directo). `OPENAI_MODEL` y `OPENAI_BASE_URL`
pueden sobrescribir el modelo y el proxy. Por defecto, el cliente usa
`https://sa-llmproxy.it.itba.edu.ar/`. El color se estima al indexar si falta
en el catálogo; `style` se omite cuando no está disponible. Si falta la key,
falla OpenAI o la
salida no pasa validación, la API devuelve los outfits recuperados.

El log registra tokens de entrada/salida y el costo calculado según las tarifas
por millón de tokens configuradas en `generator`. Actualizá esas tarifas cuando
cambies de modelo o modalidad de facturación.

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

El script valida las referencias y la existencia de todas las imágenes, estima y guarda el color faltante de cada garment, y después **reconstruye la colección completa** con embeddings CLIP y metadata de categoría/color. Si agregás, quitás o modificás fotos, volvé a ejecutarlo.

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
| `LLM_API_KEY` | Credencial del proxy LiteLLM de ITBA | Necesaria para el proxy |
| `OPENAI_API_KEY` | Credencial fallback para uso directo de OpenAI | Opcional; no reemplaza la clave del proxy |
| `OPENAI_MODEL` | Sobrescribe el modelo del generador | `gpt-4.1-mini` |
| `OPENAI_BASE_URL` | Sobrescribe la URL base del proxy | `https://sa-llmproxy.it.itba.edu.ar/` |
| `CLIP_MODEL` | Modelo de CLIP local | `ViT-B/32` |
| `CHROMA_PATH` | Directorio persistente | `storage/chroma` |
| `COLLECTION_NAME` | Colección vectorial | `fashion_garments` |
| `TOP_K_GARMENTS` | Cantidad máxima de prendas candidatas | `30` |
| `MAX_OUTFITS` | Outfits devueltos | `5` |
| `MAX_IMAGES_PER_OUTFIT_FOR_LLM` | Imágenes del catálogo enviadas al LLM por outfit | `2` |
| `MAX_UPLOAD_MB` | Tamaño máximo de foto subida | `10` |

Si la API key falta o OpenAI falla, el backend todavía puede devolver outfits recuperados y una explicación de fallback. Instalá las dependencias con `pip install -r requirements.txt`.

Ejecutá las pruebas de validación de IDs con:

```bash
python -m unittest discover -s tests
```

Ejemplo de respuesta con generación exitosa (los datos de las prendas se
resuelven siempre desde el catálogo):

```json
{
  "recommendations": [{
    "outfit_id": "generated_01",
    "name": "Capas urbanas neutras",
    "description": "Una combinación relajada para todos los días.",
    "explanation": "Las prendas crean un conjunto equilibrado.",
    "style_tags": ["urbano", "casual"],
    "source_outfits": ["210750761"],
    "generated": true,
    "items": [{"garment_id": "154249722", "category": "shoes", "title": "...", "color": null, "style": null, "price": null, "image_url": "/images/..."}]
  }],
  "explanation": "Sugerencias nuevas basadas en prendas recuperadas.",
  "generation_enabled": true,
  "generation_status": "generated"
}
```

## Cómo se calcula el ranking

Cada prenda recuperada en la posición `r` aporta `1 / (r + 1)` puntos a cada outfit al que pertenece. Después se suman esos puntos por outfit y se devuelven los de mayor puntuación. Es un baseline transparente para empezar; **el valor no es una probabilidad de compatibilidad**.

Conviene evaluarlo con fotos de prueba etiquetadas y luego ajustar el ranking, por ejemplo agregando diversidad entre outfits, filtros por categoría, rangos de precio o disponibilidad. Actualmente solo se recomiendan outfits completos en los que todas las prendas estén disponibles.

## Consideraciones antes de producción

- El endpoint comprueba tamaño, formato y dimensiones de la imagen; las imágenes subidas se normalizan y se borran al finalizar la solicitud.
- El backend sirve las fotos del catálogo por `/catalog/...`. No coloques allí archivos privados.
- La clave de OpenAI se mantiene en el backend.
- En producción, agregá autenticación/rate limiting, logging estructurado, límites de concurrencia para CLIP, monitoreo de costos de OpenAI y almacenamiento de imágenes adecuado.
- El ranking visual debe validarse con fotos reales de usuarios. CLIP puede verse influido por fondo, pose, iluminación o por varias prendas visibles en una sola foto.
- El frontend formatea el precio como ARS solo a modo de demostración. Cambiá `money()` en `frontend/src/App.jsx` a la moneda real de tu tienda.
