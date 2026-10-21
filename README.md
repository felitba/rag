# Fashion RAG — CLIP + ChromaDB + LangChain + OpenAI

Proyecto de referencia para recibir una foto de una prenda, encontrar prendas visualmente similares con CLIP, recuperar los outfits prearmados relacionados y generar una explicación en español con un modelo multimodal de OpenAI.

## Arquitectura

1. **Indexación offline:** CLIP genera un embedding por imagen del catálogo. LangChain almacena los vectores en una colección persistente de ChromaDB junto con metadatos escalares. Los IDs de outfits se guardan como JSON dentro de los metadatos.
2. **Consulta visual:** el usuario sube una imagen. El mismo modelo CLIP genera el vector de consulta y LangChain consulta ChromaDB.
3. **Recuperación de outfits:** Python traduce las prendas similares a IDs de outfit y calcula un ranking inicial basado en la posición de recuperación. Los IDs de prendas de cada outfit vienen de `outfits.json`, que es la fuente canónica para la composición de los looks.
4. **Explicación multimodal:** LangChain construye una cadena LCEL (`RunnableLambda | ChatOpenAI | StrOutputParser`) y envía la imagen del usuario, algunas imágenes del catálogo y los metadatos de los outfits seleccionados.
5. **Frontend:** React muestra los outfits recuperados, las fotografías del catálogo y la explicación.

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
