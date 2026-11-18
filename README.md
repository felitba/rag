# Fashion RAG — CLIP + ChromaDB + OpenAI

Proyecto de referencia para recibir una foto de una prenda, encontrar prendas visualmente similares con CLIP, recuperar los outfits prearmados relacionados y generar looks nuevos en español con un LLM de texto (gpt-4.1-mini vía el proxy LiteLLM de ITBA). También acepta consultas de texto, que se reescriben con el LLM antes de buscar (`query_rewriting` en `home_made_rag/config.json`).

## Arquitectura

1. **Indexación offline:** CLIP genera un embedding por imagen del catálogo y una rutina visual estima su color dominante. La categoría existente del dataset se conserva. ChromaDB guarda embeddings y los metadatos `category` y `color`.
2. **Consulta visual:** el usuario sube una imagen. Un segmentador local quita el fondo y centra el objeto sobre un fondo neutro; esa misma imagen procesada se usa para el embedding, la categoría y el color.
3. **Recuperación de outfits:** Python aplica el re-ranking híbrido a las prendas similares y las traduce a IDs de outfit. `outfits.json` es la fuente canónica para la composición de los looks.
4. **Generación opcional:** `OutfitGenerator` recibe los outfits candidatos y metadatos permitidos del catálogo, y propone nuevas combinaciones con IDs estructurados.
5. **Validación y frontend:** `RAG` valida los IDs y resuelve las prendas desde el catálogo original; React muestra las recomendaciones y explicaciones.

### Re-ranking híbrido

La primera etapa recupera hasta `vector_store.top_k` candidatos por similitud
coseno del embedding CLIP. Después se ordenan con una suma ponderada de
similitud coseno, similitud gradual de color y coincidencia de categoría. Los pesos
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
También se conservan los canales RGB medianos de los píxeles de la clase
dominante.

El reranker compara esos colores con distancia CIE Lab y transforma la distancia
en un score continuo de 0 a 1 (`exp(-ΔE / scale)`), en vez de exigir igualdad de
etiqueta. `hybrid_reranker.color_similarity.delta_e_scale` controla cuánto se
parecen los tonos distintos; un valor mayor da más crédito a diferencias de
color. El score de color combina similitud de tono (`tone_weight`, 0.75) y
similitud de cobertura (`coverage_weight`, 0.25), calculada como
`1 - |confianza_query - confianza_prenda|`. Esa confianza es el porcentaje de
píxeles de primer plano que pertenecen al color dominante. Ambos pesos se
pueden configurar. Los índices Chroma existentes no tienen todavía los canales RGB: el
reranker les asigna un color representativo de la etiqueta como fallback. Para
usar los RGB medidos por prenda, corré `python scripts/index_catalog.py` y
reconstruí el índice.

`context_precision` se calcula como AP@k en la lista recuperada y en la lista
rerankeada. La relevancia es un proxy automático: exige que coincida la
categoría estimada cuando FashionCLIP la considera confiable y que la similitud
de color supere `context_precision.color_similarity_threshold` cuando hay color
disponible. La respuesta de `/api/recommend` incluye el resultado antes y después
del reranking; el monitor también muestra qué prendas se marcaron relevantes en
cada etapa. No es ground truth ni una evaluación humana; sirve para observar
cambios de ranking con el criterio de metadata configurado.

FashionCLIP se carga de forma diferida cuando llega la primera consulta visual.
La respuesta incluye las estimaciones de categoría/color, las componentes del
ranking y una comparación orientativa con los metadatos de los garments
recuperados. Las similitudes del clasificador son relativas a las categorías
del catálogo.

La eliminación del fondo está configurada en `image_preprocessing` y usa
`rembg` con el modelo local U²-Net. El modelo se descarga la primera vez que se
procesa una foto y se reutiliza. Si la segmentación falla, la request continúa con la imagen original.
Se puede desactivar con `image_preprocessing.enabled: false`.

### Clasificación visual de la consulta

Con `category_classifier.enabled`, FashionCLIP clasifica la foto de entrada
entre las categorías existentes en el catálogo. El color de la consulta se
calcula con el estimador de píxeles descrito arriba. La respuesta de
`/api/recommend` incluye `query_attributes`; las similitudes del clasificador
son relativas a las etiquetas del catálogo, no probabilidades calibradas. Para
reducir errores como clasificar un buzo como `accessories`, se comparan varias
descripciones de prendas por categoría (incluyendo sweatshirt, sweater, hoodie
y crewneck para `tops`). Para fotos apaisadas también promedia una vista central
más cerrada con la imagen completa.

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
`https://sa-llmproxy.it.itba.edu.ar/`. También se pueden configurar fallbacks
opcionales. El primero es OpenRouter: agregá `OPENROUTER_API_KEY` a `.env`; usa
`openrouter/free` por defecto, sobrescribible con `OPENROUTER_MODEL`. Hugging Face
queda desactivado por defecto porque sus créditos gratuitos son limitados. Si
querés activarlo después de OpenRouter, configurá `HF_TOKEN`, opcionalmente
`HF_MODEL` (por defecto `Qwen/Qwen3-32B`) y
`generator.fallbacks.huggingface.enabled=true`. Se intenta el proveedor principal,
luego cada fallback habilitado. La respuesta de cada modelo se valida contra el
esquema y contra los IDs reales del catálogo antes de aceptarla. Si ningún
proveedor responde con una combinación válida, la API devuelve los outfits
recuperados.

OpenRouter usa su ruta de modelos gratuitos por defecto; esos modelos tienen
límites de uso y su disponibilidad puede variar. Hugging Face requiere un token
propio y sus créditos gratuitos son limitados. Para activar un fallback, agregá
su clave al `.env` y reiniciá el backend. Las tarifas estimadas de estos fallbacks
están en cero por defecto; actualizalas en `generator.fallbacks` si cambiás a un
modelo de pago.

El log registra tokens de entrada/salida y el costo calculado según las tarifas
por millón de tokens configuradas en `generator`. Actualizá esas tarifas cuando
cambies de modelo o modalidad de facturación.

### Evaluación de compatibilidad type-aware

Las salidas del generador también se pueden puntuar con el modelo `Learning
Type-Aware Embeddings for Fashion Compatibility` de ECCV 2018. El evaluador
recibe las imágenes individuales del catálogo, calcula la distancia en el
espacio específico de cada par de categorías y guarda en el monitor, por
outfit y por par, la distancia media y `compatibility_score = 1 - distancia`.
El score sirve para ordenar looks; no es una probabilidad calibrada.

Los artefactos están incluidos en `models/type_aware/`. Para volver a
descargarlos, verificar sus hashes y prepararlos en los paths que usa la
aplicación, corré desde la raíz:

```bash
python scripts/download_type_aware.py
```

El README de los [autores](https://github.com/mvasil/fashion-compatibility)
todavía publica el checkpoint nondisjoint, pero su enlace de Google Drive
responde 404. El descargador usa como alternativa un checkpoint público basado
en el modelo ECCV 2018, entrenado con el split disjoint ([fuente y versión
fijada](https://github.com/gyao15/fashion/tree/0a4b64b459870a0f117d2bfed054d83f9d780cbc/runs/Type_Specific_Fashion_Compatibility/origin)),
y el `typespaces.p` de ese split ([archivo y versión fijada](https://huggingface.co/datasets/Stylique/Polyvore/blob/1365152fecb3a8daee9d49c7d9d7f5809cc60b05/disjoint/typespaces.p)).
El wrapper valida que coincidan los 66 espacios y carga el checkpoint depurado
con el cargador restringido de PyTorch. La configuración está en
`home_made_rag/config.json`, bajo `type_aware_compatibility`. Si faltan los
archivos, la aplicación continúa funcionando y el monitor muestra el motivo de
disponibilidad en vez de fallar. Este checkpoint no incluye una métrica
aprendida; el wrapper calcula la distancia euclidiana entre los embeddings.
Las categorías del catálogo se normalizan a los tipos de Polyvore mediante
aliases; se pueden agregar aliases propios en `category_aliases`.

Para calcular las métricas de evaluación del paper en el split disjoint, corré:

```bash
python scripts/evaluate_type_aware_metrics.py
```

El script calcula Compatibility AUC y FITB accuracy sobre los archivos de test
del benchmark y transmite las imágenes desde `mvasil/polyvore-outfits` en
Hugging Face; necesitás acceso aprobado a ese dataset. El resultado se guarda en
`reports/type_aware_metrics.json`. Con el checkpoint disjoint instalado, la
corrida actual obtuvo **AUC 0.8425** sobre 30.290 outfits y **FITB accuracy
56.31 %** sobre 15.145 preguntas. Son resultados de esta variante con distancia
euclidiana; no son directamente comparables con el 0.88/57.6 % que el README de
los autores reporta para su checkpoint nondisjoint.

Cada outfit generado también recibe `type_aware_compatibility` en la respuesta
de `/api/recommend`; la interfaz muestra una similitud normalizada de 0 a 1 y
la cantidad de pares evaluados. Para cada par se normalizan los embeddings y se
calcula la distancia euclidiana normalizada `d` (entre 0 y 2), seguida de
`compatibility_similarity = 1 − d / 2`, que queda entre 0 y 1. El campo
`compatibility_score` conserva el score original `1 − distancia media`, que
puede ser negativo si la distancia media supera 1. Para pares puntuados en un
espacio tipo-aware, el cambio de escala conserva el orden por distancia; cuando
hay categorías desconocidas que usan el embedding general, el orden frente al
score original puede cambiar porque ese vector se vuelve a normalizar.
AUC requiere outfits positivos y
negativos etiquetados, y FITB requiere preguntas con respuesta correcta, por lo
que esos dos valores son métricas agregadas del benchmark y no scores por look.

### Monitor local

Con `monitoring.enabled`, cada llamada a `/api/recommend` registra un request ID,
estado y latencia; también guarda los candidatos y sus metadatos antes y después
del re-ranking. Si se usa el generador, registra llamadas, tokens y costo
estimado según las tarifas de `generator`, con un desglose por variable de entorno
y una huella SHA-256 truncada para distinguir claves rotadas. El historial se
guarda durante 30 días en `storage/monitoring.sqlite3`. El monitor también
guarda la foto original y la procesada de cada consulta en
`storage/monitoring_inputs/` (ignorado por Git) durante el mismo período; no
conserva claves API.

Con el frontend local iniciado, abrí `http://localhost:5173/monitor`. También
están disponibles `/api/monitor/summary`, `/api/monitor/requests` y
`/api/monitor/requests/{request_id}`. El costo es una estimación de esta
aplicación; el monitor no consulta saldo, cuota ni uso facturado en la cuenta del
proveedor.

## Estructura

```text
fashion-rag/
├── home_made_rag/       # backend FastAPI, RAG, catálogo y modelos
├── frontend/            # interfaz React + Vite
├── scripts/             # preparación del catálogo, indexación y evaluación
├── data/                # catálogo e imágenes locales (generados, ignorados por Git)
├── chroma_db/           # índice vectorial local (generado, ignorado por Git)
├── models/type_aware/   # pesos y espacios type-aware
├── storage/             # uploads y base del monitor
├── .env.example
└── requirements.txt
```

## 1. Requisitos

- Python 3.11 recomendado.
- Node.js 20 o superior para el frontend.
- Una API key del proxy LiteLLM de ITBA (o de OpenRouter como fallback).
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
Copy-Item .env.example .env
```

Editá `.env` y reemplazá `LLM_API_KEY=replace_me` por tu clave. No la guardes en React ni la subas a Git.

La instalación de CLIP se hace desde el repositorio oficial para no confundirlo con otros paquetes de Python que también se llaman `clip`.

## 3. Preparar el catálogo

Para descargar una muestra de Polyvore y guardar el catálogo en `data/`, ejecutá:

```bash
python scripts/prepare_polyvore.py --n 200
```

El tamaño de la muestra se puede cambiar con `--n`. La configuración del dataset,
las rutas locales y el split están en `home_made_rag/config.json`.

## 4. Indexar / reconstruir ChromaDB

```powershell
python scripts/index_catalog.py
```

El script valida las referencias y la existencia de todas las imágenes, estima y guarda el color faltante de cada garment, y después **reconstruye la colección completa** con embeddings CLIP y metadata de categoría/color. Si agregás, quitás o modificás fotos, volvé a ejecutarlo.

ChromaDB persiste los vectores en `chroma_db/`. No subas esa carpeta a Git; podés reconstruirla usando el catálogo.

## 5. Iniciar FastAPI

```powershell
uvicorn home_made_rag.main:app --reload --port 8000
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
| `OPENROUTER_API_KEY` | Clave opcional para el proveedor fallback OpenRouter | Opcional |
| `OPENROUTER_MODEL` | Sobrescribe el modelo OpenRouter | `openrouter/free` |
| `HF_TOKEN` | Clave opcional para activar el fallback Hugging Face | Opcional |
| `HF_MODEL` | Sobrescribe el modelo Hugging Face | `Qwen/Qwen3-32B` |
| `VITE_API_BASE_URL` | URL del backend para el frontend | `http://localhost:8000` |

El modelo CLIP, el directorio de Chroma, el top-k y los límites de resultados se
configuran en `home_made_rag/config.json`.

Si la API key falta o OpenAI falla, el backend todavía puede devolver outfits recuperados y una explicación de fallback. Instalá las dependencias con `pip install -r requirements.txt`.

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

Las prendas recuperadas se ordenan con el re-ranking híbrido y se recorren en ese orden; cada prenda aporta los outfits reales que la contienen. Un outfit toma el puntaje de la primera prenda que lo trajo y se devuelven los primeros en aparecer. **El valor no es una probabilidad de compatibilidad.**

## Consideraciones antes de producción

- El endpoint todavía no valida tamaño, formato ni dimensiones de la imagen. El archivo temporal de la subida se borra al finalizar la solicitud, pero el monitor conserva una copia (ver "Monitor local").
- El backend sirve las fotos del catálogo desde `data/images/` bajo `/images/...`. No coloques allí archivos privados.
- La clave de OpenAI se mantiene en el backend.
- En producción, agregá autenticación/rate limiting, logging estructurado, límites de concurrencia para CLIP, monitoreo de costos de OpenAI y almacenamiento de imágenes adecuado.
- El ranking visual debe validarse con fotos reales de usuarios. CLIP puede verse influido por fondo, pose, iluminación o por varias prendas visibles en una sola foto.
- El frontend formatea el precio como ARS solo a modo de demostración. Cambiá `money()` en `frontend/src/App.jsx` a la moneda real de tu tienda.
