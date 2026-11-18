import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib.pyplot as plt

from home_made_rag.chroma_db import ChromaDB

method = sys.argv[1] if len(sys.argv) > 1 else "pca"

data = ChromaDB().collection.get(include=["embeddings", "metadatas"])
X = np.array(data["embeddings"])
categories = np.array([m["category"] for m in data["metadatas"]])

if method == "umap":
    import umap
    points = umap.UMAP(n_components=2, metric="cosine", random_state=42).fit_transform(X)
else:
    centered = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    points = centered @ Vt[:2].T

fig, ax = plt.subplots(figsize=(10, 8))
for i, category in enumerate(sorted(set(categories))):
    mask = categories == category
    ax.scatter(points[mask, 0], points[mask, 1], s=6, alpha=0.6,
               color=plt.cm.tab20(i), label=f"{category} ({mask.sum()})")
ax.set_title(f"Embeddings CLIP del catálogo ({method.upper()}, {len(X)} prendas)")
ax.legend(markerscale=3, fontsize=8)
fig.savefig(f"embeddings_{method}.png", dpi=200, bbox_inches="tight")
print(f"saved embeddings_{method}.png")
