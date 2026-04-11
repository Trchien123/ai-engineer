import json
import os
import numpy as np
import faiss
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from umap import UMAP

# ─── LOAD SAVED ARTIFACTS ─────────────────────────────────────────────────────

INDEX_PATH    = "traffic_signs_dino_only.index"
METADATA_PATH = "traffic_signs_metadata_dino_only.json"

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    valid_signs = json.load(f)

# Reconstruct the raw embeddings from the FAISS index
print("Reconstructing embeddings from FAISS index...")
embeddings_array = np.zeros((index.ntotal, index.d), dtype='float32')
for i in range(index.ntotal):
    embeddings_array[i] = index.reconstruct(i)

# ─── REBUILD LABELS FROM METADATA ─────────────────────────────────────────────
# Metadata has 2 entries per sign (original + 1 augmented), in order
# We infer the label by alternating original/augmented per sign group

unique_sign_ids = []
seen = {}
embedding_labels = []
sign_types = []

for sign in valid_signs:
    key = sign.get('local_image_path', '') + sign.get('name', '')
    if key not in seen:
        seen[key] = 0
    
    embedding_labels.append('original' if seen[key] == 0 else 'augmented')
    seen[key] += 1
    sign_types.append(sign.get('category', 'unknown').lower())

print(f"Loaded {index.ntotal} vectors, {len(valid_signs)} metadata entries.")

# ─── PLOT 1 & 2: EMBEDDING SPACE ──────────────────────────────────────────────

print("Running UMAP dimensionality reduction (this may take a moment)...")
reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
reduced = reducer.fit_transform(embeddings_array)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: Original vs Augmented
label_colors = {'original': '#2196F3', 'augmented': '#FF5722'}
colors = [label_colors[l] for l in embedding_labels]

axes[0].scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.5, s=10)
axes[0].set_title('Embedding Space: Original vs Augmented', fontsize=13)
axes[0].set_xlabel('UMAP-1')
axes[0].set_ylabel('UMAP-2')
patches = [mpatches.Patch(color=v, label=k) for k, v in label_colors.items()]
axes[0].legend(handles=patches)

# Plot 2: By Sign Type
unique_types = list(set(sign_types))
palette = plt.cm.get_cmap('tab10', len(unique_types))
type_color_map = {t: palette(i) for i, t in enumerate(unique_types)}
type_colors = [type_color_map[t] for t in sign_types]

axes[1].scatter(reduced[:, 0], reduced[:, 1], c=type_colors, alpha=0.5, s=10)
axes[1].set_title('Embedding Space: By category', fontsize=13)
axes[1].set_xlabel('UMAP-1')
axes[1].set_ylabel('UMAP-2')
patches2 = [mpatches.Patch(color=type_color_map[t], label=t) for t in unique_types]
axes[1].legend(handles=patches2, fontsize=8, loc='best')

plt.tight_layout()
plt.savefig('embedding_space.png', dpi=150)
plt.show()
print("Saved: embedding_space.png")

# ─── PLOT 3: RETRIEVAL SANITY CHECK ───────────────────────────────────────────

print("Running retrieval sanity check...")
sample_indices = np.random.choice(len(embeddings_array), size=3, replace=False)

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
fig.suptitle('Retrieval Check: Query → Top 3 Nearest Neighbors', fontsize=13)

for row, query_idx in enumerate(sample_indices):
    query_vec = embeddings_array[query_idx].reshape(1, -1)
    D, I = index.search(query_vec, k=4)  # k=4 to skip self

    query_sign = valid_signs[query_idx]
    query_img_path = query_sign.get('local_image_path')

    # Query image
    ax = axes[row, 0]
    if query_img_path and os.path.exists(query_img_path):
        ax.imshow(Image.open(query_img_path).convert('RGB'))
    ax.set_title(f"Query\n{query_sign.get('name', '')[:25]}", fontsize=7)
    ax.axis('off')

    # Top 3 neighbors (skip index 0 = self)
    for col, (dist, idx) in enumerate(zip(D[0][1:4], I[0][1:4])):
        retrieved_sign = valid_signs[idx]
        retrieved_img_path = retrieved_sign.get('local_image_path')
        ax = axes[row, col + 1]
        if retrieved_img_path and os.path.exists(retrieved_img_path):
            ax.imshow(Image.open(retrieved_img_path).convert('RGB'))
        ax.set_title(f"Score: {dist:.3f}\n{retrieved_sign.get('name', '')[:25]}", fontsize=7)
        ax.axis('off')

plt.tight_layout()
plt.savefig('retrieval_sanity_check.png', dpi=150)
plt.show()
print("Saved: retrieval_sanity_check.png")