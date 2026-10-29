"""Base vectorielle Qdrant : deux collections, créées automatiquement si elles n'existent pas.

    « maintidoc_chunks »      : vecteurs BGE-M3 (1024) des passages, tableaux et descriptions d'images
    « maintidoc_images_clip » : vecteurs CLIP (512) des images

LlamaIndex fournit l'interface pour ajouter, chercher, filtrer et supprimer. La lecture complète
d'une collection (statistiques, inventaire, lecture d'une page) passe directement par le client Qdrant.
"""
import json

from app.config import QDRANT_PREFIX, QDRANT_URL

_qdrant = None


def qdrant_client():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        _qdrant = QdrantClient(url=QDRANT_URL, timeout=60)
    return _qdrant


def make_store(name: str, embed_dim: int):
    """Connexion à une collection Qdrant (distance cosinus), créée si elle n'existe pas."""
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import models
    client, collection = qdrant_client(), f"{QDRANT_PREFIX}_{name}"
    if not client.collection_exists(collection):
        client.create_collection(collection, vectors_config=models.VectorParams(
            size=embed_dim, distance=models.Distance.COSINE))
    return QdrantVectorStore(client=client, collection_name=collection)


def all_chunks(name: str) -> list[dict]:
    """Tous les morceaux d'une collection : [{"text": ..., "type": ..., "page": ..., "source": ...}]."""
    client, collection = qdrant_client(), f"{QDRANT_PREFIX}_{name}"
    if not client.collection_exists(collection):
        return []
    chunks, offset = [], None
    while True:
        points, offset = client.scroll(collection, limit=256, offset=offset, with_payload=True)
        for p in points:
            payload = dict(p.payload)
            text = json.loads(payload.pop("_node_content", "{}")).get("text", "")
            chunks.append({"text": text, **payload})
        if offset is None:
            return chunks
