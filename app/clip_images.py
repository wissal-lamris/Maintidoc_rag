"""Représenter et rechercher les images avec CLIP.

CLIP place les images et les textes dans le même espace de vecteurs (512 dimensions) :
    image  --(CLIP image)--> vecteur ─┐
                                      ├─ proches si l'image montre ce que dit la phrase
    phrase --(CLIP texte)--> vecteur ─┘

Les vecteurs CLIP sont rangés dans une 2e collection Qdrant (« images_clip »),
séparée de celle de BGE-M3 (1024 dimensions) : on ne mélange pas deux tailles de vecteurs.
"""
from pathlib import Path

from llama_index.core.vector_stores import (FilterOperator, MetadataFilter, MetadataFilters,
                                            VectorStoreQuery)
from PIL import Image

from app.config import CLIP_MODEL, CLIP_COLLECTION, ROOT
from app.extraction import extract_images
from app.stores import make_store

_clip = None


def get_clip():
    """Modèle CLIP open source (téléchargé une seule fois depuis Hugging Face)."""
    global _clip
    if _clip is None:
        from sentence_transformers import SentenceTransformer
        _clip = SentenceTransformer(CLIP_MODEL)
    return _clip


def get_clip_store():
    """Connexion à la collection des vecteurs CLIP (créée automatiquement si elle n'existe pas)."""
    return make_store(CLIP_COLLECTION, len(get_clip().encode("test")))  # 512 pour clip-ViT-B-32


def index_images_clip(pdf_path: str | Path) -> int:
    """Extrait les images du PDF, calcule leur vecteur CLIP et les range dans Qdrant."""
    pdf_path = Path(pdf_path)
    nodes = extract_images(pdf_path)
    clip = get_clip()
    for node in nodes:
        image = Image.open(ROOT / node.metadata["image_path"]).convert("RGB")
        node.embedding = clip.encode(image).tolist()  # le vecteur vient de l'image, pas du texte

    store = get_clip_store()
    # Si ce PDF a déjà été indexé, on supprime l'ancienne version pour éviter les doublons
    store.delete_nodes(filters=MetadataFilters(filters=[
        MetadataFilter(key="source", value=pdf_path.name, operator=FilterOperator.EQ)]))
    if nodes:
        store.add(nodes)
    return len(nodes)


def search_images_clip(question: str, top_k: int = 3, source: str | None = None):
    """Retrouve les images qui ressemblent le plus à la question. Retourne (score, node).
    source : chercher seulement dans ce document (nom du PDF)."""
    query_vector = get_clip().encode(question).tolist()
    filters = None
    if source:
        filters = MetadataFilters(filters=[MetadataFilter(key="source", value=source)])
    result = get_clip_store().query(VectorStoreQuery(query_embedding=query_vector,
                                                     similarity_top_k=top_k, filters=filters))
    return list(zip(result.similarities, result.nodes))


def search_by_image(image_path: str | Path, top_k: int = 3, source: str | None = None):
    """Recherche image -> image : retrouve les figures du document qui ressemblent à une photo."""
    query_vector = get_clip().encode(Image.open(image_path).convert("RGB")).tolist()
    filters = None
    if source:
        filters = MetadataFilters(filters=[MetadataFilter(key="source", value=source)])
    result = get_clip_store().query(VectorStoreQuery(query_embedding=query_vector,
                                                     similarity_top_k=top_k, filters=filters))
    return list(zip(result.similarities, result.nodes))
