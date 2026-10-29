"""Transformer les morceaux en vecteurs et les ranger dans la base vectorielle
Qdrant.

    PDF --(extraction.py)--> morceaux --(modèle d'embeddings)--> vecteurs --> base vectorielle
"""
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from app.config import EMBED_MODEL, TEXT_COLLECTION
from app.describe import add_descriptions
from app.extraction import extract_pdf
from app.stores import all_chunks, make_store

_embed_model = None


def get_embed_model():
    """Modèle d'embeddings open source (téléchargé une seule fois depuis Hugging Face)."""
    global _embed_model
    if _embed_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        _embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    return _embed_model


def get_vector_store():
    """Connexion à la collection des vecteurs BGE-M3 (créée automatiquement si elle n'existe pas)."""
    return make_store(TEXT_COLLECTION, len(get_embed_model().get_text_embedding("test")))


def index_pdf(pdf_path: str | Path) -> dict:
    """Extrait le PDF, calcule les vecteurs et les range dans Qdrant. Retourne un petit bilan."""
    pdf_path = Path(pdf_path)
    nodes = extract_pdf(pdf_path)
    add_descriptions(nodes)  # images : légende + description écrite par Qwen2.5-VL
    for node in nodes:
        # Seuls le type et la page sont ajoutés au texte pour le calcul du vecteur
        node.excluded_embed_metadata_keys = ["source", "caption", "image_path", "image_name", "position"]
        node.excluded_llm_metadata_keys = ["image_path", "image_name", "position"]

    vector_store = get_vector_store()
    # Si ce PDF a déjà été indexé, on supprime l'ancienne version pour éviter les doublons
    vector_store.delete_nodes(filters=MetadataFilters(filters=[
        MetadataFilter(key="source", value=pdf_path.name, operator=FilterOperator.EQ)]))

    storage = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage, embed_model=get_embed_model(), show_progress=True)

    counts = {}
    for n in nodes:
        counts[n.metadata["type"]] = counts.get(n.metadata["type"], 0) + 1
    return {"source": pdf_path.name, "total": len(nodes), **counts}


def get_index() -> VectorStoreIndex:
    """Index déjà rempli, pour faire des recherches."""
    return VectorStoreIndex.from_vector_store(get_vector_store(), embed_model=get_embed_model())


def _filters(type_filter: str | None = None, source: str | None = None):
    """Filtres sur les métadonnées : type de morceau et/ou document."""
    conditions = []
    if type_filter:
        conditions.append(MetadataFilter(key="type", value=type_filter))
    if source:
        conditions.append(MetadataFilter(key="source", value=source))
    return MetadataFilters(filters=conditions) if conditions else None


def search(question: str, top_k: int = 3, type_filter: str | None = None, source: str | None = None):
    """Retrouve les morceaux les plus proches de la question.
    type_filter : un seul type ("texte", "tableau", "image") ; source : un seul document (nom du PDF)."""
    retriever = get_index().as_retriever(similarity_top_k=top_k,
                                         filters=_filters(type_filter, source))
    return retriever.retrieve(question)


def delete_document(source: str) -> None:
    """Supprime un document : ses morceaux, ses vecteurs CLIP, ses images et le fichier PDF."""
    import shutil

    from app.clip_images import get_clip_store
    from app.config import IMAGE_DIR, PDF_DIR
    for store in (get_vector_store(), get_clip_store()):
        store.delete_nodes(filters=_filters(source=source))
    shutil.rmtree(IMAGE_DIR / Path(source).stem, ignore_errors=True)
    (PDF_DIR / source).unlink(missing_ok=True)


def base_stats() -> dict:
    """Nombre de morceaux par type et liste des documents dans la base (pour l'interface)."""
    types = {}
    chunks = all_chunks(TEXT_COLLECTION)
    for c in chunks:
        types[c["type"]] = types.get(c["type"], 0) + 1
    return {"types": types, "sources": sorted({c["source"] for c in chunks})}


def page_chunks(page: int, source: str | None = None) -> list[dict]:
    """Tous les morceaux d'une page (pour lire une page entière) : texte, puis tableaux, puis images."""
    order = {"texte": 0, "tableau": 1, "image": 2}
    chunks = [c for c in all_chunks(TEXT_COLLECTION)
              if c["page"] == page and (source is None or c["source"] == source)]
    return sorted(chunks, key=lambda c: order.get(c["type"], 3))


def inventory(source: str | None = None) -> dict:
    """Inventaire d'un document : nombre de pages et de passages, liste des tableaux et des figures."""
    metas = [c for c in all_chunks(TEXT_COLLECTION) if source is None or c["source"] == source]
    tables = sorted({(m["source"], m["page"], m.get("caption") or "Tableau sans légende")
                     for m in metas if m["type"] == "tableau"})
    figures = sorted({(m["source"], m["page"], m.get("caption") or "Image sans légende", m.get("image_name", ""))
                      for m in metas if m["type"] == "image"})
    return {"documents": sorted({m["source"] for m in metas}),
            "pages": max((m["page"] for m in metas), default=0),
            "passages": sum(m["type"] == "texte" for m in metas),
            "tableaux": tables, "figures": figures}
