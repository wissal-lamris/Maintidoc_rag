"""API de MaintiDoc (FastAPI) : tout le travail se fait ici, l'interface Streamlit ne fait qu'afficher.

    Interface Streamlit  ──HTTP──>  API FastAPI  ──>  extraction, base vectorielle, agent, Qwen

Lancement (depuis le dossier du projet, Docker et Ollama ouverts) :
    uvicorn api:app --port 8000
Documentation interactive de l'API : http://127.0.0.1:8000/docs
"""
import json
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import LLM_MODEL, PDF_DIR, ROOT, VISION_MODEL

DATA_DIR = (ROOT / "data").resolve()
UPLOAD_DIR = DATA_DIR / "uploads"  # photos jointes aux questions

app = FastAPI(title="MaintiDoc API",
              description="RAG agentique multimodal : questions sur des documents techniques PDF.")


@app.get("/sante")
def sante():
    """Vérifie que l'API tourne et indique la configuration utilisée."""
    return {"statut": "ok", "base_vectorielle": "Qdrant", "llm": LLM_MODEL, "vision": VISION_MODEL}


@app.get("/documents")
def documents():
    """Documents indexés et nombre de passages, tableaux et images dans la base."""
    from app.indexing import base_stats
    try:
        return base_stats()
    except Exception as e:
        raise HTTPException(503, f"Qdrant injoignable : {e}") from e


@app.post("/documents")
def indexer(pdf: UploadFile = File(...)):
    """Enregistre un PDF puis l'indexe : texte, tableaux, descriptions d'images (BGE-M3) et CLIP."""
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Seuls les fichiers PDF sont acceptés.")
    from app.clip_images import index_images_clip
    from app.indexing import index_pdf
    pdf_path = PDF_DIR / Path(pdf.filename).name
    pdf_path.write_bytes(pdf.file.read())
    start = time.time()
    bilan = index_pdf(pdf_path)
    bilan["images_clip"] = index_images_clip(pdf_path)
    bilan["secondes"] = round(time.time() - start)
    return bilan


@app.delete("/documents/{nom}")
def supprimer(nom: str):
    """Supprime un document : ses morceaux, ses vecteurs CLIP, ses images et le fichier PDF."""
    from app.indexing import delete_document
    delete_document(Path(nom).name)
    return {"supprime": nom}


@app.post("/questions")
def questions(question: str = Form(...), historique: str = Form("[]"), document: str = Form(""),
              photo: UploadFile | None = File(None)):
    """Pose une question à l'agent. Facultatif : historique (JSON), document ciblé, photo jointe."""
    from app.agent import ask
    photo_path = None
    if photo is not None and photo.filename:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved = UPLOAD_DIR / f"{int(time.time())}_{Path(photo.filename).name}"
        saved.write_bytes(photo.file.read())
        photo_path = saved
    start = time.time()
    result = ask(question, json.loads(historique), source=document or None,
                 photo=str(photo_path) if photo_path else None)
    result["secondes"] = round(time.time() - start)
    result["photo"] = photo_path.relative_to(ROOT).as_posix() if photo_path else None
    return result


@app.get("/fichiers/{chemin:path}")
def fichier(chemin: str):
    """Renvoie une image du dossier data/ (figures extraites ou photos jointes)."""
    path = (ROOT / chemin).resolve()
    if not path.is_relative_to(DATA_DIR) or not path.is_file() or \
            path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(404, "Fichier introuvable.")
    return FileResponse(path)
