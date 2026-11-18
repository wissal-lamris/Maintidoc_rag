"""Client de l'API : ce que l'interface Streamlit utilise pour parler à l'API FastAPI."""
import json

import requests

from app.config import API_URL

QUESTION_TIMEOUT = 900   # une réponse de l'agent peut prendre plusieurs minutes sur CPU
INDEX_TIMEOUT = 3600     # l'indexation d'un PDF avec beaucoup d'images peut être longue


class APIError(Exception):
    pass


def _call(method: str, path: str, timeout: float = 30, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=(5, timeout), **kwargs)
    except requests.ConnectionError as e:
        raise APIError(f"L'API ne répond pas sur {API_URL}. Lancez-la avec : uvicorn api:app --port 8000") from e
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise APIError(f"Erreur de l'API ({response.status_code}) : {detail}")
    return response.json()


def sante() -> dict:
    return _call("GET", "/sante", timeout=5)


def documents() -> dict:
    return _call("GET", "/documents")


def indexer(nom: str, contenu: bytes) -> dict:
    return _call("POST", "/documents", timeout=INDEX_TIMEOUT,
                 files={"pdf": (nom, contenu, "application/pdf")})


def supprimer(nom: str) -> dict:
    return _call("DELETE", f"/documents/{nom}")


def poser_question(question: str, historique: list[dict], document: str | None = None,
                   photo: tuple[str, bytes] | None = None) -> dict:
    files = {"photo": (photo[0], photo[1])} if photo else None
    data = {"question": question, "historique": json.dumps(historique), "document": document or ""}
    return _call("POST", "/questions", timeout=QUESTION_TIMEOUT, data=data, files=files)


def url_fichier(chemin: str) -> str:
    """Adresse d'une image servie par l'API (figure extraite ou photo jointe)."""
    return f"{API_URL}/fichiers/{chemin}"
