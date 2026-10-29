"""Réglages du projet, lus depuis le fichier .env (adresse de Qdrant, noms des modèles...)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

# Les modèles Hugging Face sont téléchargés dans le dossier du projet (et non sur le disque C:)
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

# Dossiers
PDF_DIR = ROOT / "data" / "pdfs"
IMAGE_DIR = ROOT / "data" / "images"

# Adresse de l'API FastAPI (utilisée par l'interface Streamlit)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Base vectorielle Qdrant (lancée avec Docker)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_PREFIX = os.getenv("QDRANT_PREFIX", "maintidoc")

# Noms des collections (préfixées par QDRANT_PREFIX)
TEXT_COLLECTION = os.getenv("TEXT_COLLECTION", "chunks")

# Modèle d'embeddings open source (texte -> vecteur)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# CLIP : modèle qui met les images et les textes dans le même espace de vecteurs
CLIP_MODEL = os.getenv("CLIP_MODEL", "clip-ViT-B-32")
CLIP_COLLECTION = os.getenv("CLIP_COLLECTION", "images_clip")

# Ollama : modèles open source qui tournent sur le PC
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:3b")  # décrit les images
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:4b")          # rédige les réponses
# Qwen 3 / 3.5 savent « réfléchir » avant de répondre : désactivé, car trop lent sur CPU
LLM_HAS_THINKING = LLM_MODEL.startswith("qwen3")

# Découpage du texte
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
