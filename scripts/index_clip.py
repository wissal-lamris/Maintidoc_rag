"""Indexer les images des PDF avec CLIP.

Lancement (depuis le dossier du projet) :
    python -m scripts.index_clip                             # tous les PDF de data/pdfs
    python -m scripts.index_clip data/pdfs/un_fichier.pdf    # un seul PDF
"""
import sys
from pathlib import Path

from app.clip_images import index_images_clip
from app.config import PDF_DIR

pdfs = [Path(p) for p in sys.argv[1:]] or sorted(PDF_DIR.glob("*.pdf"))
for pdf in pdfs:
    print(f"{pdf.name} : {index_images_clip(pdf)} images indexées dans la table CLIP")
