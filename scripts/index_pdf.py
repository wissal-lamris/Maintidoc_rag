"""Indexer les PDF dans la base vectorielle (texte, tableaux, descriptions d'images).

Lancement (depuis le dossier du projet) :
    python -m scripts.index_pdf                              # tous les PDF de data/pdfs
    python -m scripts.index_pdf data/pdfs/un_fichier.pdf     # un seul PDF
"""
import sys
from pathlib import Path

from app.config import PDF_DIR
from app.indexing import index_pdf

pdfs = [Path(p) for p in sys.argv[1:]] or sorted(PDF_DIR.glob("*.pdf"))
for pdf in pdfs:
    print(f"\nIndexation de {pdf.name} ...")
    print(f"Terminé : {index_pdf(pdf)}")
