"""Affiche ce que LlamaIndex extrait du PDF (sans encore remplir la base).

Lancement (depuis le dossier du projet) :
    python -m scripts.test_extraction                                  # la fiche technique
    python -m scripts.test_extraction data/pdfs/un_autre_fichier.pdf
"""
import sys
from collections import Counter

from app.config import PDF_DIR
from app.extraction import extract_pdf

pdf = sys.argv[1] if len(sys.argv) > 1 else PDF_DIR / "fiche_technique_pompe_HM-160.pdf"
nodes = extract_pdf(pdf)

counts = Counter(n.metadata["type"] for n in nodes)
print(f"\nDocument : {pdf}")
print(f"Morceaux extraits : {len(nodes)}  ->  {dict(counts)}\n")

print("=== Exemple de passage de texte ===")
texte = next(n for n in nodes if n.metadata["type"] == "texte")
print(f"[page {texte.metadata['page']}] {texte.text[:300]}...\n")

for kind, title in (("tableau", "Tableaux"), ("image", "Images")):
    print(f"=== {title} ===")
    for n in nodes:
        if n.metadata["type"] == kind:
            extra = f" -> {n.metadata['image_path']}" if kind == "image" else ""
            print(f"[page {n.metadata['page']}] {n.text.splitlines()[0][:110]}{extra}")
    print()
