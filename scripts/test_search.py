"""Vérifier la base en cherchant les morceaux les plus proches d'une question.

Lancement :
    python -m scripts.test_search
    python -m scripts.test_search "Quel est le débit nominal de la pompe ?"
"""
import sys

from app.indexing import search

questions = sys.argv[1:] or [
    "Quel est le débit nominal de la pompe HM-160 ?",
    "Que faire en cas de fuite d'eau au niveau de l'arbre ?",
    "Montre-moi le synoptique de la station",
]

for q in questions:
    print(f"\n=== Question : {q}")
    for r in search(q, top_k=3):
        m = r.node.metadata
        extrait = " ".join(r.node.get_content().split())[:120]
        print(f"  score {r.score:.3f} | {m['type']:<7} | page {m['page']:<2} | {extrait}")
