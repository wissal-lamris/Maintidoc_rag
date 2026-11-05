"""Chercher des images avec CLIP, à partir d'une phrase.

Lancement :
    python -m scripts.test_clip
    python -m scripts.test_clip "a diagram of water tanks and pumps"
"""
import sys

from app.clip_images import search_images_clip

questions = sys.argv[1:] or [
    "a flowchart with yes and no decisions",        # logigramme des vibrations (guide de dépannage)
    "a curve plot of flow rate and efficiency",     # courbe caractéristique (fiche technique)
    "a cross section of a centrifugal pump",        # coupe de la pompe (fiche technique)
]

for q in questions:
    print(f"\n=== Question : {q}")
    for score, node in search_images_clip(q, top_k=3):
        m = node.metadata
        print(f"  score {score:.3f} | page {m['page']} | {m['image_path']} | {m['caption'][:60]}")
