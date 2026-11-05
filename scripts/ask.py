"""Poser des questions au RAG (Ollama doit être ouvert).

Lancement :
    python -m scripts.ask
    python -m scripts.ask "Quel est le couple de serrage de l'écrou de roue ?"
"""
import sys
import time

from app.rag import answer

questions = sys.argv[1:] or [
    "Quel est le débit nominal de la pompe HM-160 ?",
    "What is the maintenance interval for replacing the mechanical seal?",
    "Que signifie le code de défaut A02 ?",
]

for q in questions:
    start = time.time()
    result = answer(q)
    print(f"\n=== Question : {q}")
    print(f"\nRéponse ({time.time() - start:.0f} s) :\n{result['answer']}")
    print("\nSources :")
    for s in result["sources"]:
        print(f"  [{s['num']}] {s['type']:<7} | page {s['page']:<2} | {s['via']:<6} | score {s['score']:.3f}")
    for path in result["images"]:
        print(f"  Image citée : {path}")
